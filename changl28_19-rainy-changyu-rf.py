# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/sp20-csestat-416-hw-5/seattle_rain_train.csv')
df.tail(10)
import seaborn as sns 
sns.countplot(x='TMRW_RAIN', data=df)
features = [
    'PRCP', 'TMAX', 'TMIN', 'RAIN', 'TMIDR', 'TRANGE', 'MONTH',
       'SEASON', 'YEST_RAIN', 'YEST_PRCP',
       'SUM7_PRCP', 'SUM14_PRCP', 'SUM30_PRCP'
]

target = 'TMRW_RAIN'                  # prediction target (y) (+1 means rain, 0 is not rain)

# Extract the feature columns and target column
df = df[features + [target]]
df.head()
from sklearn.model_selection import train_test_split

train_data, validation_data = train_test_split(df, test_size=0.2)
from sklearn.preprocessing import StandardScaler # scale the features
from sklearn.linear_model import LogisticRegression # import logistic regression model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# Next standardize the dataset
data = df.copy()
ohe1 = preprocessing.OneHotEncoder()
print(data['SEASON'])
ohe1.fit(data[['SEASON']])
tmp=ohe1.transform(data[['SEASON']]).toarray()
data['SPRING']=tmp[:,0]
data['SUMMER']=tmp[:,1]
data['AUTUMN']=tmp[:,2]
data['WINTER']=tmp[:,3]
data.head()

ohe2 = preprocessing.OneHotEncoder()
ohe2 = ohe2.fit(data[["MONTH"]])
tmp= ohe2.transform(data[["MONTH"]]).toarray()
data['JAN']=tmp[:,0]
data['FEB']=tmp[:,1]
data['MAR']=tmp[:,2]
data['APR']=tmp[:,3]
data['MAY']=tmp[:,4]
data['JUN']=tmp[:,5]
data['JUL']=tmp[:,6]
data['AUG']=tmp[:,7]
data['SEP']=tmp[:,8]
data['OCT']=tmp[:,9]
data['NOV']=tmp[:,10]
data['DEC']=tmp[:,11]

fea = ['PRCP', 'TMAX', 'TMIN', 'RAIN', 'TMIDR', 'TRANGE','YEST_RAIN', 'YEST_PRCP',
           'SUM7_PRCP', 'SUM14_PRCP', 'SUM30_PRCP', 'SPRING','SUMMER','WINTER','AUTUMN',
           'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
data.head()
data[fea]
scaler = StandardScaler().fit(data[fea])
data[fea] = scaler.transform(data[fea])
data.head()
train, val = train_test_split(data, test_size=0.1,random_state = 1)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid={"C":np.logspace(-4,1,30)}
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(train[fea],train[target])
logreg_cv.predict(val[fea])
from ipywidgets import interactive, fixed
import sklearn.metrics as metrics
probas_pred = logreg_cv.predict_proba(val[fea])[:,1] 

fpr, tpr, thresholds = metrics.roc_curve(val[target], probas_pred)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Logistic regression')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_data[features], train_data[target])
rf_random.best_params_
best_random = rf_random.best_estimator_
random_accuracy = accuracy_score(best_random.predict(validation_data[features]), validation_data[target])
random_accuracy
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [False],
    'max_depth': [7, 9, 10, 11, 13],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'min_samples_split': [8, 9, 10, 11, 12],
    'n_estimators': [500, 1000, 2000, 2500]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
grid_search.fit(train_data[features], train_data[target])
grid_search.best_params_
best_grid = grid_search.best_estimator_
grid_accuracy = accuracy_score(best_grid.predict(validation_data[features]), validation_data[target])
grid_accuracy
# First calculate the accuracies for each depth
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt # for graphing
depths = list(range(1, 50, 2))
#dt_accuracies = []
rf_accuracies = []

for i in depths:
    print(f'Depth {i}')
    rf = RandomForestClassifier(max_depth=i)
    rf.fit(train_data[features], train_data[target])

#     dt_accuracies.append((
#         accuracy_score(tree.predict(train_data[features]), train_data[target]),
#         accuracy_score(tree.predict(validation_data[features]), validation_data[target])
#     ))
    
    
#     rf = RandomForest416(15, max_depth=i)
#     rf.fit(train_data[features], train_data[target])
    
    rf_accuracies.append((     
        accuracy_score(rf.predict(train_data[features]), train_data[target]),
        accuracy_score(rf.predict(validation_data[features]), validation_data[target])
    ))
    
# Then plot them 
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

#axs[0].plot(depths, [acc[0] for acc in dt_accuracies], label='DecisionTree')
axs[0].plot(depths, [acc[0] for acc in rf_accuracies], label='RandomForest')

#axs[1].plot(depths, [acc[1] for acc in dt_accuracies], label='DecisionTree')
axs[1].plot(depths, [acc[1] for acc in rf_accuracies], label='RandomForest')

# Customize plots
axs[0].set_title('Train Data')
axs[1].set_title('Validation Data')
for ax in axs:
    ax.legend()
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Accuracy')
from sklearn.model_selection import GridSearchCV
hyperparameters = {'max_depth':[1,5,10,15,20], 'min_samples_leaf':[1, 10,50,100,200,300]}
tree = DecisionTreeClassifier()
search = GridSearchCV(tree, hyperparameters, cv=6,return_train_score=True).fit(train_data[features], train_data['safe_loans'])
df_test.head()
df_test = pd.read_csv('/kaggle/input/sp20-csestat-416-hw-5/seattle_rain_test.csv')

# Any code to pre-process the data 
...
# The code to make the predictions on the test data 
# (likely something like model.predict(...))
predictions = best_random.predict(df_test[features])
predictions = np.around(predictions)
prediction = predictions.astype(int)

to_save = df_test[['Id']].copy()
to_save.loc[:, 'Category'] = predictions
to_save.to_csv('submission.csv', index=False)