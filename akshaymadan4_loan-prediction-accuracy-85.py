import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline 
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set(font_scale=1.5)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import xgboost as xgb

df_train=pd.read_csv(r'../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df_test= pd.read_csv(r'../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')
                     
df_train.head()
df_train.describe()
%matplotlib inline
import matplotlib.pyplot as plt
df_train.hist(bins=10, figsize=(20,15))
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df_train, df_train['Loan_Status']):
    strat_train_set = df_train.loc[train_index]
    strat_test_set = df_train.loc[test_index]
strat_test_set['Loan_Status'].value_counts()/ len(strat_test_set)
df_train['Loan_Status'].value_counts()/len(df_train)
sns.FacetGrid(df_train,hue="Loan_Status",height=5).map(plt.scatter,"ApplicantIncome","LoanAmount").add_legend();
plt.show()
corr=df_train.corr

sns.set(style="white")

corr = df_train.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(15, 10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

plt.title('Correlation Matrix', fontsize=18)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show()
df_train.columns
# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',]
scatter_matrix(df_train[attributes], figsize=(10, 6))

strat_train_set.isnull().sum()
strat_train_set.dtypes
X_train=strat_train_set.drop("Loan_Status",axis=1)
y_train=strat_train_set['Loan_Status'].copy()
df_train['Loan_Amount_Term'].value_counts()
X_train['Loan_Amount_Term_Cat'] = pd.cut(x=X_train['Loan_Amount_Term'], bins=[6,119,239,359,480])
X_train['Loan_Amount_Term_Cat'].value_counts()
X_train.head()
X_train.columns
X_train_Num = X_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]
X_train_Cat = X_train[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Credit_History', 'Property_Area',
       'Loan_Amount_Term_Cat']]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

X_train_Num_tr = num_pipeline.fit_transform(X_train_Num)
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse=False)

Cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('cat_encoder', OneHotEncoder(sparse=False)),
     ])
X_train_cat_tr = Cat_pipeline.fit_transform(X_train_Cat)
from sklearn.compose import ColumnTransformer

num_attribs = list(X_train_Num)
cat_attribs = list(X_train_Cat)
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", Cat_pipeline, cat_attribs),
    ])

X_train_prepared = full_pipeline.fit_transform(X_train)
X_train_prepared.shape
y_train.replace(["Y","N"],[1,0],inplace=True)
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train_prepared, y_train)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
p1=forest_clf.predict(X_train_prepared)
print(confusion_matrix(y_train,p1))
print("Accuracy Score {}".format(accuracy_score(y_train,p1)))
print("Classification report: {}".format(classification_report(y_train,p1)))
X_test=strat_test_set.drop("Loan_Status",axis=1)
y_test=strat_test_set['Loan_Status'].copy()
X_test['Loan_Amount_Term_Cat'] = pd.cut(x=X_test['Loan_Amount_Term'], bins=[6,119,239,359,480])
X_test_Num = X_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]
X_test_Cat = X_test[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Credit_History', 'Property_Area',
       'Loan_Amount_Term_Cat']]
num_pipeline2 = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

X_test_Num_tr = num_pipeline2.fit_transform(X_test_Num)
Cat_pipeline2 = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('cat_encoder', OneHotEncoder(sparse=False)),
     ])
X_test_cat_tr = Cat_pipeline2.fit_transform(X_test_Cat)
num_attribs2 = list(X_test_Num)
cat_attribs2 = list(X_test_Cat)
full_pipeline = ColumnTransformer([
        ("num", num_pipeline2, num_attribs2),
        ("cat", Cat_pipeline2, cat_attribs2),
    ])

X_test_prepared = full_pipeline.fit_transform(X_test)
X_test_prepared.shape
X_test.shape
y_test.replace(["Y","N"],[1,0],inplace=True)
p2=forest_clf.predict(X_test_prepared)
print(confusion_matrix(y_test,p2))
print("Accuracy Score {}".format(accuracy_score(y_test,p2)))
print("Classification report: {}".format(classification_report(y_test,p2)))
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)
rf=RandomForestClassifier()
rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,
                               random_state=100,n_jobs=-1)
### fit the randomized model
rf_randomcv.fit(X_train_prepared,y_train)
rf_randomcv.best_params_
rf_randomcv.best_estimator_
best_random_grid=rf_randomcv.best_estimator_
p3=best_random_grid.predict(X_test_prepared)
print(confusion_matrix(y_test,p3))
print("Accuracy Score {}".format(accuracy_score(y_test,p3)))
print("Classification report: {}".format(classification_report(y_test,p3)))
rf_randomcv.best_params_
## Defining  a range around the best parameters from the Random Search CV
from sklearn.model_selection import GridSearchCV

param_grid1 = {
    'criterion': [rf_randomcv.best_params_['criterion']],
    'max_depth': [rf_randomcv.best_params_['max_depth']],
    'max_features': [rf_randomcv.best_params_['max_features']],
    'min_samples_leaf': [rf_randomcv.best_params_['min_samples_leaf'], 
                         rf_randomcv.best_params_['min_samples_leaf']+2, 
                         rf_randomcv.best_params_['min_samples_leaf'] + 4],
    'min_samples_split': [rf_randomcv.best_params_['min_samples_split'] - 2,
                          rf_randomcv.best_params_['min_samples_split'] - 1,
                          rf_randomcv.best_params_['min_samples_split'], 
                          rf_randomcv.best_params_['min_samples_split'] +1,
                          rf_randomcv.best_params_['min_samples_split'] + 2],
    'n_estimators': [rf_randomcv.best_params_['n_estimators'] - 200, rf_randomcv.best_params_['n_estimators'] - 100, 
                     rf_randomcv.best_params_['n_estimators'], 
                     rf_randomcv.best_params_['n_estimators'] + 100, rf_randomcv.best_params_['n_estimators'] + 200]
}

print(param_grid1)
rfc=RandomForestClassifier()
grid_search=GridSearchCV(estimator=rfc,param_grid=param_grid1,cv=10,n_jobs=-1,verbose=2)
grid_search.fit(X_train_prepared,y_train)
best_grid=grid_search.best_estimator_
p4=best_grid.predict(X_test_prepared)
print(confusion_matrix(y_test,p4))
print("Accuracy Score {}".format(accuracy_score(y_test,p4)))
print("Classification report: {}".format(classification_report(y_test,p4)))
import pickle
# open a file, where you ant to store the data
file = open('RFbestGridLat.pkl', 'wb')

# dump information to that file
pickle.dump(best_grid, file)

from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
        'max_depth': hp.quniform('max_depth', 10, 1200, 10),
        'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
        'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
        'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200,1300,1500])
    }
space
def objective(space):
    model = RandomForestClassifier(criterion = space['criterion'], max_depth = space['max_depth'],
                                 max_features = space['max_features'],
                                 min_samples_leaf = space['min_samples_leaf'],
                                 min_samples_split = space['min_samples_split'],
                                 n_estimators = space['n_estimators'], 
                                 )
    
    accuracy = cross_val_score(model, X_train_prepared, y_train, cv = 5).mean()

    # We aim to maximize accuracy, therefore we return it as a negative value
    return {'loss': -accuracy, 'status': STATUS_OK }
from sklearn.model_selection import cross_val_score
trials = Trials()
best = fmin(fn= objective,
            space= space,
            algo= tpe.suggest,
            max_evals = 80,
            trials= trials)
best
crit = {0: 'entropy', 1: 'gini'}
feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200,5:1300,6:1500}


print(crit[best['criterion']])
print(feat[best['max_features']])
print(est[best['n_estimators']])
best['min_samples_leaf']
trainedforest = RandomForestClassifier(criterion = crit[best['criterion']], max_depth = best['max_depth'], 
                                       max_features = feat[best['max_features']], 
                                       min_samples_leaf = best['min_samples_leaf'], 
                                       min_samples_split = best['min_samples_split'], 
                                       n_estimators = est[best['n_estimators']]).fit(X_train_prepared,y_train)
predictionforest = trainedforest.predict(X_test_prepared)
print(confusion_matrix(y_test,predictionforest))
print(accuracy_score(y_test,predictionforest))
print(classification_report(y_test,predictionforest))
acc5 = accuracy_score(y_test,predictionforest)
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
param = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(param)
from tpot import TPOTClassifier
tpot_classifier = TPOTClassifier(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,
                                 config_dict={'sklearn.ensemble.RandomForestClassifier': param}, 
                                 cv = 4, scoring = 'accuracy')
tpot_classifier.fit(X_train_prepared,y_train)
accuracy = tpot_classifier.score(X_test_prepared, y_test)
print(accuracy)