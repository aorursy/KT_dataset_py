import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pandas import Series, DataFrame

from sklearn.model_selection import cross_val_score

import os

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder



%matplotlib inline



pd.set_option('display.max_columns', 100)
df = pd.read_csv('/kaggle/input/data-mining-assignment-2/train.csv')
df.shape
df.head()
df.isnull().sum()
df.corr()
X = df.iloc[0:700, 1:65]

X.head()
y = df["Class"].copy()

y.head()
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

classes = ['3', '2', '0', '1']

ax.bar(classes,y.value_counts())

plt.show()
for i in range(0,59):

    X[X.columns[i]] = X[X.columns[i]].astype('category')

    X[X.columns[i]] = X[X.columns[i]].cat.codes

    X[X.columns[i]] = X[X.columns[i]].astype('float64')

X.dtypes
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X_scaled = scaler.fit_transform(X) 

X_scaled
corr = df.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(12, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
from numpy import loadtxt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 30)
## reading files

train = pd.read_csv('/kaggle/input/data-mining-assignment-2/train.csv')

test = pd.read_csv('/kaggle/input/data-mining-assignment-2/test.csv')



#### preprocessing ####



## missing values

for i in train.columns:

    if train[i].dtype == 'object':

      train[i] = train[i].fillna(train[i].mode().iloc[0])

    if (train[i].dtype == 'int' or train[i].dtype == 'float'):

      train[i] = train[i].fillna(np.mean(train[i]))





for i in test.columns:

    if test[i].dtype == 'object':

      test[i] = test[i].fillna(test[i].mode().iloc[0])

    if (test[i].dtype == 'int' or test[i].dtype == 'float'):

      test[i] = test[i].fillna(np.mean(test[i]))



## label encoding

number = LabelEncoder()

for i in train.columns:

    if (train[i].dtype == 'object'):

      train[i] = number.fit_transform(train[i].astype('str'))

      train[i] = train[i].astype('object')



for i in test.columns:

    if (test[i].dtype == 'object'):

      test[i] = number.fit_transform(test[i].astype('str'))

      test[i] = test[i].astype('object')



## creating a new feature origin

train['origin'] = 0

test['origin'] = 1

training = train.drop('Class',axis=1) #droping target variable



## taking sample from training and test data

training = training.sample(700, random_state=12)

testing = test.sample(300, random_state=11)



## combining random samples

combi = training.append(testing)

y = combi['origin']

combi.drop('origin',axis=1,inplace=True)



## modelling

model = RandomForestClassifier(n_estimators = 100, max_depth = 3)

drop_list = []

for i in combi.columns:

    score = cross_val_score(model,pd.DataFrame(combi[i]),y,cv=2,scoring='roc_auc')

    if (np.mean(score) > 0.8):

        drop_list.append(i)

        print(i,np.mean(score))
# using a basic model with all the features

training = train.drop('origin',axis=1)

testing = test.drop('origin',axis=1)



rf = RandomForestClassifier(n_estimators=200, max_depth=6,max_features=10)

rf.fit(training.drop('Class',axis=1),training['Class'])

pred = rf.predict(testing)

columns = ['Class']

sub = pd.DataFrame(data=pred,columns=columns)

sub['ID'] = test['ID']

sub = sub[['ID','Class']]
### plotting importances

features = training.drop('Class',axis=1).columns.values

imp = rf.feature_importances_

indices = np.argsort(imp)[::-1][:63]



#plot

plt.figure(figsize=(18,5))

plt.bar(range(len(indices)), imp[indices], color = 'b', align='center')

plt.xticks(range(len(indices)), features[indices], rotation='vertical')

plt.xlim([-1,len(indices)])

plt.show()
drop_list
## dropping drifting features which are not important.

drift_train = training.drop(['ID', 'col49', 'col59'], axis=1)

drift_test = testing.drop(['ID', 'col49', 'col59'], axis=1)



rf = RandomForestClassifier(n_estimators=100, min_samples_split=3)

rf.fit(drift_train.drop('Class',axis=1),training['Class'])

pred = rf.predict(drift_test)

columns = ['Class']

sub = pd.DataFrame(data=pred,columns=columns)

sub['ID'] = test['ID']

sub = sub[['ID','Class']]
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(random_state = 42)

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



rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)

predictions = rf_random.predict(X_test)

errors = abs(predictions - y_test)

mape = 100 * np.mean(errors / y_test)

accuracy = 100 - mape

print('Model Performance')

print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

print('Accuracy = {:0.2f}%.'.format(accuracy))
best_random = rf_random.best_estimator_

predictions = best_random.predict(X_test)

errors = abs(predictions - y_test)

mape = 100 * np.mean(errors / y_test)

accuracy = 100 - mape

print('Model Performance')

print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

print('Accuracy = {:0.2f}%.'.format(accuracy))
from sklearn.ensemble import RandomForestClassifier as RFC



rfc_b = RFC(n_estimators=100, min_samples_split=3)

rfc_b.fit(X_train,y_train)

y_pred = rfc_b.predict(X_train)



print('Train accuracy score:',accuracy_score(y_train,y_pred))

print('Test accuracy score:', accuracy_score(y_test,rfc_b.predict(X_test)))

df1 = pd.read_csv('/kaggle/input/data-mining-assignment-2/test.csv')
df1.shape
df1.isnull().sum()
X_pred = df1.iloc[0:700, 1:65]

X_pred.head()
for i in range(0,59):

    X_pred[X_pred.columns[i]] = X_pred[X_pred.columns[i]].astype('category')

    X_pred[X_pred.columns[i]] = X_pred[X_pred.columns[i]].cat.codes

    X_pred[X_pred.columns[i]] = X_pred[X_pred.columns[i]].astype('float64')
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X_pred_scaled = scaler.fit_transform(X_pred) 



X_pred_scaled.shape
y_pred =  rfc_b.predict(X_pred_scaled)

predictions = [round(value) for value in y_pred]

submission = pd.DataFrame({'ID':df1['ID'],'Class':predictions})

submission.shape
filename = 'predictions1.csv'



submission.to_csv(filename,index=False)
from IPython.display import HTML 

import pandas as pd 

import numpy as np 

import base64 

def create_download_link(df, title = "Download CSV file", filename = "final_data.csv"): 

    csv = df.to_csv(index=False)     

    b64 = base64.b64encode(csv.encode())     

    payload = b64.decode()     

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'     

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html) 

create_download_link(submission)