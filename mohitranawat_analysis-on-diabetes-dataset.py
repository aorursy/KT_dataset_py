import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
%matplotlib inline 
plt.style.use('fivethirtyeight')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
pima_column_names = ['times_pregnant', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness', 'serum_insulin', 'bmi', 'pedigree_function', 'age', 'onset_diabetes']
pima = pd.read_csv(r'../input/pima-indians-diabetes-database/diabetes.csv',names = pima_column_names,skiprows=1)
pima.head()


pima.info()
pima['onset_diabetes'].value_counts(normalize=True) 
col = 'plasma_glucose_concentration'
plt.figure(figsize=(10,5))
plt.hist(pima[pima['onset_diabetes']==0][col], 10, alpha=0.5, label='non-diabetes')
plt.hist(pima[pima['onset_diabetes']==1][col], 10, alpha=0.5, label='diabetes')
plt.legend(loc='upper right')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.title('Histogram of {}'.format(col))
plt.show()
for col in ['bmi', 'diastolic_blood_pressure', 'serum_insulin','triceps_thickness', 'plasma_glucose_concentration']:
    plt.figure(figsize=(8,4))
    plt.hist(pima[pima['onset_diabetes']==0][col], 10, alpha=0.5, label='non-diabetes')
    plt.hist(pima[pima['onset_diabetes']==1][col], 10, alpha=0.5, label='diabetes')
    plt.legend(loc='upper right')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title('Histogram of {}'.format(col))
    plt.show()
# look at the heatmap of the correlation matrix of our dataset
plt.figure(figsize=(12,8))
corr=round(pima.corr(),2)
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr,mask=mask, square=True, annot=True)
plt.xticks(rotation=90)
plt.show()
# plasma_glucose_concentration definitely seems to be an interesting feature here

#Following is the correlation matrix of our dataset. This is showing us the correlation amongst 
#the different columns in our Pima dataset. The output is as follows:
pima.corr()['onset_diabetes'] 
pima.describe()
pima['serum_insulin'] = pima['serum_insulin'].map(lambda x:x if x != 0 else None)
# manually replace all 0's with a None value

pima['serum_insulin'].isnull().sum()
pima.describe()
columns = ['bmi', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness']

for col in columns:
    pima[col] = pima[col].map(lambda x:x if x != 0 else None)
pima.isnull().sum()
pima.info()
pima.describe()
pima.head(5)
pima['plasma_glucose_concentration'].mean(), pima['plasma_glucose_concentration'].std()

empty_plasma_index = pima[pima['plasma_glucose_concentration'].isnull()].index
pima.loc[empty_plasma_index]['plasma_glucose_concentration']
# Will try to impute the missing values from the existing v
def relation_with_output( column ):
    temp = pima[pima[column].notnull()]
    d= temp[[column,'onset_diabetes']].groupby(['onset_diabetes'])[column].apply(lambda x: x.median()).reset_index()
    return d
#lets look relation of missing columns with onset_diabetes
relation_with_output('plasma_glucose_concentration')
relation_with_output('diastolic_blood_pressure')
relation_with_output('triceps_thickness')
relation_with_output('serum_insulin')
relation_with_output('bmi')
pima.isnull().sum()
pima.loc[(pima['onset_diabetes'] == 0 ) & (pima['serum_insulin'].isnull()), 'serum_insulin'] = 102.5
pima.loc[(pima['onset_diabetes'] == 1 ) & (pima['serum_insulin'].isnull()), 'serum_insulin'] = 169.5
pima.loc[(pima['onset_diabetes'] == 0 ) & (pima['bmi'].isnull()), 'bmi'] = 30.1
pima.loc[(pima['onset_diabetes'] == 1 ) & (pima['bmi'].isnull()), 'bmi'] = 34.3


pima.loc[(pima['onset_diabetes'] == 0 ) & (pima['triceps_thickness'].isnull()), 'triceps_thickness'] = 27.0
pima.loc[(pima['onset_diabetes'] == 1 ) & (pima['triceps_thickness'].isnull()), 'triceps_thickness'] = 32.0


pima.loc[(pima['onset_diabetes'] == 0 ) & (pima['diastolic_blood_pressure'].isnull()), 'diastolic_blood_pressure'] = 70.0
pima.loc[(pima['onset_diabetes'] == 1 ) & (pima['diastolic_blood_pressure'].isnull()), 'diastolic_blood_pressure'] = 75.0
pima.loc[(pima['onset_diabetes'] == 0 ) & (pima['plasma_glucose_concentration'].isnull()), 'plasma_glucose_concentration'] = 107.0
pima.loc[(pima['onset_diabetes'] == 1 ) & (pima['plasma_glucose_concentration'].isnull()), 'plasma_glucose_concentration'] = 140.0

# fill the column's missing values with the mean of the rest of the column
#pima['plasma_glucose_concentration'].fillna(pima['plasma_glucose_concentration'].mean(), inplace=True)
pima.isnull().sum()
X = pima.loc[:,:'age']
y = pima['onset_diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
pima.hist(figsize=(15, 15))
plt.show()
pima.info()

pima.hist(figsize=(15, 15), sharex=True)
plt.show()
print (pima['plasma_glucose_concentration'].head())
# get the mean of the column
mu = pima['plasma_glucose_concentration'].mean()

# get the standard deviation of the column
sigma = pima['plasma_glucose_concentration'].std()

# calculate z scores for every value in the column.
print (((pima['plasma_glucose_concentration'] - mu) / sigma).head())
# mean and std before z score standardizing
pima['plasma_glucose_concentration'].mean(), pima['plasma_glucose_concentration'].std()

(121.68676277850591, 30.435948867207657)


ax = pima['plasma_glucose_concentration'].hist()
ax.set_title('Distribution of plasma_glucose_concentration')
scaler = StandardScaler()

glucose_z_score_standardized = scaler.fit_transform(pima[['plasma_glucose_concentration']])
glucose_z_score_standardized.mean(), glucose_z_score_standardized.std()
ax = pd.Series(glucose_z_score_standardized.reshape(-1,)).hist()
ax.set_title('Distribution of plasma_glucose_concentration after Z Score Scaling')
scale = StandardScaler() # instantiate a z-scaler object

pima_scaled = pd.DataFrame(scale.fit_transform(pima), columns=pima_column_names)
pima_scaled.hist(figsize=(15, 15), sharex=True)
plt.show()
mean_impute_standardize = Pipeline([('imputer', SimpleImputer()), ('standardize', StandardScaler()), ('classify', knn)])
X = pima.drop('onset_diabetes', axis=1)
y = pima['onset_diabetes']

knn_params = {'imputer__strategy':['mean', 'median'], 'classify__n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)
grid = GridSearchCV(mean_impute_standardize, knn_params)
grid.fit(X, y)

print (grid.best_score_, grid.best_params_)
min_max = MinMaxScaler()
pima_min_maxed = pd.DataFrame(min_max.fit_transform(pima), columns=pima_column_names)
pima_min_maxed.describe()
mean_impute_standardize = Pipeline([('imputer', SimpleImputer()), ('standardize', MinMaxScaler()), ('classify', knn)])
X = pima.drop('onset_diabetes', axis=1)
y = pima['onset_diabetes']

knn_params = {'imputer__strategy': ['mean', 'median'], 'classify__n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
grid = GridSearchCV(mean_impute_standardize, knn_params)
grid.fit(X, y)

print (grid.best_score_, grid.best_params_)
np.sqrt((pima**2).sum(axis=1)).mean() 
# average vector length of imputed matrix
normalize = Normalizer()
pima_normalized = pd.DataFrame(normalize.fit_transform(pima), columns=pima_column_names)
np.sqrt((pima_normalized**2).sum(axis=1)).mean()
# average vector length of row normalized imputed matrix
mean_impute_normalize = Pipeline([('imputer', SimpleImputer()), ('normalize', Normalizer()), ('classify', knn)])
X = pima.drop('onset_diabetes', axis=1)
y = pima['onset_diabetes']

knn_params = {'imputer__strategy': ['mean', 'median'], 'classify__n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
grid = GridSearchCV(mean_impute_normalize, knn_params)
grid.fit(X, y)

print (grid.best_score_, grid.best_params_)
def run_model(model,hyp,X,y,cv, Scaler):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=3)
    mean_impute_standardize = Pipeline([('imputer',SimpleImputer()),
                                       ('standardize_values',Scaler),
                                       ('classification',model)])
    
    grid = GridSearchCV(mean_impute_standardize,hyp,cv=cv)
    grid.fit(X_train,y_train)
    pred = grid.best_estimator_.predict(X_test)
    print(grid.best_params_)
    print(grid.best_estimator_)
    return metrics.accuracy_score(pred,y_test)
hyper_parameters = {'classification__penalty':['l1','l2'],'imputer__strategy':['mean','median']}
print('Logistic Regression accuracy: ')
run_model(LogisticRegression(solver='liblinear'),hyper_parameters,
          pima.drop(labels='onset_diabetes',axis=1),pima['onset_diabetes'],3,MinMaxScaler())
hyper_parameters = {'classification__penalty':['l1','l2'],'imputer__strategy':['mean','median']}
print('Logistic Regression accuracy: ')
run_model(LogisticRegression(solver='liblinear'),hyper_parameters,
          pima.drop(labels='onset_diabetes',axis=1),pima['onset_diabetes'],3,StandardScaler())
hyper_parameters = {'classification__criterion':['gini','entropy'],
                   'classification__n_estimators':[40,50,100,150,200],
                   'imputer__strategy':['mean','median']}
print('RandomForest Accuracy: ')
run_model(RandomForestClassifier(n_jobs=-1),hyper_parameters,
         pima.drop(labels='onset_diabetes',axis=1),pima['onset_diabetes'],3,MinMaxScaler())
hyper_parameters = {'classification__criterion':['gini','entropy'],
                   'classification__n_estimators':[40,50,100,150,200],
                   'imputer__strategy':['mean','median']}
print('RandomForest Accuracy: ')
run_model(RandomForestClassifier(n_jobs=-1),hyper_parameters,
         pima.drop(labels='onset_diabetes',axis=1),pima['onset_diabetes'],3,StandardScaler())
hyper_parameters = {'classification__kernel':['rbf','sigmoid','poly'],
                   'classification__C':[0.1,0.001,0.3,1],
                   'imputer__strategy':['mean','median']}
print('SupportVectorClassifier Accuracy: ')
run_model(SVC(),hyper_parameters,
         pima.drop(labels='onset_diabetes',axis=1),pima['onset_diabetes'],3,MinMaxScaler())
hyper_parameters = {'classification__kernel':['rbf','sigmoid','poly'],
                   'classification__C':[0.1,0.001,0.3,1],
                   'imputer__strategy':['mean','median']}
print('SupportVectorClassifier Accuracy: ')
run_model(SVC(),hyper_parameters,
         pima.drop(labels='onset_diabetes',axis=1),pima['onset_diabetes'],3,StandardScaler())
hyper_parameters = {'classification__p':[1.3,1.5,2],
                   'classification__n_neighbors':[5,7,8,9],
                   'classification__weights':['uniform','distance'],
                    'imputer__strategy':['mean','median']}
print('KNeighborClassifier accuracy: ')
run_model(KNeighborsClassifier(n_jobs=-1),hyper_parameters,
         pima.drop(labels='onset_diabetes',axis=1),pima['onset_diabetes'],3,MinMaxScaler())
hyper_parameters = {'classification__p':[1.3,1.5,2],
                   'classification__n_neighbors':[5,7,8,9],
                   'classification__weights':['uniform','distance'],
                    'imputer__strategy':['mean','median']}
print('KNeighborClassifier accuracy: ')
run_model(KNeighborsClassifier(n_jobs=-1),hyper_parameters,
         pima.drop(labels='onset_diabetes',axis=1),pima['onset_diabetes'],3,StandardScaler())
hyper_parameters = {'classification__learning_rate':[0.1,0.3,0.6,1],
                   'classification__n_estimators':[30,50,80,100],
                   'imputer__strategy':['mean','median']}
print('AdaBoostClassifier accuracy: ')
run_model(AdaBoostClassifier(),hyper_parameters,
         pima.drop(labels='onset_diabetes',axis=1),pima['onset_diabetes'],3,MinMaxScaler())
hyper_parameters = {'classification__learning_rate':[0.1,0.3,0.6,1],
                   'classification__n_estimators':[30,50,80,100],
                   'imputer__strategy':['mean','median']}
print('AdaBoostClassifier accuracy: ')
run_model(AdaBoostClassifier(),hyper_parameters,
         pima.drop(labels='onset_diabetes',axis=1),pima['onset_diabetes'],3,StandardScaler())
hyper_parameters = {'imputer__strategy':['mean','median'],
                   'classification__learning_rate':[0.1,0.3,0.5,1],
                    'classification__max_depth':[3,6,8],
                    'classification__n_estimators':[30,60,100,150]
                   }
print('GradientBoostingClassifier accuracy: ')
run_model(GradientBoostingClassifier(),hyper_parameters,
         pima.drop(labels='onset_diabetes',axis=1),pima['onset_diabetes'],3,MinMaxScaler())
hyper_parameters = {'imputer__strategy':['mean','median'],
                   'classification__learning_rate':[0.1,0.3,0.5,1],
                    'classification__max_depth':[3,6,8],
                    'classification__n_estimators':[30,60,100,150]
                   }
print('GradientBoostingClassifier accuracy: ')
run_model(GradientBoostingClassifier(),hyper_parameters,
         pima.drop(labels='onset_diabetes',axis=1),pima['onset_diabetes'],3,StandardScaler())
