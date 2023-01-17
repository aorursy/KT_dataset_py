import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")

test = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
data.isnull().sum()
data.fillna(data.mean(),inplace=True)

test.fillna(data.mean(),inplace=True)
data.drop_duplicates(inplace=True)
y=data['rating']

x=data.drop(['rating'],axis=1)

x.head()
data= pd.get_dummies(data, prefix='type', columns=['type'])

test= pd.get_dummies(test,prefix='type', columns=['type'])
test_ids = test['id']

data.drop(['id'],axis =1,inplace=True)

test.drop(['id'],axis =1,inplace=True)
y = data['rating']

x = data.drop('rating',axis = 1)
x.columns
cols= list(x.columns)

cols.remove('type_old')

cols.remove('type_new')
test_scaled = test.copy()
test_scaled.columns
#from sklearn import preprocessing

#s_scaler = preprocessing.StandardScaler()

#x[cols] = pd.DataFrame(s_scaler.fit_transform(x[cols]))

#test_scaled[cols] = pd.DataFrame(s_scaler.transform(test[cols]))
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 0,stratify = y)
# counts = dict(y_train.value_counts())

# temp_df = x_train.copy()

# temp_df['y']=y_train

# # Divide by class

# df_classes = []

# for i in range(7):

#     df_classes.append(temp_df[temp_df['y']==i])

                             

# df_train_over = df_classes[3]

# for i in range(7):

#     if(i!=3):

#         df_train_over = pd.concat([df_train_over,df_classes[i].sample(counts[3], replace=True)], axis=0)

# y_train = df_train_over['y']

# x_train = df_train_over.drop('y',axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
np.random.seed(2000)

from sklearn.naive_bayes import GaussianNB as NB
y_train.value_counts()
nb = NB()

nb.fit(x_train,y_train)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_val,nb.predict(x_val)))
score_train_RF = []

score_test_RF = []



for i in range(1590,1591,1):

    rf = RandomForestClassifier(n_estimators=i,max_depth = 60, random_state = 42)

    rf.fit(x_train, y_train)

    sc_train = np.sqrt(mean_squared_error(y_train,rf.predict(x_train)))

    score_train_RF.append(sc_train)

    sc_test = np.sqrt(mean_squared_error(y_val,rf.predict(x_val)))

    score_test_RF.append(sc_test)

print(score_test_RF)
# gdb = GradientBoostingClassifier(n_estimators = 1000, max_depth = 30,random_state=42)

# gdb.fit(x_train, y_train)

# sc_train = np.sqrt(mean_squared_error(y_train,gdb.predict(x_train)))

# sc_test = np.sqrt(mean_squared_error(y_val,gdb.predict(x_val)))

# print(sc_train,sc_test)
score_test_RF
y_train.value_counts()
from sklearn.ensemble import ExtraTreesClassifier

rf_best_now = ExtraTreesClassifier(n_estimators = 1590,max_depth = 60, random_state = 42)

rf_best_now.fit(x,y)
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

{'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
# Use the random grid to search for best hyperparameters

# First create the base model to tune

#rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

#rf_random.fit(x_train,y_train)
#rf_random.best_params_
#rf_best_now=RandomForestRegressor(max_depth=60, random_state=42,n_estimators=600,min_samples_split=5,min_samples_leaf=1,max_features='sqrt',bootstrap=False)

#rf_best_now.fit(x,y)
rmse=np.sqrt(mean_squared_error(y_val,rf_best_now.predict(x_val)))
rmse
preds = rf_best_now.predict(test)
preds
df = pd.DataFrame(preds,columns = ['rating']) 
df
df.round(0)
df['rating']=df['rating'].apply(int)
df['rating'].value_counts()
test_ids
#data_final=ids.join(df[0],how='left')

df['id'] = test_ids
df.head()
df.to_csv('sub_f20160082.csv',columns=['id','rating'],index=False)
file=pd.read_csv('sub_f20160082.csv')
file['rating'].value_counts()