import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
df= pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

df.head()
df.dtypes
df=pd.get_dummies( df, columns = ['type'] )

df.isnull().sum()
df = df.fillna(df.mean())

df.isnull().sum()
plt.figure(figsize=(15,15))

sns.heatmap(data=df.corr(),annot=True)
#features=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','type_new','type_old','feature10','feature11']

features=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','type_new','type_old','feature10','feature11']

X=df[features]

y=df['rating']

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=69)



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
'''

rf2 = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf2, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=420, n_jobs = -1)



rf_random.fit(X_train, y_train)

'''

rf_random = RandomForestRegressor(n_estimators= 1400,min_samples_split= 2,min_samples_leaf=1,max_features='sqrt',max_depth= 50,bootstrap= False)

rf_random.fit(X, y)
pre=np.rint(rf_random.predict(X_val))

rms = np.sqrt(mean_squared_error(y_val, pre))

print(rms)
#rf_random.best_params_
from sklearn.model_selection import GridSearchCV



param_grid = {

    'bootstrap': [True,False],

    'max_depth': [40, 50, 60, 70, 80],

    'max_features': ['auto','sqrt'],

    'min_samples_leaf': [1,2,4],

    'min_samples_split': [2,5,8],

    'n_estimators': [800, 1000, 1200, 1400, 1600]

}

'''

rf3 = RandomForestRegressor()



grid_search = GridSearchCV(estimator = rf3, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train,y_train)

'''
grid_search = RandomForestRegressor(bootstrap= False,max_depth= 80,max_features= 'sqrt',min_samples_leaf= 1,min_samples_split= 2,n_estimators= 1000)

grid_search.fit(X, y)
#grid_search.best_params_
pre=grid_search.predict(X_val)

rms = np.sqrt(mean_squared_error(y_val, pre))

print(rms)
test_data=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

test_data.head()
test_data.isnull().sum()
test_data=pd.get_dummies( test_data, columns = ['type'] )

test_data.dtypes
test_data.isnull().sum()
for x in test_data.columns:

    test_data[x].fillna(df[x].mean(),inplace=True)
test_data.isnull().sum()
test_data.head()
X_test=test_data[features]

predicted=np.rint(rf_random.predict(X_test))

test_data['rating']=np.array(predicted)

out=test_data[['id','rating']]

out=out.astype(int)

out.to_csv('submit1.csv',index=False)
predicted=np.rint(grid_search.predict(X_test))

test_data['rating']=np.array(predicted)

out=test_data[['id','rating']]

out=out.astype(int)

out.to_csv('submit2.csv',index=False)