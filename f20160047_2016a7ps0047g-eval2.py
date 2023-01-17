import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
df= pd.read_csv('train.csv')

df.head()
df.dtypes
df.isnull().sum()
df.describe()
plt.figure(figsize=(10,10))

sns.heatmap(data=df.corr(),annot=True)
#features=['chem_0','chem_1','chem_2','chem_3','chem_4','chem_5','chem_6','chem_7','attribute']

features=['chem_0','chem_1','chem_4','chem_6','chem_7','attribute']

X=df[features]

y=df['class']

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=69)

from sklearn.model_selection import GridSearchCV



param_grid = {

    'bootstrap': [True,False],

    'max_depth': [30, 40,50,60],

    'max_features': ['auto','sqrt'],

    'min_samples_leaf': [1,2,4],

    'min_samples_split': [2,5,8],

    'n_estimators': [600,800, 1000, 1200]

}

from sklearn.ensemble import ExtraTreesClassifier



etc = ExtraTreesClassifier(random_state=47)



grid_search = GridSearchCV(estimator = etc, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X, y)
grid_search.best_params_
pre=grid_search.predict(X_val)

rms = np.sqrt(mean_squared_error(y_val, pre))

print(rms)

print(pre)
test_data=pd.read_csv('test.csv')

test_data.head()
test_data.isnull().sum()
X_test=test_data[features]

#X_test=preprocessing.scale(X_test)

predicted=grid_search.predict(X_test)
test_data['class']=np.array(predicted)

out=test_data[['id','class']]

out=out.astype(int)

out.to_csv('submit.csv',index=False)