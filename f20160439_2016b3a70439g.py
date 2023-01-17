import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

df.head()
df.info()
df.fillna(value=df.mean(),inplace=True)  #Since there is no categorical data of int64 dtype, we can do this freely. Had there been categorical data of type int64 we would have to do df.fillna(value=df[numerical_features].mean(),inplace=True)

# Only NaNs in float and int dtype columns are replaced with mean. [See output of df.mean() to understand this]

df.head()
df=pd.get_dummies(data=df,columns=['type'])

df.head()
df.isnull().any().any()
corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
X=df[["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11","type_new"]].copy()

y=df["rating"].copy()
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=50) #changed from 50
#for pvt

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

reg3 = ExtraTreesRegressor()

#parameters = {'n_estimators':[439,489], 'max_depth':[68,69], 'random_state':[280,68]}

parameters = {'n_estimators':[444,354,439,456], 'max_depth':[67,68,69], 'random_state':[50,51,69]}

#parameters = {'n_estimators':[439,569,756], 'max_depth':[69,75], 'random_state':[50,69,68]}

scorer = make_scorer(mean_squared_error,greater_is_better=False)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(reg3,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X,y)

best_reg3 = grid_fit.best_estimator_   
df1=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

df1.info()
df1.fillna(value=df1.mean(),inplace=True)  #Since there is no categorical data of int64 dtype, we can do this freely. Had there been categorical data of type int64 we would have to do df.fillna(value=df[numerical_features].mean(),inplace=True)

# Only NaNs in float and int dtype columns are replaced with mean. [See output of df.mean() to understand this]

df1=pd.get_dummies(data=df1,columns=['type'])

df1.head()
X_test=df1[["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11","type_new"]].copy()

y_pred3=best_reg3.predict(X_test)

y3=np.around(y_pred3)
df4=pd.DataFrame()

df4['id']=df1['id']

df4['rating']=y3

df4.head()
df4.to_csv('sol2.csv',index=False)
#for public

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import mean_squared_error

reg33 = ExtraTreesRegressor()

parameters = {'n_estimators':[439,456], 'max_depth':[69,67], 'random_state':[50,68]}

scorer = make_scorer(mean_squared_error,greater_is_better=False)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(reg33,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X,y)

best_reg33 = grid_fit.best_estimator_  
y_pred33=best_reg33.predict(X_test)

y33=np.around(y_pred33)
df5=pd.DataFrame()

df5['id']=df1['id']

df5['rating']=y33

df5.head()
df5.to_csv('sol11.csv',index=False)