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
import pandas as pd

import numpy as np
df=pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
df.head()

df.shape
print(df['Seller_Type'].unique())

print(df['Fuel_Type'].unique())

print(df['Transmission'].unique())

print(df['Owner'].unique())
df.isnull().sum()
df.describe()
final_df= df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
final_df.head()
final_df['Current Year']=2020
final_df.head()
final_df['no_year']=final_df['Current Year']- final_df['Year']
final_df.head()

final_df.drop(['Year'],axis=1,inplace=True)
final_df.head()
final_df=pd.get_dummies(final_df, drop_first=True)
final_df.head()
final_df.corr()
import seaborn as sns
sns.pairplot(final_df)
import seaborn as sns

import matplotlib.pyplot as plt

#get correlations of each features in dataset

corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

x=final_df.iloc[:, 1:]

y=final_df.iloc[:,0]
x['Owner'].unique()
x.head()
y.head()
from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt

model=ExtraTreesRegressor()

model.fit(x,y)
print(model.feature_importances_)
#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=x.columns)

feat_importances.nlargest(6).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

print(n_estimators)
from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
random_grid ={'n_estimators':n_estimators,

                         'max_features':max_features,

                         'max_depth':max_depth,

                         'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



print(random_grid)
rf=RandomForestRegressor()
rf_random= RandomizedSearchCV(estimator=rf, param_distributions=random_grid , scoring='neg_mean_squared_error', n_iter = 10, cv = 5, 

                             verbose=2,random_state=42 ,n_jobs=1)

rf_random.fit(x_train, y_train)
rf_random.best_params_
rf_random.best_score_

pred=rf_random.predict(x_test)
sns.distplot(y_test-pred)
plt.scatter(y_test , pred)
from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import mean_squared_error as mse

print("MAE:" , mae(y_test,pred))

print("MSE:" , mse(y_test,pred))

print("RMSE:" ,np.sqrt(mse(y_test,pred)))