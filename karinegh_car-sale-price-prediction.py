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

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,RandomizedSearchCV

from datetime import datetime
df=pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/car data.csv")
df.head()
#Dataset dimensionality
df.shape

df.columns

#Dataset info
df.info()


#Data description
df.describe()
#check missing values
df.isnull().sum()

#No missing values
#Check if there are duplicates
df.duplicated().any() 
#Drop duplicates
df_copy=df.copy()
df.drop_duplicates(inplace = True)
df.reset_index()
df.shape
#outlier detection
sns.boxplot(df['Present_Price'])
Q1 = df['Present_Price'].quantile(0.25)
Q3 = df['Present_Price'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

#to remove outliers
#df_outl = df[~((df['Present_Price'] < (Q1 - 1.5 * IQR)) |(df['Present_Price']> (Q3 + 1.5 * IQR)))]
#check the cars acting like outliers
df_out=df[df['Present_Price'] > (Q3 + 1.5 * IQR)][['Car_Name','Year','Present_Price','Selling_Price']]
df_out
#unique values for categorical variables
cat_var=['Seller_Type','Fuel_Type','Transmission','Owner']
for col in cat_var:
    print(col, "unique values are: \n" ,df[col].unique())
    
fig, ax =plt.subplots(1,4,figsize=(25, 4))
i=0
for col in cat_var:    
    sns.countplot(x =col, data =df,ax=ax[i])
    i=i+1
    

#variables distribution
sns.pairplot(df)
#feature engineering


#Create a new column named no_year to contain the age of the car
df['no_year']=datetime.now().year - df['Year']

#drop unneeded columns (Year and Car_Name)
df.drop(['Year', 'Car_Name'], axis=1,inplace=True)
df
# convert categorical variable into numerical variable
# Create dummy variables 
df=pd.get_dummies(df,drop_first=True)
df.head()
#correlation
df.corr()

corrmat=df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heatmap
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#define independent (X) and dependent (y) variables
X=df.drop(['Selling_Price'], axis=1)
y=df['Selling_Price'] 
X.shape,y.shape
#train test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#Random forest

##hyperparameters using Randomized Search CV
#Create the random grid to search for best hyperparameters
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100 , stop= 1200 , num=12)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(5, 30, num = 6)],
               'min_samples_split': [2, 5, 10, 15, 100],
               'min_samples_leaf': [1, 2, 5, 10]}

# First we create the base model to tune
rf = RandomForestRegressor()

# Random search of parameters, using 5 fold cross validation, 

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


rf_random.fit(X_train,y_train)

print("best parameters : ")
rf_random.best_params_
print("Score : ")
rf_random.best_score_
#predictions
predictions = rf_random.predict(X_test)
predictions
sns.distplot(y_test-predictions)
pred1=pd.DataFrame(predictions)
# plot original vs predicted values
plt.figure(figsize = (14, 6))
index=y_test.reset_index()["Selling_Price"]
ax=index.plot(label="original_values")
ax=pred1[0].plot(label = "predicted_values")
plt.legend(loc='upper right')
plt.title("test VS pred")
plt.xlabel("indexes")
plt.ylabel("values")
plt.show()
print("MSE  : ",mean_squared_error(y_test, predictions))
print("r2  value is : ",r2_score(ytest,pred))
r2 = r2_score(ytest,pred)
n = len(xtest)
k = xtest.shape[1]
adj_r2_score = 1 - (((1- r2)*(n-1)) / (n - k - 1))
print("adj_r2_score  value is : ",adj_r2_score)
print('R2:', r2_score(y_test, predictions))

R2 = r2_score(y_test, predictions)
n = len(X_test)
k = X_test.shape[1]
Adj_R2_score = 1 - (((1- R2)*(n-1)) / (n - k - 1))
print('Adj_R2 : ',Adj_R2_score)

print('MAE:', mean_absolute_error(y_test, predictions))

print('MSE:', mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))



