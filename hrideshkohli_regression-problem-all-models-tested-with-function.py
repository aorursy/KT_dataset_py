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
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
plt.rcParams['figure.figsize']= (12,6)
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette='dark', style="whitegrid")
df=pd.read_csv('../input/diamonds/diamonds.csv')
df.head()
# shape of the data
df.shape
# drop Unnamed column
df.drop(df.columns[0], axis=1, inplace=True)
# checking data types
df.info()
df.isnull().sum()
df.shape
#visualizing missing numbers
msno.matrix(df)
df.describe()
df[(df['x']==0) | (df['y']==0) | (df['z']==0)]
df=df[~((df['x']==0) | (df['y']==0) | (df['z']==0))]
df[(df['x']==0) | (df['y']==0) | (df['z']==0)]
# check correlation b/w features
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot=True, cmap='viridis', cbar=True)
plt.show()
numerical_cols=df.select_dtypes(include=np.number).columns.to_list()
categorical_cols=df.select_dtypes(exclude=np.number).columns.to_list()
numerical_cols
categorical_cols
sns.catplot('cut', data=df, kind='count',aspect=2.5)
sns.catplot(x='cut', y='price', kind='box', data=df, aspect=2.5)
sns.catplot('color', kind='count', data=df, aspect=2.5)
sns.catplot(x='color', y='price', data=df, aspect =2.5, kind='box')
numerical_cols
sns.pairplot(df[numerical_cols], kind='reg')
# Let's create a new column volume
df['volume']=df['x']*df['y']*df['z']
df.head()
df.drop(['x', 'y', 'z'], axis=1, inplace=True)
# Apply categorical encoding
df=pd.get_dummies(df, drop_first=True)
df.head()
df.info()
# Conver to X & y
X=df.drop('price', axis=1)
y=df['price']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
sc=StandardScaler()
X_train_tx=sc.fit_transform(X_train)
X_test_tx=sc.transform(X_test)
dataset_1=(X_train, X_test, y_train, y_test, 'dataset_1')
# Blank lists for all the details
model_=[]
cv_score_test=[]
cv_score_train=[]
mse_=[]
mae_=[]
rmse_=[]
r2_=[]


def run_model(model, dataset, modelname):
    model.fit(dataset[0], dataset[2])
    accuracies=cross_val_score(estimator=model, X=dataset[0], y=dataset[2], cv=5, verbose=1)
    y_pred=model.predict(dataset[1])
    print('')
    score_1=model.score(dataset[1], dataset[3])
    print(f'#### {modelname} ####')
    print("score :%.4f" %score_1)
    print(accuracies)
    
    
    mse=mean_squared_error(dataset[3], y_pred)
    mae=mean_absolute_error(dataset[3], y_pred)
    rmse=mean_squared_error(dataset[3], y_pred)**0.5
    r2=r2_score(dataset[3], y_pred)
    
    
    print('')
    print('MSE    : %0.2f ' % mse)
    print('MAE    : %0.2f ' % mae)
    print('RMSE   : %0.2f ' % rmse)
    print('R2     : %0.2f ' % r2)
    
    ## appending to the lists
    
    model_.append(modelname)
    cv_score_test.append(score_1)
    cv_score_train.append(np.mean(accuracies))
    mse_.append(mse)
    mae_.append(mae)
    rmse_.append(rmse)
    r2_.append(r2)
model_dict={'LinearRegression': LinearRegression(), 'LassoRegression': Lasso(normalize=True), 
             'AdaBoostRegressor': AdaBoostRegressor(n_estimators=1000),
            'RidgeRegression': Ridge(normalize=True),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, verbose=1),
           'RandomForestRegressor': RandomForestRegressor(), 
           'KNeighborsRegressor': KNeighborsRegressor()
           }
run_model(model_dict['LinearRegression'], dataset_1, "LinearRegression")
for models in model_dict:
    run_model(model_dict[models], dataset_1, models)
accuracy_data=pd.DataFrame(zip(model_, cv_score_test, cv_score_train, mse_, mae_, rmse_, r2_), columns=['Model', 'CV Test score', 'CV Train score (mean)', '%%SVGean Squared error', 'Mean Absolute error', 'Root Mean Squared error', 'R2 Score'])
accuracy_data