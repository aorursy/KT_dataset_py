import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
df_train = pd.read_csv('/kaggle/input/train.csv')
df_sample = pd.read_csv('/kaggle/input/submission.csv')
df_train.columns
df_train.isnull().sum()
# Replacing all the Province_State that are null by the Country_Region values
df_train.Province_State.fillna(df_train.Country_Region, inplace=True)

df_train.County.fillna(df_train.Province_State, inplace=True)

df_train.isnull().sum()
df_train.columns

# taking care of categorical values from train set
# we can also use labelencoder for date column
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
df_train['County'] = labelencoder.fit_transform(df_train['County'])
df_train['Province_State'] = labelencoder.fit_transform(df_train['Province_State'])
df_train['Country_Region'] = labelencoder.fit_transform(df_train['Country_Region'])
df_train['Target'] = labelencoder.fit_transform(df_train['Target'])
# df_train['Date'] = labelencoder.fit_transform(df_train['Date'])
#taking care of the date column
df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

df_train.loc[:, 'Date'] = df_train.Date.dt.strftime("%Y%m%d")
df_train.loc[:, 'Date'] = df_train['Date'].astype(int)
df_train.columns
# splitting the dataset for training and testing
from sklearn.model_selection import train_test_split

X = df_train.iloc[:, [7, 8]].values
y = df_train.iloc[:, [8]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=0)

y_train = np.ravel(y_train)
# Fitting Random Forest Regression to the dataset
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1)
estimators = 100
model.set_params(n_estimators=estimators)

scores = []

pipeline = Pipeline([('scaler2' , StandardScaler()),
                        ('RandomForestRegressor: ', model)])
pipeline.fit(X_train , y_train)
y_pred = pipeline.predict(X_test)

pipeline.fit(X_train, y_train)
scores.append(pipeline.score(X_test, y_test))
y_pred
from sklearn.metrics import explained_variance_score , max_error ,mean_absolute_error , mean_squared_error
from math import sqrt
print('variance_score:',explained_variance_score(y_test, y_pred))
print('max_error:',max_error(y_test, y_pred))
print('mean_absolute_error score:',mean_absolute_error(y_test, y_pred))
print('mean_squared_error score:',mean_squared_error(y_test, y_pred))
print('root mean_squared_error:',sqrt(mean_squared_error(y_test,y_pred)))
 
#fitting the decision tree regression to the dataset

from sklearn.tree import DecisionTreeRegressor
regressor_new = DecisionTreeRegressor(random_state=0)
regressor_new.fit(X_train,y_train)
y_pred = regressor_new.predict(X_test)
from sklearn.metrics import explained_variance_score , max_error ,mean_absolute_error , mean_squared_error
from math import sqrt
print('variance_score:',explained_variance_score(y_test, y_pred))
print('max_error:',max_error(y_test, y_pred))
print('mean_absolute_error score:',mean_absolute_error(y_test, y_pred))
print('mean_squared_error score:',mean_squared_error(y_test, y_pred))
print('root mean_squared_error:',sqrt(mean_squared_error(y_test,y_pred)))
df_test = pd.read_csv('/kaggle/input/test.csv')
df_test.isnull().sum()
df_test.columns
# Replacing all the Province_State that are null by the Country_Region values
df_test.Province_State.fillna(df_test.Country_Region, inplace=True)
df_test.County.fillna(df_test.Province_State, inplace=True)

# taking care of categorical values from train set
# we can also use labelencoder for date column
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
df_test['County'] = labelencoder.fit_transform(df_test['County'])
df_test['Province_State'] = labelencoder.fit_transform(df_test['Province_State'])
df_test['Country_Region'] = labelencoder.fit_transform(df_test['Country_Region'])
df_test['Target'] = labelencoder.fit_transform(df_test['Target'])

#taking care of the date column
df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)

df_test.loc[:, 'Date'] = df_test.Date.dt.strftime("%Y%m%d")
df_test.loc[:, 'Date'] = df_test['Date'].astype(int)

df_test
df_test.columns
df_test.drop(['ForecastId', 'County', 'Province_State', 'Country_Region','Target','Date'], axis = 1 , inplace = True)
df_test.index.name = 'Id'
df_test
y_pred2 = pipeline.predict(df_test)
pred_list = [int(x) for x in y_pred2]

output = pd.DataFrame({'Id': df_test.index, 'TargetValue': pred_list})
print(output)
a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']
b.columns=['Id','q0.5']
c.columns=['Id','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05']
a['q0.5']=a['q0.5']
a['q0.95']=a['q0.95']
a
a['Id'] =a['Id']+ 1
a
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub.head()