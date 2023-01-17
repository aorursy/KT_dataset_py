import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

%config InlineBackend.figure_format='svg'
df = pd.read_csv('../input/auto-mpg.csv')
df.head()
df.shape
def check_data_types(data):

    data_types = data.dtypes.reset_index()

    data_types.columns = ['columnn_name','data_type']

    return data_types 
check_data_types(df)
df.loc[df.horsepower=='?']
df = df.loc[df.horsepower!='?']
df['horsepower'] = df['horsepower'].astype('float64')
check_data_types(df)
numerical_cont = ['displacement', 'horsepower', 'weight', 'acceleration','mpg']

numerical_discrete = ['cylinders','model_year', 'origin']

categorical = ['car_name']
df.isnull().any()
corr_matrix = df.corr()

sns.heatmap(corr_matrix)
from pandas.plotting import scatter_matrix

scatter_matrix(df[numerical_cont],figsize=(12,12))

plt.show()
sns.boxplot(x = 'cylinders', y = 'mpg', data = df, palette = "Set2")
sns.boxplot(x = 'origin', y = 'mpg', data = df, palette = "Set2")
sns.boxplot(x = 'model year', y = 'mpg', data = df, palette = "Set2")
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler



num_attribs = ['displacement', 'horsepower', 'weight', 'acceleration']

cat_attribs = ['cylinders','model year', 'origin']





num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('std_scaler', StandardScaler()),

])

cat_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="most_frequent")),

        ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore',sparse=False)),

])
from sklearn.compose import ColumnTransformer



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", cat_pipeline, cat_attribs),

    ])
df_x = df.drop(columns=['mpg'])

df_y = df['mpg']

df_MPG = full_pipeline.fit_transform(df_x, df_y)
pd.DataFrame(df_MPG).head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_MPG, df_y, test_size = 0.2, random_state = 42)
from sklearn.metrics import mean_squared_error

mpg_mean = np.mean(y_train)

y_pred = np.full(len(X_test),mpg_mean)



print('test loss is....')

print(np.sqrt(mean_squared_error(y_pred,y_test)))
from sklearn.linear_model import LinearRegression





regressor = LinearRegression()

regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_test)

print('test loss is:')

print(np.sqrt(mean_squared_error(y_pred,y_test)))



y_pred = regressor.predict(X_train)

print('train loss is:')

print(np.sqrt(mean_squared_error(y_pred,y_train)))