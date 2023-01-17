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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
%matplotlib inline 

import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec

#Plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

#Some styling
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")

#Subplots
from plotly.subplots import make_subplots


train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_data.head()
print('Train Data :>',train_data.shape)
print('-'*123)
print('Test Data :>',test_data.shape)
print('Train Data :>',train_data.info())
print('-' * 123)
print('Test Data :>',test_data.info())
# Descriptive measure of data
train_data.describe(include='all')
train_data.isna().sum().sort_values(ascending=False)
# First we create a list of missing values by each feature 
temp = list(train_data.isna().sum())
# them we create a list of columns and their missing values as inner list to a separate list
lst=[]
i=0
for col in train_data.columns:
    insert_lst = [col,temp[i]]
    lst.append(insert_lst)
    i+=1

# finally create a dataframe
temp_train_data = pd.DataFrame(data=lst,columns=['Column_Name','Missing_Values'])
fig = px.bar(temp_train_data.sort_values(by='Missing_Values'),x='Missing_Values',y='Column_Name',
             orientation='h',height=1500,width=900,color='Missing_Values',text='Missing_Values')
fig.update_traces(textposition='outside')
fig.show()
# The following columns have missing values
temp_train_data[temp_train_data['Missing_Values']>0].sort_values(by='Missing_Values',
                                                 ascending=False).reset_index(drop=True).style.background_gradient(cmap='Reds')
columns_name = []
for i in train_data.columns:
    if train_data[i].isna().sum()/len(train_data)*100 >=10:
        columns_name.append(i)
print(columns_name)# List of columns which has >=10% missing data      
# Droping columns in train data
train_data = train_data.drop(columns=columns_name,axis = 1)
train_data.sample(5)
train_data.SalePrice.describe()
fig = make_subplots(rows=1,cols=2)

fig.add_trace(go.Histogram(x=train_data['SalePrice']),row=1,col=1)
fig.add_trace(go.Box(y=train_data['SalePrice'],boxpoints='all',line_color='orange'),row=1,col=2)

fig.update_layout(height=500, showlegend=False,title_text='SalePrice Distribution and Box Plot')
# checking the skewness
train_data.skew().sort_values(ascending=False)
plt.figure(figsize=(100,90))
sns.heatmap(train_data.corr(),annot = True,fmt=".1f",annot_kws={'size':48})
# Returns correlation among fatures which obervations
fig = px.histogram(train_data, x="SalePrice", color='OverallQual',barmode="overlay",title="Overall Quality of the house")
fig.update_layout(height=500)
fig.show()

fig = px.histogram(train_data, x="SalePrice", color='OverallCond',barmode="overlay",title="Overall Condition of the house")
fig.update_layout(height=500)
fig.show()
feature_data = train_data.drop(columns=['SalePrice'],axis=1)
target_data =train_data.SalePrice 
# int and Float columns
int_float_data = feature_data.select_dtypes(include=['int','float'])
# categorical columns
cat_data = feature_data.select_dtypes(include=['object'])
# creating sub-pipeline 
float_int_pipeline = make_pipeline(SimpleImputer(strategy='median'),MinMaxScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OrdinalEncoder())
# transforming columns (preprocessing)
preprocessor = make_column_transformer(
    (float_int_pipeline,int_float_data.columns),
    (cat_pipeline,cat_data.columns)
)
# using LinearRegression
pipeline = make_pipeline(preprocessor, LinearRegression())
trainX, testX, trainY, testY = train_test_split(feature_data, target_data)
pipeline.fit(feature_data, target_data)
pipeline.score(testX,testY) # using LinearRegression wwe geting 0.69
# now using RandomForestClassifier
pipeline = make_pipeline(preprocessor, RandomForestClassifier()) 
pipeline.fit(feature_data, target_data)
pipeline.score(testX,testY) # For this data set the RandomForestClassifier is best model to predict
# Now check the mean_squared_error
y_pred = pipeline.predict(testX)
mean_squared_error(testY, y_pred)
r2_score(testY, y_pred)
y_pred = pipeline.predict(test_data)
submission = pd.DataFrame({'Id': test_data.index,'SalePrice': y_pred})
submission.to_csv("house_prices_submission.csv", index=False)
