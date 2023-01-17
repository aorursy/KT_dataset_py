# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
print(df_train.shape)
print(df_test.shape)
df_train.head()
#check the columns
df_train.columns
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(30)
def nullvalue_function(df_train,percentage):
    
    # Checking the null value occurance
    
    print(df_train.isna().sum())

    # Printing the shape of the data 
    
    print(df_train.shape)
    
    # Converting  into percentage table
    
    null_value_table=pd.DataFrame((df_train.isna().sum()/df_train.shape[0])*100).sort_values(0,ascending=False )
    
    null_value_table.columns=['null percentage']
    
    # Defining the threashold values 
    
    null_value_table[null_value_table['null percentage']>percentage].index
    
    # Drop the columns that has null values more than threashold 
    df_train.drop(null_value_table[null_value_table['null percentage']>percentage].index,axis=1,inplace=True)
    
    # Replace the null values with median() # continous variables 
    for i in df_train.describe().columns:
        df_train[i].fillna(df_train[i].median(),inplace=True)
    # Replace the null values with mode() #categorical variables
    for i in df_train.describe(include='object').columns:
        df_train[i].fillna(df_train[i].value_counts().index[0],inplace=True)
  
    print(df_train.shape)
    
    return df_train
df_train=nullvalue_function(df_train,30)
## Top 20 columns important as per variance
df_train.var().sort_values(ascending=False).index[0:20]
## Continuous and categorical columns

cont=df_train.describe().columns
cat=df_train.describe(include='object').columns
cat
cat.shape
cont
cont.shape
df_train_cat=df_train[cat]
df_train_cat.shape
for i in df_train_cat.columns:
    print(i,df_train_cat[i].nunique())
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
for i in df_train_cat.columns:
    le = preprocessing.LabelEncoder()
    le.fit(df_train_cat[i])
    transformed=le.transform(df_train_cat[i]) 
    df_train_cat[i]=transformed
    
df_train_cat.columns
df_train_cont=df_train[cont]
df_train_cont.shape
df_train_cat.shape
df_train.shape
df_train_cont.head()
df_train_cat.head()
analytical_dataset=pd.DataFrame()
for i in df_train_cont.columns:
    analytical_dataset[i]=df_train_cont[i]
for i in df_train_cat.columns:
    analytical_dataset[i]=df_train_cat[i]
analytical_dataset.shape
## Prioritizing the variables based on varaince 

analytical_dataset_var=analytical_dataset[analytical_dataset.var().sort_values(ascending=False).index[0:30]]
analytical_dataset_var.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(analytical_dataset_var)
normalized_x=scaler.transform(analytical_dataset_var)
analytica_dataset_norm=pd.DataFrame(normalized_x)
analytica_dataset_norm.columns=analytical_dataset_var.columns
analytica_dataset_norm.head()
analytica_dataset_norm.shape
analytica_dataset_norm['SalePrice']=df_train['SalePrice']
#descriptive statistics summary
analytica_dataset_norm['SalePrice'].describe()
#histogram
sns.distplot(analytica_dataset_norm['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % analytica_dataset_norm['SalePrice'].skew())
print("Kurtosis: %f" % analytica_dataset_norm['SalePrice'].kurt())
analytica_dataset_norm.columns
#scatterplot
col=['SalePrice', 'LotArea', 'GrLivArea', 'MiscVal', 'BsmtFinSF1','BsmtUnfSF', 'TotalBsmtSF', '2ndFlrSF',]
sns.pairplot(analytica_dataset_norm[col], size = 2.5)
plt.show();
plt.figure(figsize=(10,10))
col=['SalePrice', 'LotArea', 'GrLivArea', 'MiscVal', 'BsmtFinSF1','BsmtUnfSF', 'TotalBsmtSF', '2ndFlrSF',]
sns.heatmap(analytica_dataset_norm[col].corr(),annot=True)
y=analytica_dataset_norm['SalePrice']
x=analytica_dataset_norm.drop('SalePrice',axis=1)
df_test.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
print(x.shape,y.shape,x_train.shape, x_test.shape, y_train.shape, y_test.shape)
### KNN Algorithms

from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train,y_train)
kNN_predicted=knn.predict(x_test)
kNN_predicted
kNN_actual = y_test.values
kNN_actual
pd.DataFrame(kNN_actual,kNN_predicted).head()
kNN_data_comparision=pd.DataFrame(kNN_actual,kNN_predicted)
kNN_data_comparision.reset_index(inplace=True)
kNN_data_comparision.columns=['kNN_actual','kNN_predicted']
kNN_data_comparision.head()
kNN_data_comparision[0:292].plot(figsize=(20,5))
Error=abs(sum(kNN_data_comparision['kNN_actual']-kNN_data_comparision['kNN_predicted']))
Error
from sklearn.metrics import mean_squared_error
mean_squared_error(kNN_actual, kNN_predicted)
sum(abs(kNN_actual-kNN_predicted))/len(kNN_actual)
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(kNN_actual, kNN_predicted))
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
lm_predicted=lm.predict(x_test)
lm_predicted
lm_actual=y_test.values
erros=pd.DataFrame(lm_actual, lm_predicted).reset_index()
erros.columns=['lm_acutal','lm_predicted']
erros.head()
erros.plot(figsize=(20,5))
from sklearn.metrics import mean_squared_error
mean_squared_error(lm_actual, lm_predicted)
sum(abs(lm_actual-lm_predicted))/len(lm_actual)
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(lm_actual, lm_predicted))