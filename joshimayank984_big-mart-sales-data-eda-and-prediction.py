# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/bigmart-sales-data/Train.csv')
train.shape

#so the TRAIN data has 8523 rows and 12 columns
# get the list of all columns in TRAIN data
train.columns
# finding out the datatype of each column
train.info()
#we shall separate the categorical and numeric columns
cat_data = []
num_data = []

for i,c in enumerate(train.dtypes):
    print(i,c)
    if c == object or c == int:
        cat_data.append(train.iloc[:, i])
    else :
        num_data.append(train.iloc[:, i])

cat_data = pd.DataFrame(cat_data).transpose()
num_data = pd.DataFrame(num_data).transpose()
cat_data.head()
import pandas_profiling as prof

#report = prof.ProfileReport(train)
#report
cat_data.Item_Fat_Content.unique()
cat_data['Item_Fat_Content'] = cat_data['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat','reg': 'Regular'})
print(cat_data.Item_Fat_Content.unique())
cat_data.head()
##now we see that we now have only two unique Fat_content types

train = pd.concat([cat_data,num_data], axis = 1)
train.head()
train.shape
## to start with a pairplot
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(train)
plt.show()
num_data.corr()
sns.heatmap(num_data.corr(),annot=True)
# lets get the basic statistical measures for num_data
num_data.describe()
cat_data.describe()
cat_data['Item_Identifier'].value_counts()
cat_data = cat_data.drop(['Item_Identifier'], axis=1)
cat_data.columns
for i in cat_data.columns:
    print()
    print(i)
    print(cat_data[i].value_counts())
    print()
    print()
cat_columns = cat_data.columns
cat_columns = list(cat_columns)
cat_columns.remove('Outlet_Establishment_Year')
cat_columns_new = cat_columns


for cat in cat_columns_new:
    print(cat)
    print()
    for i in num_data.columns:
        print(i, "vs", cat)
        sns.set(style="whitegrid")
        sns.boxplot(train[i], train[cat])
        plt.show()
train.columns
## 1. Which class in each category corresponds to maximum sales

for cat in cat_data.columns:
    print("Item_Outlet_Sales in Thousands ('000)")
    print()
    print("-"*20 + cat + '  vs' + '  Item_Outlet_Sales' + "-"*20)
    output = train[[cat,'Item_Outlet_Sales']].groupby([cat]).apply(lambda x: x['Item_Outlet_Sales'].sum()/1000).sort_values(ascending=False)
    output = pd.DataFrame(output)
    output.columns = ['Item_Outlet_Sales']
    ax = sns.barplot(output.index,'Item_Outlet_Sales', data =output)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width()/ 2., p.get_height()),ha='center', va='center', rotation=90, xytext=(0,40), textcoords='offset points')  #vertical bars
    plt.tight_layout()
    plt.show()
    print()
    print("Maximum Sales : ")
    print(output.head(1))
    print()
    print("-" *50)
cat_data.describe()
train.groupby(['Outlet_Identifier','Outlet_Establishment_Year']).size().reset_index(name='Freq')
#train['Outlet_Identifier'].unique()
#train['Outlet_Establishment_Year'].unique()
train.columns
year_store_sales = train[['Outlet_Identifier','Outlet_Establishment_Year','Item_Outlet_Sales']].groupby(['Outlet_Identifier','Outlet_Establishment_Year']).apply(lambda x: x['Item_Outlet_Sales'].sum()/1000).sort_values(ascending=False)
year_store_sales = pd.DataFrame(year_store_sales)
year_store_sales.columns = ['Outlet_Sales']
year_store_sales
#To find out the reason for the above case we can act smart and subset data on the Outlet_Identifier and
#then remove duplicates (assuming that an outlet will be at only one location)

outlets = train[['Outlet_Identifier',"Outlet_Size","Outlet_Location_Type","Outlet_Type"]]
print(outlets.head())
print()
print("Before removing NA : " , outlets.shape)
# removing duplicates
outlets_new = outlets.drop_duplicates(subset=['Outlet_Identifier'])
outlets_new
train[train.Outlet_Identifier.isin(['OUT010', "OUT045", "OUT017"])].shape[0] / train.shape[0]
ld = train[["Outlet_Identifier","Outlet_Location_Type","Outlet_Size","Outlet_Type"]].groupby(['Outlet_Location_Type',"Outlet_Type","Outlet_Identifier"]).size()
ld = pd.DataFrame(ld)
ld.columns = ['Count']
ld
## finding the null values and treating them
heat = sns.heatmap(train.isnull(), cbar=False)
plt.show()
Null_percent = train.isna().mean().round(4)*100

Null_percent = pd.DataFrame({'Null_percentage' : Null_percent})
Null_percent.head()
Null_percent = Null_percent[Null_percent.Null_percentage > 0].sort_values(by = 'Null_percentage', ascending = False)
print("Percentage of Null cells : \n \n " , Null_percent)
print(cat_data.columns)
print(num_data.columns)

cat_data_new = cat_data.drop(['Outlet_Size'], axis =1)
num_data_new = num_data.drop(['Item_Weight'], axis =1) 
cat_data_new.head()
num_data_new.head()

Null_percent_cat = cat_data_new.isna().mean().round(4)*100
print(Null_percent_cat)
Null_percent_num = num_data_new.isna().mean().round(4)*100
print(Null_percent_num)
## the columns we have to encode
## we shall not encode Outlet Establishment Year column
cat_data_new['Outlet_Establishment_Year'] = pd.to_numeric(cat_data_new['Outlet_Establishment_Year'])
cat_data_new.info()
## loading Standardscaler and OneHotEncoder from Sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
num_data_new.columns
## num_data has certain columns with some very high valued columns and some very low, thus we 
#should standardize the values of these columns

y = num_data_new['Item_Outlet_Sales']
num_data_new = num_data_new.drop(['Item_Outlet_Sales'], axis = 1)
print(num_data_new.columns)

for col in num_data_new.columns:
    num_data_new[col] = (num_data_new[col]-num_data_new[col].min())/(num_data_new[col].max() - num_data_new[col].min())
    
num_data_new.head()
##Label Encoding
from sklearn.preprocessing import LabelEncoder  
le = LabelEncoder()
cat_data_new.head()
# transform categorical columns columns

for i in cat_data_new.loc[:,~cat_data_new.columns.isin(['Outlet_Establishment_Year'])]:
    cat_data_new[i] = le.fit_transform(cat_data_new[i])

cat_data_new.head()
cat_data_new.shape
cat_data_new.columns
cat_data_new = pd.get_dummies(cat_data_new, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier','Outlet_Location_Type', 'Outlet_Type'])
cat_data_new.head()
cat_data_new.shape
x = pd.concat([num_data_new,cat_data_new], axis = 1)
x.head()
y
## I have an idea, why not convert the Establishment year 
#column to Years of existence by subtracting it by the current year

x['Outlet_Establishment_Year'] = 2020 - x['Outlet_Establishment_Year']

x.rename(columns = {"Outlet_Establishment_Year" : "Years_since"}, inplace = True)
 
x['Years_since'].describe()
data_dummy = pd.concat([x,y], axis =1)
data_dummy.columns
from sklearn.model_selection import train_test_split
train,test = train_test_split(data_dummy,test_size=0.20,random_state=2019)
print(train.shape)
print(test.shape)
train_label=train['Item_Outlet_Sales']
test_label=test['Item_Outlet_Sales']
del train['Item_Outlet_Sales']
del test['Item_Outlet_Sales']
# algos to be used
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

# evaluating the model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

model_df = {'Name':['LR', 'Ridge', 'Lasso', 'E_Net','SVR','Dec_Tree','RF','Bagging_Reg','AdaBoost','Grad_Boost'],
             'Model' : [LinearRegression(), Ridge(alpha=0.05,solver='cholesky'), Lasso(alpha=0.01) ,ElasticNet(alpha=0.01,l1_ratio=0.5),
                     SVR(epsilon=15,kernel='linear'),DecisionTreeRegressor(),
                     RandomForestRegressor(),BaggingRegressor(max_samples=70),AdaBoostRegressor(),GradientBoostingRegressor()]}

model_df = pd.DataFrame(model_df)
model_df['Cross_val_score_mean'], model_df['Cross_val_score_STD'] = 0,0
model_df

for m in range(0,model_df.shape[0]):
    print(model_df['Name'][m])
    score=cross_val_score(model_df['Model'][m] , train , train_label , cv=10 , scoring='neg_mean_squared_error')
    score_cross=np.sqrt(-score)
    model_df['Cross_val_score_mean'][m] = np.mean(score_cross)
    model_df['Cross_val_score_STD'][m] = np.std(score_cross)
    
model_df
model_df.sort_values(by=['Cross_val_score_mean'])