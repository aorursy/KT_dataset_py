# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot

import scipy
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from IPython.display import Image
from sklearn.model_selection import KFold

df=pd.read_csv("../input/pricehouse/DataFull.csv")
print("Count Columns  :",df.shape[0])
print("Count Rows     :",df.shape[1])
df.head(5)
# print feature
print(df.columns)
print(df.info())
df.describe(percentiles=[0.01, 0.25,0.75, 0.99])
print(df[df["Rooms"]>5].Rooms)
# Number
numerical_feature = df.dtypes[df.dtypes != "object"].index
print(numerical_feature)
# Category
categorical_feature = df.dtypes[df.dtypes == "object"].index
print(categorical_feature)
# distribution chart of price(biểu đồ phân phối giá)
sns.distplot(df['Price'])
# convert price to log
Price_log=np.log(df['Price'])
sns.distplot(Price_log)
# ['Rooms', 'Price', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
# 'Landsize','BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude','Propertycount'],
# Distribution chart number feature
fig,ax = plt.subplots(6,2, figsize=(9,9))     
sns.distplot(df['Rooms'], ax = ax[0,0]) 
sns.distplot(df['Distance'], ax = ax[0,1]) 
sns.distplot(df["Postcode"], ax = ax[1,0]) 
sns.distplot(df["Bedroom2"], ax = ax[1,1]) 
sns.distplot(df["Bathroom"], ax = ax[2,0]) 
sns.distplot(df['Car'], ax = ax[2,1]) 
sns.distplot(df['Landsize'], ax = ax[3,0]) 
sns.distplot(df['BuildingArea'], ax = ax[3,1]) 
sns.distplot(df['YearBuilt'], ax = ax[4,0]) 
sns.distplot(df["Lattitude"], ax = ax[4,1]) 
sns.distplot(df["Longtitude"], ax = ax[5,0]) 
sns.distplot(df["Propertycount"], ax = ax[5,1]) 
plt.tight_layout()
plt.show()
# example
Image("../input/image/outlier.png")
fig,ax = plt.subplots(3,3, figsize=(9,9))     
sns.boxplot(df['Rooms'], ax = ax[0,0]) 
sns.boxplot(df['Distance'], ax = ax[0,1]) 
sns.boxplot(df["Postcode"], ax = ax[0,2]) 
sns.boxplot(df["Bedroom2"], ax = ax[1,0]) 
sns.boxplot(df["Bathroom"], ax = ax[1,1]) 
sns.boxplot(df['Car'], ax = ax[1,2])  
sns.boxplot(df["Lattitude"], ax = ax[2,0]) 
sns.boxplot(df["Longtitude"], ax = ax[2,1]) 
sns.boxplot(df["Propertycount"], ax = ax[2,2]) 
plt.tight_layout()
plt.show()
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
df2=df
count_outlier=[]
for i in range(13):
    index = df2[(df2[IQR.index[i]] < (Q1[i]-1.5*IQR[i])) | (df2[IQR.index[i]] >(Q3[i]+1.5*IQR[i]))].index
    count_outlier.append(len(index))
percent=[]
for count in count_outlier:
    percent.append(100*(count/(df2.shape[0])))
out_lier=pd.DataFrame({"count outlier":count_outlier,"percent":percent},
                      index=IQR.index)
print(out_lier)

for i in range(13):
    index = list(df2[(df2[IQR.index[i]] >= (Q3[i]+1.5*IQR[i]))|(df2[IQR.index[i]] <=(Q1[i]-1.5*IQR[i]))].index)
    df2.drop(index, inplace=True)
df2.shape
sns.boxplot(df2['Price'])
fig,ax = plt.subplots(3,3, figsize=(9,9))     
sns.boxplot(df2['Rooms'], ax = ax[0,0]) 
sns.boxplot(df2['Distance'], ax = ax[0,1]) 
sns.boxplot(df2["Postcode"], ax = ax[0,2]) 
sns.boxplot(df2["Bedroom2"], ax = ax[1,0])
sns.boxplot(df2["Bathroom"], ax = ax[1,1]) 
sns.boxplot(df2['Car'], ax = ax[1,2])  
sns.boxplot(df2["Lattitude"], ax = ax[2,0]) 
sns.boxplot(df2["Longtitude"], ax = ax[2,1]) 
sns.boxplot(df2["Propertycount"], ax = ax[2,2]) 
plt.tight_layout()
plt.show()
sns.pairplot(df2, x_vars=['Rooms', 'Distance', 'Bedroom2', 'Bathroom'],
             y_vars=["Price"],height=8, aspect=.8, kind="reg");
sns.pairplot(df2, x_vars=['Car','Lattitude','Longtitude','Propertycount'],
             y_vars=["Price"],height=8, aspect=.8, kind="reg");

for catg in list(categorical_feature) :
    print(df2[catg].value_counts())
    print("****************************************************************")
print(df2["Type"].unique().shape,df2['Method'].unique().shape,df2['Regionname'].unique().shape)
sns.catplot(x="Type", y='Price', data=df2,kind='box')
sns.catplot(x="Method", y='Price', data=df2,kind='box')
f , ax = plt.subplots(figsize=(18,12))
sns.violinplot(x="Regionname", y='Price', data=df2)

df2["Date"].value_counts()
# Convert to Datetime Format
df2['Date'] = pd.to_datetime(df2['Date'], format="%d/%m/%Y")
# Analyze seasonality per month (and answer the question in which month there is more demand)
df2['Month'] = df2['Date'].dt.month
df2=df2.drop(columns=["Date"])
df2["Month"].value_counts()
# Subplot #1
total_sales = df2['Price'].sum()

def month_sales(df2, month, sales=total_sales):
    share_month_sales = df2['Price'].loc[df['Month'] == month].sum()/sales
    return share_month_sales

january_sales = month_sales(df2, 1)
february_sales = month_sales(df2, 2)
march_sales = month_sales(df2, 3)
april_sales = month_sales(df2, 4)
may_sales = month_sales(df2, 5)
june_sales = month_sales(df2, 6)
july_sales = month_sales(df2, 7)
august_sales = month_sales(df2, 8)
september_sales = month_sales(df2, 9)
october_sales = month_sales(df2, 10)
november_sales = month_sales(df2, 11)
december_sales = month_sales(df2, 12)
month_total_sales = [january_sales, february_sales, march_sales, april_sales,
                     may_sales, june_sales, july_sales, august_sales, 
                     september_sales, october_sales, november_sales, december_sales]
labels = ['January', 'February', 'March', 'April',
          'May', 'June', 'July', 'August', 'September', 
          'October', 'November', 'December']

colors = ['#ffb4da', '#b4b4ff', '#daffb4', '#fbab60', '#fa8072', '#FA6006',
          '#FDB603', '#639702', '#dacde6', '#faec72', '#9ab973', '#87cefa']

pie_plot = go.Pie(labels=labels, values=month_total_sales,
               hoverinfo='label+percent',
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))
data = [pie_plot]
layout = go.Layout(title="Price by Month")
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='lowest-oecd-votes-cast')
df2.shape
df2.isnull().sum()
#Take data have Price not missing value
df3=df2[df2.Price.isnull()==False]
df3.shape
total = df3.isnull().sum().sort_values(ascending=False)
percent = (df3.isnull().sum()/df3.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data.head(16)
# Delete Row missing of Distance,Postcode,CouncilArea,Regionname,Propertycount
row_missing=df3[df3.Distance.isnull()==True].index
row_missing=row_missing.append(df3[df3.Postcode.isnull()==True].index)
row_missing=row_missing.append(df3[df3.CouncilArea.isnull()==True].index)
row_missing=row_missing.append(df3[df3.Regionname.isnull()==True].index)
row_missing=row_missing.append(df3[df3.Propertycount.isnull()==True].index)
row_missing=list(set(row_missing))
df3=df3.drop(row_missing)
print("shape data after del row :",df3.shape)
# Delete columns if columns have count missing value > 20% :
# => del 8 column
df3=df3.drop(columns=["Car","Bathroom","Bedroom2","Lattitude","Longtitude","Landsize","BuildingArea","YearBuilt"])
print("shape data after del columns :",df3.shape)
# nếu thay 30% >missing > 20% = mean
#missing_replace=["Bedroom2","Bathroom","Car","Lattitude","Longtitude"]

#for i in range(len(missing_replace)):
#    df3[missing_replace[i]]=df3[missing_replace[i]].replace(np.NaN,df3[missing_replace[i]].mean())
#df3.head(5)
df3.head(5)
df3.isnull().sum().sum()
# convert month to category
df3["Month"]=df3["Month"].replace(1,"January")
df3["Month"]=df3["Month"].replace(2,"February")
df3["Month"]=df3["Month"].replace(3,"March")
df3["Month"]=df3["Month"].replace(4,"April")
df3["Month"]=df3["Month"].replace(5,"May")
df3["Month"]=df3["Month"].replace(6,"June")
df3["Month"]=df3["Month"].replace(7,"July")
df3["Month"]=df3["Month"].replace(8,"August")
df3["Month"]=df3["Month"].replace(9,"September")
df3["Month"]=df3["Month"].replace(10,"October")
df3["Month"]=df3["Month"].replace(11,"November")
df3["Month"]=df3["Month"].replace(12,"December")
df3["Month"].value_counts()

# Number
numeber_feature = df3.dtypes[df3.dtypes != "object"].index
print(numeber_feature)

f , ax = plt.subplots(figsize=(20,12))
df3_corr=df3.corr()
sns.heatmap(df3_corr,vmax=1, annot=True)
# Category
category_feature = df3.dtypes[df.dtypes == "object"].index
print(category_feature)
df_Type_u=df3[df3["Type"]=="u"].Price
df_Type_h=df3[df3["Type"]=="h"].Price
df_Type_t=df3[df3["Type"]=="t"].Price
stats.f_oneway(df_Type_u,df_Type_h,df_Type_t)
Month_set=set(df3["Month"])
print("Number of unique of month is :",len(Month_set))
sample=[]
for uni in Month_set:
    po=list(df3["Price"][df3[df3["Month"]==uni].index])
    sample.append(po)
# có 12 sample dùng anova cho từng cặp 0 và 1 ,1 và 2 xem thử có 1 cặp khác hay là tất cả đều same nhau
# nếu có 1 cặp khác 
khac=0
for i in range(len(sample)-1):
    f,p=stats. f_oneway(sample[i],sample[i+1])
    if p <0.05:
        khac=1
        break
print("have reject H0 : ",khac)
Method_set=set(df3["Method"])
print(Method_set)
sample=[]
for uni in Method_set:
    po=list(df3["Price"][df3[df3["Method"]==uni].index])
    sample.append(po)
# có 12 sample dùng anova cho từng cặp 0 và 1 ,1 và 2 xem thử có 1 cặp khác hay là tất cả đều same nhau
# nếu có 1 cặp khác 
khac=0
for i in range(len(sample)-1):
    f,p=stats. f_oneway(sample[i],sample[i+1])
    if p <0.05:
        khac=1
        break
print("Have reject H0 :",khac)
df3["Regionname"].value_counts()
df3["SellerG"].value_counts()

df3["CouncilArea"].value_counts()
CouncilArea_set=set(df3["CouncilArea"])
sample=[]
for uni in CouncilArea_set:
    po=list(df2["Price"][df3[df3["CouncilArea"]==uni].index])
    sample.append(po)
# có 12 sample dùng anova cho từng cặp 0 và 1 ,1 và 2 xem thử có 1 cặp khác hay là tất cả đều same nhau
# nếu có 1 cặp khác 
khac=0
for i in range(len(sample)-1):
    f,p=stats. f_oneway(sample[i],sample[i+1])
    if p <0.05:
        khac=1
        break
print("Reject H0 :",khac)
df3.head(5)
df_Model=df3.drop(columns=["Suburb","Address","SellerG","Distance","Postcode","Regionname","Propertycount"])
df_Model=pd.get_dummies(df_Model)
df_Model.head(5)
Y=df_Model.Price
Y=np.log(Y)
X=df_Model.drop(columns=["Price"])
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X.shape)
X1_train,X1_test,Y1_train,Y1_test=train_test_split(X,Y,test_size=0.2,random_state=0)
# create model
regressor1= LinearRegression()
regressor1.fit(X1_train,Y1_train)
## Predictiions for TEST
Y1_pred=regressor1.predict(X1_test)
ms_error1 = metrics.mean_squared_error(Y1_test, Y1_pred)
print(ms_error1)
regressor1.score(X1_test,Y1_test)
regressor1= LinearRegression(copy_X=True,fit_intercept=True,n_jobs=1,normalize=True)
regressor1.fit(X1_train,Y1_train)
regressor1.score(X1_test,Y1_test)
kfold=KFold(n_splits=10, shuffle=False, random_state=None)
scores1= cross_val_score(regressor1, X, Y,cv=kfold)
print( scores1)
kfold=KFold(n_splits=100, shuffle=False, random_state=None)
scores1= cross_val_score(regressor1, X, Y,cv=kfold)
print( scores1)
print(X.shape)
cov_mat = np.cov(X.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca.T[0], X_pca.T[1], alpha=0.75, c='blue')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
tot = sum(eigen_vals)
# var_exp ratio is fraction of eigen_val to total sum
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# calculate the cumulative sum of explained variances
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1, 48), var_exp, alpha=0.75, align='center',label='individual explained variance')
plt.step(range(1, 48), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylim(0, 1.1)
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.legend(loc='best')
plt.show()
pca_42 = PCA(n_components=42) #PCA with 8 primary components

# fit and transform both PCA models
X_pca_42 = pca_42.fit_transform(X)
print(X_pca_42.shape)

X2_train,X2_test,Y2_train,Y2_test=train_test_split(X_pca_42,Y,test_size=0.2,random_state=0)
regressor2= LinearRegression(copy_X=True,fit_intercept=True,n_jobs=1,normalize=True)
regressor2.fit(X2_train,Y2_train)
## Predictiions for TEST
Y2_pred=regressor2.predict(X2_test)
ms_error2 = metrics.mean_squared_error(Y2_test, Y2_pred)
print(ms_error2)
regressor2.score(X2_test,Y2_test)
kfold=KFold(n_splits=10, shuffle=False, random_state=None)
scores2 = cross_val_score(regressor2, X_pca_42, Y,cv=kfold)
print(scores2)
sns.residplot(Y1_pred,Y1_test)
sns.residplot(Y2_pred,Y2_test)