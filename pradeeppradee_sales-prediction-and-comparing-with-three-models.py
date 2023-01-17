import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

df=pd.read_csv('../input/sales-prediction-for-big-mart-outlets/train.csv')

#X=df.drop('Item_Outlet_Sales',axis=1)

#y=df['Item_Outlet_Sales']

#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error

import seaborn as sns
import matplotlib.pyplot as plt
list(df.columns.values.tolist()) 
df.shape
df.head()
df.isnull().sum()[:]
df.apply(lambda x : len(x.unique()))
df.hist(figsize = (10,10),color='gray')
cat_cols = df.select_dtypes(include=['object']).columns

num_cols = df.select_dtypes(exclude=['object']).columns

#print(df[cat_cols].columns)

#print(df[num_cols].columns)
#check for missing values

import seaborn as sns

df.info()

#df.isnull().sum()

sns.heatmap(df.isnull(), yticklabels=False)
#correlation_matrix

corr_m = df.corr() 

f, ax = plt.subplots(figsize =(7,6)) 

sns.heatmap(corr_m,annot=True, cmap ="YlGnBu", linewidths = 0.1) 
print(df[cat_cols].columns)
#bar charts for cat_columns 
#1_plot(before change)

df['Item_Fat_Content'].value_counts().sort_index().plot.bar()
#1_plot(after_change) #1st change in dataset

#LF,low fat and reg is a correction and should be replaced as Low Fat and Regular 

df['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'},inplace = True)

df['Item_Fat_Content'].value_counts().sort_index().plot.bar()
#2_plot

sns.catplot('Item_Type',kind = 'count',data = df,aspect =4)
#3_plot

sns.catplot('Outlet_Identifier',kind = 'count',data = df,aspect =2)
#4_plot (missing values)

sns.catplot('Outlet_Size',kind = 'count',data = df,aspect =2)
#5_plot

sns.catplot('Outlet_Location_Type',kind = 'count',data = df,aspect =4)
#6_plot

sns.catplot('Outlet_Type',kind = 'count',data = df,aspect =3)
print(df[num_cols].columns)
#bar charts for num_columns 
#7_plot

sns.catplot('Outlet_Establishment_Year',kind = 'count',data = df,aspect =3)
#from pandas.plotting import scatter_matrix

#fig, ax = plt.subplots(figsize=(12,12))

#scatter_matrix(df, alpha=1, ax=ax)
#8_plot

sns.scatterplot(x = 'Item_Visibility',y = 'Item_Outlet_Sales',data = df,alpha = 0.5);
#In above plot, there are more than 500 data points at 0 which doesn't make sense. so considering them as missing values and imputing with mean values.

#2nd change in dataset

a= df[df['Item_Visibility']!=0]['Item_Visibility'].mean()

df['Item_Visibility'] = df['Item_Visibility'].replace(0.00,a)

sns.scatterplot(x = 'Item_Visibility',y = 'Item_Outlet_Sales',data = df,alpha = 0.5);
df['Item_Weight'].isnull().sum(), df['Outlet_Size'].isnull().sum()
#1_missing value treated

#from sklearn.impute import KNNImputer

#kn= KNNImputer(weights='distance')

#a= kn.fit_transform(df["Item_Weight"].values.reshape(-1,1))

#df["Item_Weight"]=a
df["Item_Weight"]=df["Item_Weight"].fillna(np.mean(df["Item_Weight"]))
df["Item_Weight"].isnull().sum()
df["Outlet_Size"].isnull().sum()
#check for mode

df["Outlet_Size"].mode()
#2_missing value treated

df["Outlet_Size"] = df['Outlet_Size'].replace(np.nan, 'Medium')

df["Outlet_Size"].isnull().sum()
sns.heatmap(df.isnull(), yticklabels=False)
df['Item_Identifier'].head(20)

#all values in 'Item_Identifier' column can be catogerized into 3: FD, DR, NC
df['Item_Category_Id'] =df['Item_Identifier'].replace({'^FD[A-Z]*[0-9]*':'FD','^DR[A-Z]*[0-9]*':'DR','^NC[A-Z]*[0-9]*':'NC'},regex = True)
sns.catplot('Item_Category_Id',kind = 'count',data = df,aspect =3)
sns.scatterplot(x = 'Item_Category_Id',y = 'Item_Type',data = df,alpha = 0.5);
#from the below graph we can say that NC type of food cannot be LowFat.

sns.scatterplot(x = 'Item_Category_Id',y = 'Item_Fat_Content',data = df,alpha = 0.5);
df['Outlet_Age_Years'] = 2020-df['Outlet_Establishment_Year']
df['Item_Type'].unique()
Breakfast = ["Breads", "Breakfast", "Dairy", "Fruits and Vegetables","Snack Foods"]

Drinks= ["Soft Drinks", "Hard Drinks","Canned"]

NV=["Meat","Frozen Foods","Seafood"]

Others=["Household","Baking Goods","Health and Hygiene","Others","Starchy Foods"]
items_list=[]

for i in df['Item_Type']:

    if i in Breakfast:

        items_list.append('Breakfast')

    elif (i in Drinks):

        items_list.append('Drinks')

    elif (i in NV):

        items_list.append('NV')    

    elif (i in Others):

        items_list.append('Others')      

df['Item_Type_new'] = items_list
cat_cols
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Outlet_Size']= le.fit_transform(df['Outlet_Size'])

df['Outlet_Location_Type'] = le.fit_transform(df['Outlet_Location_Type'])

df['Item_Fat_Content'] = le.fit_transform(df['Item_Fat_Content'])

df['Outlet_Identifier'] = le.fit_transform(df['Outlet_Identifier'])

#df['Item_Type'] = le.fit_transform(df['Item_Type'])

df['Outlet_Type'] = le.fit_transform(df['Outlet_Type'])

df['Item_Identifier'] = le.fit_transform(df['Item_Identifier'])

df['Item_Type_new'] = le.fit_transform(df['Item_Type_new'])
df['Item_Category_Id'] = le.fit_transform(df['Item_Category_Id'])
df
df = pd.get_dummies(df, columns=['Item_Category_Id','Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',

                     'Item_Type_new','Outlet_Identifier'])
#df['Item_MRP'] = np.log(df['Item_MRP'])

#df['Item_Visibility'] = np.log(df['Item_Visibility'])
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

min_max_scaler = preprocessing.MinMaxScaler()
df['Item_Identifier'] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df['Item_Identifier'])))
df.head()
#df.info()
X=df.drop(['Item_Outlet_Sales','Outlet_Establishment_Year','Item_Identifier','Item_MRP','Item_Type'],axis=1)

y=df['Item_Outlet_Sales']
#model_check_1

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from math import sqrt

from sklearn.linear_model import LinearRegression
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

L_R= LinearRegression()

L_R.fit(X_train,y_train)

y_pred=L_R.predict(X_test)

r2scores= r2_score(y_test,y_pred)

rmses= sqrt(mean_squared_error(y_test,y_pred))

print("r2scores : ",r2scores)

print("rmses : ",rmses)

L_R.score(X_train,y_train),L_R.score(X_test,y_test)

print(L_R.intercept_)
df.apply(lambda x : len(x.unique()))
sns.scatterplot(x = 'Item_MRP',y = 'Item_Visibility',data = df,alpha = 0.5);
sns.scatterplot(x = 'Item_MRP',y = 'Item_Outlet_Sales',data = df,alpha = 0.5);
def clusters(x):

    if x<70:

        return 'a'

    elif x in range(70,135):

        return 'b'

    elif x in range(135,200):

        return 'c'

    else:

        return 'd'

df['Item_MRP_Clusters'] = df['Item_MRP'].astype('int').apply(clusters)

df['Item_MRP_Clusters'] = le.fit_transform(df['Item_MRP_Clusters'])
X=df.drop(['Item_Outlet_Sales','Outlet_Establishment_Year','Item_Identifier','Item_MRP','Item_Type'],axis=1)

y=df['Item_Outlet_Sales']
#model_check_2

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

L_R= LinearRegression()

L_R.fit(X_train,y_train)

y_pred=L_R.predict(X_test)

r2scores= r2_score(y_test,y_pred)

rmses= sqrt(mean_squared_error(y_test,y_pred))

print("r2scores : ",r2scores)

print("rmses : ",rmses)

L_R.score(X_train,y_train),L_R.score(X_test,y_test)

print(L_R.intercept_)
sns.scatterplot(x = 'Item_Visibility',y = 'Item_Outlet_Sales',data = df,alpha = 0.5);
X=df.drop(['Item_Outlet_Sales','Outlet_Establishment_Year','Item_Identifier','Item_MRP','Item_Type'],axis=1)

y=df['Item_Outlet_Sales']
#same_duplicate

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

L_R= LinearRegression()

L_R.fit(X_train,y_train)

y_pred=L_R.predict(X_test)

r2scores= r2_score(y_test,y_pred)

rmses= sqrt(mean_squared_error(y_test,y_pred))

print("r2scores : ",r2scores)

print("rmses : ",rmses)

L_R.score(X_train,y_train),L_R.score(X_test,y_test)

print(L_R.intercept_)
plt.boxplot(df["Item_Visibility"])
print(df['Item_Visibility'].quantile(0.50)) 

print(df['Item_Visibility'].quantile(0.91)) 
df['Item_Visibility'] = np.where(df['Item_Visibility'] > 0.14270015850000003, 0.062516602, df['Item_Visibility'])
plt.boxplot(df["Item_Visibility"])
#model_1 using LinearRegression
X=df.drop(['Item_Outlet_Sales','Outlet_Establishment_Year','Item_Identifier','Item_MRP','Item_Type'],axis=1)

y=df['Item_Outlet_Sales']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

L_R= LinearRegression()

L_R.fit(X_train,y_train)

y_pred=L_R.predict(X_test)

r2scores= r2_score(y_test,y_pred)

print("r2scores : ",r2scores)

L_R.score(X_train,y_train),L_R.score(X_test,y_test)

print(L_R.intercept_)
m_a_e=mean_absolute_error(y_test, y_pred)

print(m_a_e)
#rmse

mse1=mean_squared_error(y_test,y_pred)

L_R_score=np.sqrt(mse1)

L_R_score
#cross_val

score=cross_val_score(L_R,X_train,y_train,cv=10,scoring='neg_mean_squared_error')

L_R_score_cross=np.sqrt(-score)

print(np.mean(L_R_score_cross),np.std(L_R_score_cross))
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1
import seaborn as sns

ax1 = sns.distplot(df1['Actual'], hist=False, color="red", label="Actual Value")

sns.distplot(df1['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)
#model_2 using RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor()

rf.fit(X_train,y_train)

y_pred2=rf.predict(X_test)
mse2=mean_squared_error(y_test,y_pred2)

rf_score=np.sqrt(mse2)

rf_score
r2scores_2= r2_score(y_test,y_pred2)

r2scores_2
#cross_val

rf=RandomForestRegressor()

score=cross_val_score(rf,X_train,y_train,cv=10,scoring='neg_mean_squared_error')

rf_score_cross=np.sqrt(-score)

print(np.mean(rf_score_cross),np.std(rf_score_cross))
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})

df2
import seaborn as sns

ax1 = sns.distplot(df2['Actual'], hist=False, color="red", label="Actual Value")

sns.distplot(df2['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)
#model_3 using Lasso
#important_features (for ref only)

from sklearn.linear_model import LassoCV

m_l = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])

m_l.fit(X_train, y_train)

#coef = pd.Series(m_l.coef_, index = X_train.columns)

#imp_features = coef.index[coef!=0].tolist()
y_pred3 = m_l.predict(X_test)

mse3=mean_squared_error(y_test,y_pred3)

mse3
score=np.sqrt(mse3)

score
r2scores_3= r2_score(y_test,y_pred3)

r2scores_3
df3 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred3})

df3
import seaborn as sns

ax1 = sns.distplot(df3['Actual'], hist=False, color="red", label="Actual Value")

sns.distplot(df3['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)
mse1,mse2,mse3
L_R_score, rf_score ,score
F_scores = {'Model':  ['L_R', 'RF_R','LASSO'],

         'RMSE': [L_R_score, rf_score ,score],

            'R2': [r2scores,r2scores_2,r2scores_3]}
df_scores = pd.DataFrame (F_scores, columns = ['Model','RMSE','R2'])

df_scores