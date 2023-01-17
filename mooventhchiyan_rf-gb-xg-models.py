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
# Variable	Definition
# User_ID	User ID
# Product_ID	Product ID
# Gender	Sex of User
# Age	Age in bins
# Occupation	Occupation (Masked)
# City_Category	Category of the City (A,B,C)
# Stay_In_Current_City_Years	Number of years stay in current city
# Marital_Status	Marital Status
# Product_Category_1	Product Category (Masked)
# Product_Category_2	Product may belongs to other category also (Masked)
# Product_Category_3	Product may belongs to other category also (Masked)
# Purchase	Purchase Amount (Target Variable)
data= pd.read_csv("/kaggle/input/black-friday/train.csv")
data.head()
data.shape
 # Target-- Purchase

 # Data understanding and cleaning
 # EDA
 # Base Model
 # Feature Selection - ANOVA, CHI-SQUARE
 # Try different models
 # Parametric Tuning    - RMSE
# Checking
data.info()
# Checking the null values in the dataset
data.isnull().sum()/data.shape[0] *100
# Only product_category_1 and product_category_2 have null values
# Denoting none of the customers have purchased the product- Let's replace that with '0'
data['Product_Category_2'].fillna(0,inplace=True)
data['Product_Category_3'].fillna(0,inplace=True)
data.isnull().sum()/data.shape[0] *100
print("Total number of USER_ID: ", data['User_ID'].nunique())
# It seems a repeadted purchases on the same user id as it near to 6000 while the data is for 5 lakhs
# Other possiblitity only 1% have a unique user_id
data['User_ID'].value_counts().plot(kind='hist')
data.groupby(['User_ID'])['Purchase'].sum().sort_values(ascending=False)[:10].plot(kind='bar')
plt.xlabel("User_ID")
plt.ylabel("sum purchase in millions")
plt.show()
# We need to target the user-ID "1004277" for more increase in the sales
# Product_ID
print("Total number of product_id :",data['Product_ID'].nunique())
data['Product_ID'].value_counts().plot(kind='hist')  # Only certain Product are contributing more
ss= data['Product_ID'].value_counts()[:10]
ss.plot(kind='bar')# Count wise product_id 
plt.xlabel("Product_id")
plt.ylabel("Count_of_Id")
plt.show()
data.groupby(['Product_ID'])['Purchase'].sum().plot(kind='hist')
data.groupby(['Product_ID'])['Purchase'].sum().sort_values(ascending=False)[:10].plot(kind='bar')
plt.xlabel("Product_id")
plt.ylabel("sum purchase in millions")
plt.show()
data_Sex = data.groupby('Gender')['Gender'].count()
data_Sex = pd.DataFrame({'Sex':data_Sex.index, 'Count':data_Sex.values})
plt.pie(data_Sex['Count'],labels = data_Sex['Sex'],autopct='%1.1f%%',shadow=True);
plt.title('Gender Split in data');

print(data.groupby(['Gender'])['Purchase'].sum())
data_GP=data.groupby(['Gender'])['Purchase'].sum()
plt.pie(data_GP,autopct='%1.1f%%',labels=['Female','Male'])
plt.show()
sns.countplot(data['Gender'],hue=data["Age"])
# Count of Male and Purchase sum is high  --- so we need to focus on them more
sns.countplot(data['Age'])
plt.title("Age distribution")
plt.show()
data_Age = data.groupby('Age')['Age'].count()
data_Age = pd.DataFrame({'Age':data_Age.index, 'Count':data_Age.values})
plt.pie(data_Age['Count'],labels = data_Age['Age'],autopct='%1.1f%%',shadow=True);
plt.title('Age split in data');
plt.show()
data.groupby('Age')['Purchase'].mean().plot()
plt.xlabel('Age group')
plt.ylabel('Average_Purchase amount in $')
plt.title('Age group vs average amount spent')
plt.show()
### If you observe here the puchase in the age group of 51-55 is comparatively higher with only 7%
sns.countplot(data['Age'],hue=data["Marital_Status"])
# 1 married and 0 unmarried
sns.countplot(data['Age'],hue=data["Gender"])
print("City wise Contribution", data['City_Category'].value_counts(normalize=True) *100)
sns.countplot(data['City_Category'])
data.groupby('City_Category')['Purchase'].mean().plot()
data['Occupation'].value_counts().plot(kind='bar')
OS= data.groupby(['Occupation'])['Purchase'].mean()
plt.plot(OS.index,OS.values,'ro-')
plt.xticks(OS.index)
plt.xlabel('Occupation types')
plt.ylabel('Average purchase amount in $')
plt.title('Average amount of purchase for each Occupation')
plt.show()
# Inference
# Number of more counts in occuptation doesn't contribute more in the purchase amount
# Mean value of purchase value for occuptation 8 & 15 is more compartievly to the number of counts(Heavy Spenders)
# More effort on the less occupation (8&15) coulld generate more purchases
# Occupation 11 to 18 looks like a target are to focus in terms of raising puchases
# On other hand We can concentrate is there a possiblity of increasing the more count occupation to contribute to purchase
plt.figure(figsize=[12,8])
sns.countplot(data['Occupation'],hue=data["Age"])
data['Stay_In_Current_City_Years'].value_counts().plot()
data1= data.groupby('Stay_In_Current_City_Years')['Purchase'].sum().reset_index()
data2= data['Stay_In_Current_City_Years'].value_counts()
data2=pd.DataFrame({"Stay_In_Current_City_Years":data2.index, "Count":data2.values})
nw_data = pd.merge(data1,data2,left_on='Stay_In_Current_City_Years',right_on='Stay_In_Current_City_Years',how = 'left');

nw_data = nw_data.sort_values(['Stay_In_Current_City_Years'],ascending=False)[0:10];
nw_data
    
plt.figure(figsize=(16,6));
plt.grid();
plt.plot(nw_data['Stay_In_Current_City_Years'],nw_data['Purchase'],'o-');
plt.xlabel('Stay_In_Current_City_Years');
plt.ylabel('Total amount it was purchased in 10\'s of Million $');
plt.title('Stay wise connection with purchases');
for a,b,c in zip(nw_data['Stay_In_Current_City_Years'], nw_data['Purchase'], nw_data['Count']): 
    plt.text(a, b+100000, str(c))  
plt.show();

data.groupby(['Gender','Marital_Status'])['Purchase'].count().plot(kind='pie',figsize=(8,8))
print("Count of martial_status", data.groupby(['Marital_Status'])['Purchase'].count())
print("Average purchase amount", data.groupby(['Gender','Marital_Status'])['Purchase'].mean())
data.groupby(['Marital_Status'])['Purchase'].mean().plot(kind='bar')
data.groupby(['Gender','Marital_Status'])['Purchase'].mean().unstack().plot(kind='bar')
plt.show()

data['Product_Category_1'].value_counts().plot(kind= 'bar', figsize=(16,5))
PC1= data.groupby('Product_Category_1')['Purchase'].mean()
plt.figure(figsize=(12,8))
plt.plot(PC1.index,PC1.values,'ro-')
plt.xticks(PC1.index)
plt.show()
data['Product_Category_2'].value_counts().plot(kind= 'bar', figsize=(16,5))
PC2= data.groupby('Product_Category_2')['Purchase'].mean()
plt.figure(figsize=(12,8))
plt.plot(PC2.index,PC2.values,'ro-')
plt.xticks(PC2.index)
plt.show()
data['Product_Category_3'].value_counts().plot(kind= 'bar', figsize=(16,5))
PC3= data.groupby('Product_Category_3')['Purchase'].mean()
plt.figure(figsize=(12,8))
plt.plot(PC3.index,PC3.values,'ro-')
plt.xticks(PC3.index)
plt.show()
data['Purchase'].plot(kind='hist')

# For the base level -- Creating a copy droping the null values
df= data.copy(deep=True)
df.info()
# Product_categories are int
#df[['Product_Category_2','Product_Category_3']]=df[['Product_Category_2','Product_Category_3']].astype('int')
# Stay_in_city
df['Stay_In_Current_City_Years'].replace({'4+':4},inplace=True)
# df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype('int')
# Gender
df['Gender'].replace({"M":1,"F":0},inplace=True)
# Age
def map_age(age):
    if age == '0-17':
        return 0
    elif age == '18-25':
        return 1
    elif age == '26-35':
        return 2
    elif age == '36-45':
        return 3
    elif age == '46-50':
        return 4
    elif age == '51-55':
        return 5
    else:
        return 6
df['Age']=df['Age'].apply(map_age)
print(df['Product_Category_2'].describe())
print("--------------------------------")
print(df['Product_Category_3'].describe())
# Mean and Median are some _what close to each other
# Hence filling the null values with the mean
df['Product_Category_2']=df['Product_Category_2'].fillna(9.0).astype(int)
df['Product_Category_3']=df['Product_Category_3'].fillna(13.0).astype(int)
df['City_Category'].value_counts()
# Mapping the City_Category 

df['City_Category']=df['City_Category'].map({"B":1,"A":2,"C":3})
df['City_Category']= df['City_Category'].astype(int)
df['Stay_In_Current_City_Years']= df['Stay_In_Current_City_Years'].astype(int)
## Making a copy if these needs to included again

ddf=df.copy()
df = df.drop(["User_ID","Product_ID"],axis=1)
# As it contains more number of unique values
# X and Y split -- train_test_split

from sklearn.model_selection import train_test_split
X = df.drop("Purchase",axis=1)
y = df['Purchase']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Base Model- Decision Tree

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
dtr= DecisionTreeRegressor()
dtr.fit(X_train,y_train)
d_predict= dtr.predict(X_test)
print("RMSE score for Decision Tree : ", np.sqrt(mean_squared_error(y_test,d_predict)))
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
rfc=RandomForestRegressor(n_estimators=150)
gbr=GradientBoostingRegressor()
xg=XGBRegressor()
rfc.fit(X_train, y_train)
r_predict= rfc.predict(X_test)
gbr.fit(X_train,y_train)
g_predict= gbr.predict(X_test)
xg.fit(X_train, y_train)
xg_predict= xg.predict(X_test)
print("RMSE score for Random_Forest : ", np.sqrt(mean_squared_error(y_test,r_predict)))
print("RMSE score for Gradient Boosting : ", np.sqrt(mean_squared_error(y_test,g_predict)))
print("RMSE score for Gradient Boosting : ", np.sqrt(mean_squared_error(y_test,xg_predict)))
import statsmodels.api as sm
#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.04):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
# Back_ward Elimination is predicting all as an important features
# RFE method of feature selection
from sklearn.feature_selection import RFE
rfc=RandomForestRegressor()
gbr=GradientBoostingRegressor()
#Initializing RFE model
rfe = RFE(rfc, 7)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model

rfe.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)
X.columns

X= df[['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
rfc.fit(X_train, y_train)
r_predict= rfc.predict(X_test)
print("RMSE score for Random_Forest : ", np.sqrt(mean_squared_error(y_test,r_predict)))
gbr.fit(X_train,y_train)
g_predict= gbr.predict(X_test)
print("RMSE score for Gradient Boosting : ", np.sqrt(mean_squared_error(y_test,g_predict)))
xg.fit(X_train, y_train)
xg_predict= xg.predict(X_test)
print("RMSE score for XG Boosting : ", np.sqrt(mean_squared_error(y_test,xg_predict)))
### Performance level have been improvised with the RFE feature selection without including the product_catergory 2 & 3
### But thats the worst thing to do and the best thing to do if you want only product category 1 to be compared
# We have removed the user_id and product_id entirely
# Lets try to map the frequent used and products rather than eliminating all
# Reference to other kaggle notebooks
# Mapping the User_ID based on the importance for the top 20 rather than excluding them totally
user_ids=ddf['User_ID']
counts=user_ids.value_counts().index[:19]
# important_counts=set(counts.index[:19])
# user_ids=user_ids.map(lambda user_id:user_id if user_id  in important_counts else 0)
#from sklearn.preprocessing import OneHotEncoder
#user_id_encoder=OneHotEncoder(categories ='auto')
#user_id_encoder.fit(user_ids.values.reshape(-1,1))
ddf['User_ID']=ddf['User_ID'].map(lambda user_id:user_id if user_id  in counts else 0)
# Product_ID
product_means=ddf.groupby(["Product_ID"])["Purchase"].mean()
total_mean=ddf['Purchase'].mean()
product_means.sort_values
pid=product_means.sort_values(ascending=False).index[:19]
ddf['Product_ID']=ddf['Product_ID'].map(lambda product_id:product_id if product_id  in pid else 0)
ddf['User_ID']=ddf['User_ID'].astype('object')
nddf= pd.get_dummies(ddf,drop_first=True)
X=nddf.drop(['Purchase'],1)
y=nddf['Purchase']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)
rfc=RandomForestRegressor()
gbr=GradientBoostingRegressor()
rfc.fit(X_train, y_train)
r_predict= rfc.predict(X_test)
gbr.fit(X_train,y_train)
g_predict= gbr.predict(X_test)
# Trying XGboost for the many sparse matrix- tree algorithm don't perform the best
from xgboost import XGBRegressor
xg=XGBRegressor()
xg.fit(X_train, y_train)
xg_predict= xg.predict(X_test)
print("RMSE score for Random_Forest : ", np.sqrt(mean_squared_error(y_test,r_predict)))
print("RMSE score for Gradient Boosting : ", np.sqrt(mean_squared_error(y_test,g_predict)))
print("RMSE score for XG Boosting : ", np.sqrt(mean_squared_error(y_test,xg_predict)))
## From the normal way of labelling, feature selection and top_20 user_id and product_id
## We got the best score with respect to validation from --> Normal Labeling the model without user_id and product_id rather than taking dummies on the top 20
## Hence we use that particular set for training and predicting on the test_data
## Checking the test_data
test= pd.read_csv("/kaggle/input/black-friday/test.csv")
test.isnull().sum()
print(" Product _category_2 \n " ,test['Product_Category_2'].describe())
print(" Product _category_3 \n " ,test['Product_Category_3'].describe())
test.info()
# Similary Meand and median are colse by.
# taking a mid range to fill the null values
test['Product_Category_2']=test['Product_Category_2'].fillna(9.0).astype(int)
test['Product_Category_3']=test['Product_Category_3'].fillna(13.0).astype(int)
# Stay_In_Current_City_Years
test['Stay_In_Current_City_Years']=test['Stay_In_Current_City_Years'].replace({'4+':4})
test['Stay_In_Current_City_Years']=test['Stay_In_Current_City_Years'].astype(int)
# Age
test['Age']=test['Age'].apply(map_age)
# Gender
test['Gender'].replace({"M":1,"F":0},inplace=True)
# Mapping the City_Category 
test['City_Category']=test['City_Category'].map({"B":1,"A":2,"C":3})
test['City_Category']=test['City_Category'].astype(int)


test.drop(['User_ID','Product_ID'],axis=1,inplace=True)
# Choosing df --> Where USER_ID,PRODUCT_ID are dropped
from sklearn.model_selection import train_test_split
X = df.drop("Purchase",axis=1)
y = df['Purchase']
# Model that performed the best is XG_Boost
xg.fit(X,y)
test_predict=xg.predict(test)
test['Purchase']=test_predict
print("Purchase distribution for the test data", sns.distplot(test['Purchase']))
print("Final prediction for the test data \n ")
test.head()