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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
file = '../input/BlackFriday.csv'
df = pd.read_csv(file)
df.shape
df.head(10)
df.describe()
df.isnull().sum()
age = df['Age'].value_counts()
occupation = df['Occupation'].value_counts()
gender = df['Gender'].value_counts()

labels_1 = df['Age'].unique()
labels_2 = df['Occupation'].unique()
labels_3 = df['Gender'].unique()

f,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,40))
ax1.pie(age,labels=labels_1,autopct='%1.1f%%')
ax1.set_title('Age Breakdown')
ax2.pie(occupation,labels=labels_2,autopct='%1.1f%%')
ax2.set_title('Occupation Breakdown')
ax3.pie(gender,labels=labels_3,autopct='%1.1f%%')
ax3.set_title('Gender Breakdown')
plt.show()
# Purchase distribution
from scipy.stats import norm
price = df.groupby('User_ID')['Purchase'].agg('sum')
plt.figure(figsize=(10,10))
sns.distplot(price)
plt.title('Mean Purchase Price Distribution')
plt.show()
# Marital Status Ratio
df['Marital_Status'].unique()
df['Marital_Status'].value_counts()
#Ratio per city
plt.figure(figsize=(10,8))
sns.countplot(x='City_Category',hue='Marital_Status',data=df)
plt.show()
df[['Product_Category_2','Product_Category_3']] = df[['Product_Category_2','Product_Category_3']].fillna(0).astype(int) 
print('Category 1:',sorted(df['Product_Category_1'].unique()))
print('Category 2:',sorted(df['Product_Category_2'].unique()))
print('Category 3:',sorted(df['Product_Category_3'].unique()))
f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,10))
sns.countplot(x='Product_Category_1',data=df,ax=ax1,color='r')
sns.countplot(x='Product_Category_2',data=df,ax=ax2,color='b')
sns.countplot(x='Product_Category_3',data=df,ax=ax3,color='g')
plt.show()

# Let set a function for this
def age_group(age):
    if (age == '0-17') or (age=='18-25') or (age == '26-35'):
        return 'Young'
    if (age == '36-45') or (age=='46-50') or (age == '51-55'):
        return 'Middle'
    else:
        return 'Old'

df['Age_group'] = df['Age'].apply(age_group)

# Young population product study
f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,8))
young_1 = df[df.Age_group == 'Young']['Product_Category_1'].value_counts()[:3].plot(kind='bar',ax=ax1)
ax1.set_xlabel('Product 1 Sub-Category')
ax1.set_ylabel('Frequency')
ax1.set_title('Young Population')
young_2 = df[df.Age_group == 'Young']['Product_Category_2'].value_counts()[:3].plot(kind='bar',ax=ax2)
ax2.set_xlabel('Product 2 Sub-Category')
ax2.set_ylabel('Frequency')
ax2.set_title('Young Population')
young_3 = df[df.Age_group == 'Young']['Product_Category_3'].value_counts()[:3].plot(kind='bar',ax=ax3)
ax3.set_xlabel('Product 3 Sub-Category')
ax3.set_ylabel('Frequency')
ax3.set_title('Young Population')

f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,8))
middle_1 = df[df.Age_group == 'Middle']['Product_Category_1'].value_counts()[:3].plot(kind='bar',ax=ax1)
ax1.set_xlabel('Product 1 Sub-Category')
ax1.set_ylabel('Frequency')
ax1.set_title('Middle Aged Population')
middle_2 = df[df.Age_group == 'Middle']['Product_Category_2'].value_counts()[:3].plot(kind='bar',ax=ax2)
ax2.set_xlabel('Product 2 Sub-Category')
ax2.set_ylabel('Frequency')
ax2.set_title('Middle Aged Population')
middle_3 = df[df.Age_group == 'Middle']['Product_Category_3'].value_counts()[:3].plot(kind='bar',ax=ax3)
ax3.set_xlabel('Product 3 Sub-Category')
ax3.set_ylabel('Frequency')
ax3.set_title('Middle Aged Population')

f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,8))
old_1 = df[df.Age_group == 'Old']['Product_Category_1'].value_counts()[:3].plot(kind='bar',ax=ax1)
ax1.set_xlabel('Product 1 Sub-Category')
ax1.set_ylabel('Frequency')
ax1.set_title('Old Age Population')
old_2 = df[df.Age_group == 'Old']['Product_Category_2'].value_counts()[:3].plot(kind='bar',ax=ax2)
ax2.set_xlabel('Product 1 Sub-Category')
ax2.set_ylabel('Frequency')
ax2.set_title('Old Age Population')
old_3 = df[df.Age_group == 'Old']['Product_Category_3'].value_counts()[:3].plot(kind='bar',ax=ax3)
ax3.set_xlabel('Product 1 Sub-Category')
ax3.set_ylabel('Frequency')
ax3.set_title('Old Aged Population')

plt.show()
# Correlation matrix between features.
corr = df[['Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3','Purchase']].corr()
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values, annot=True)
plt.show()
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
product_enc = LabelEncoder()
df['User_ID'] = label_enc.fit_transform(df.User_ID)
df['Product_ID'] = product_enc.fit_transform(df.Product_ID)

# One Hot Encoding Age, Stay in Current City Years, City_Category
df_Age = pd.get_dummies(df.Age)
df_city = pd.get_dummies(df.City_Category)
df_staycity = pd.get_dummies(df.Stay_In_Current_City_Years)
df_Gender = pd.Series(np.where(df.Gender == 'M',1,0), name='Gender')
df_Agegroup = pd.get_dummies(df.Age_group)

df_new = pd.concat([df,df_Gender,df.Age_group, df_Age, df_city, df_staycity], axis=1)
df_new.drop(['Age','Gender','City_Category','Age_group','Stay_In_Current_City_Years'], axis=1, inplace=True)
df_new = df_new.rename(columns={'0':'Gender'})
# let's sample only half the df_new data
df_sample = df_new.sample(frac=0.05, random_state=100)
X2 = df_sample.drop(['Purchase'], axis=1)
y2 = df_sample.Purchase
# Linear Model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
mod = LinearRegression()

scoring = 'neg_mean_squared_error'
linear_cv = cross_val_score(mod, X2,y2, cv=5, scoring=scoring)
print((-1*linear_cv.mean())**0.5)
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import learning_curve
train_sizes, train_scores, valid_scores = learning_curve(mod, X2, y2, cv=3, scoring='neg_mean_squared_error')

train_scores = (-1*train_scores)**0.5
valid_scores = (-1*valid_scores)**0.5
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(train_sizes,valid_scores_mean,label='valid')
plt.plot(train_sizes,train_scores_mean,label='train')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.3,color="g")
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,valid_scores_mean + valid_scores_std, alpha=0.3, color="b")
plt.xlabel('Number of samples')
plt.ylabel('RMSE')
plt.legend()
plt.show()
# Random Forest Regressor 
from sklearn.ensemble import RandomForestRegressor
mod = RandomForestRegressor()
scoring = 'neg_mean_squared_error'
RF_cv = cross_val_score(mod, X2,y2, cv=5, scoring=scoring)
print((-1*RF_cv.mean())**0.5)
# Random Forest Regressor Learning Curve
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import learning_curve
train_sizes, train_scores, valid_scores = learning_curve(RandomForestRegressor(), X2, y2, cv=3, scoring='neg_mean_squared_error')

train_scores = (-1*train_scores)**0.5
valid_scores = (-1*valid_scores)**0.5
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(train_sizes,valid_scores_mean,label='valid')
plt.plot(train_sizes,train_scores_mean,label='train')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.3,color="g")
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,valid_scores_mean + valid_scores_std, alpha=0.3, color="b")
plt.xlabel('Number of samples')
plt.ylabel('RMSE')
plt.legend()
plt.show()
rf = RandomForestRegressor(n_estimators=100).fit(X2,y2)
f_im = rf.feature_importances_.round(3)
ser_rank = pd.Series(f_im,index=X2.columns).sort_values(ascending=False)
plt.figure()
sns.barplot(y=ser_rank.index,x=ser_rank.values,palette='deep')
plt.xlabel('relative importance')
X2 = df_sample.drop(['User_ID','Product_ID','Purchase'], axis=1)
y2 = df_sample.Purchase

# Random Forest Regressor 
from sklearn.ensemble import RandomForestRegressor
mod = RandomForestRegressor()
scoring = 'neg_mean_squared_error'
RF_cv = cross_val_score(mod, X2,y2, cv=5, scoring=scoring)
print((-1*RF_cv.mean())**0.5)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.20, random_state=123)
xg_reg = xgb.XGBRegressor()
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
DM_train = xgb.DMatrix(X_train,y_train)
DM_test =  xgb.DMatrix(X_test,y_test)
params = {"booster":"gblinear", "objective":"reg:linear"}
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)
preds = xg_reg.predict(DM_test)
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))
