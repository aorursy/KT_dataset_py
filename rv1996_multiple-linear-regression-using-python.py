#standard import for importing the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.metrics import r2_score
# ignoring all the warning the we get
import warnings
warnings.filterwarnings('ignore')
#Reading the data set
data_url = "/kaggle/input//housing-simple-regression/Housing.csv"

housing = pd.read_csv(data_url)
housing.head()
housing.info()
housing.shape
housing.describe()
# ploting the dataset
# numerical variable
sns.pairplot(housing)
# visualizng the categorical variable
# make a box plot between continous varaible and categorical variable
plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
sns.boxplot(x="mainroad",y="price",data=housing)

plt.subplot(2,3,2)
sns.boxplot(x="airconditioning",y="price",data=housing)

plt.subplot(2,3,3)
sns.boxplot(x="furnishingstatus",y="price",data=housing)

plt.subplot(2,3,4)
sns.boxplot(x="guestroom",y="price",data=housing)

plt.subplot(2,3,5)
sns.boxplot(x="basement",y="price",data=housing)

plt.subplot(2,3,6)
sns.boxplot(x="prefarea",y="price",data=housing)
plt.figure(figsize=(10,5))
sns.boxplot(x="furnishingstatus",hue="airconditioning", y="price",data=housing)
# we can infer that usually having an airconditioned increases the price of house as compared to not having it
# convert the binary categorical variable to 1 or 0 or we can have one hot encoding
housing.columns
var_list = ['mainroad','guestroom', 'basement',
            'hotwaterheating', 'airconditioning', 'prefarea']

# one way doing it is mentioned below
# for var in var_list:
#     housing[var] = housing[var].apply(lambda x: 1 if x=="yes" else 0)

# can also be done by subseting the dataset
housing[var_list] = housing[var_list].apply(lambda x: x.map({"yes":1,"no":0}))
housing.head()
# converting furnishing status to one hot encoding or dummy variable
status = pd.get_dummies(housing['furnishingstatus'])
status.head()
# dropping the redundent columns
status = pd.get_dummies(housing['furnishingstatus'],drop_first=True)
status.head()
# concat the dummy data to housing

housing = pd.concat([housing,status],axis=1)
housing = housing.drop(['furnishingstatus'],axis=1)
housing.head()
# performing the train test split
df_train, df_test = train_test_split(housing,train_size=0.7,random_state=100)
print(df_train.shape)
print(df_test.shape)
housing.columns
# sklearn geneally comes with 3 types of methods for preprocessing MinMaxScaler
# fit() learn, will just calculate min and max values
# transform() x - xmin/(xmax - xmin)
# fit_tranform() does above two in just one step
# create class object
scaler = MinMaxScaler()

#create a list of only numeric variable
scaler_list = ['area','bedrooms','bathrooms','stories','parking','price']

# fit the scaler in training data set
df_train[scaler_list] = scaler.fit_transform(df_train[scaler_list])
df_train.head()
# how many of the features show we choose for optimum model training
#plotting a heat map to understand the correlation among feature
plt.figure(figsize=(14,10))
sns.heatmap(df_train.corr(), annot= True, cmap='YlGnBu')
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
y_train = df_train.pop('price')
X_train = df_train
X_train.head()
# for every new feature varaible we add we see the following
# - signification of the variable 
# - if varaible is correlated we'll look at VIF

#only using area for now
X_train_sm = sm.add_constant(X_train['area'])

lr = sm.OLS(y_train, X_train_sm)

lr_model = lr.fit()
lr_model.params
lr_model.summary()
# now we add another variable and see the result

X_train_sm = X_train[['area','bathrooms']]
X_train_sm = sm.add_constant(X_train_sm)

lr = sm.OLS(y_train,X_train_sm)

lr_model = lr.fit()
lr_model.summary()
# now we add another variable bedrooms


X_train_sm = X_train[['area','bathrooms','bedrooms']]
X_train_sm = sm.add_constant(X_train_sm)

lr = sm.OLS(y_train,X_train_sm)

lr_model = lr.fit()
lr_model.summary()
# now we ada all the varaibles

X_train_sm = sm.add_constant(X_train)

lr = sm.OLS(y_train,X_train_sm)

lr_model = lr.fit()
lr_model.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif_calculate(X_df):
    vif = pd.DataFrame()
    vif['Features'] = X_df.columns
    vif['vif'] = [variance_inflation_factor(X_df.values,i) for i in range(X_df.shape[1])]
    vif['vif'] = round(vif['vif'],2)
    vif = vif.sort_values(by="vif",ascending=False)
    return vif
def train_model(y_train,X_train):
    X_train_sm = sm.add_constant(X_train)

    lr = sm.OLS(y_train,X_train_sm)

    lr_model = lr.fit()
    return lr_model
vif_calculate(X_train)
# we usually stick with vif less than 5
# calculate everything once again and eliminate the feature base on the above rules
X_train = X_train.drop('semi-furnished',axis=1)
X_train.head()
train_model(y_train,X_train).summary()
vif_calculate(X_train)
# again doing the eleminating step
# bedrooms have high p-value so we eliminat it
X_train = X_train.drop('bedrooms',axis=1)
X_train_sm = sm.add_constant(X_train)
lr_model = train_model(y_train,X_train)
lr_model.summary()
vif_calculate(X_train) # now almost most of the every feature is below 5 so this could be our final model
y_train_pred = lr_model.predict(X_train_sm)

res = y_train - y_train_pred
res
#Distribution of the error terms - it should have a normal distribution
sns.distplot(res)
# same transformation needs to be on the training set also
# we never perform fit() operation on the test set
# we only transform() on the dataset

df_test[scaler_list] = scaler.transform(df_test[scaler_list])
df_test.head()
df_test.describe()
y_test = df_test.pop('price')
X_test = df_test
X_test_sm = sm.add_constant(X_test)
X_test_sm
X_test_sm = X_test_sm.drop(['semi-furnished','bedrooms'],axis=1)
X_test_sm
y_test_pred = lr_model.predict(X_test_sm)
y_test_pred
r2_score(y_true=y_test,y_pred=y_test_pred)
import sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df_train, df_test =train_test_split(housing,train_size=0.7,random_state=100)
scaler = MinMaxScaler()
scaler_list = ['area','bedrooms','bathrooms','stories','parking','price']

# fit the scaler in training data set
df_train[scaler_list] = scaler.fit_transform(df_train[scaler_list])
df_train.head()
y_train = df_train.pop('price')
X_train = df_train
# setting dimension of y varriable
y_train = y_train.values.reshape(-1,1)
lr = LinearRegression()
lr.fit(y_train, X_train)
rfe = RFE(lr,10) # we choose have the best 10 variable 
rfe = rfe.fit(X_train,y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# our 10 best features would be
X_train.columns[rfe.support_].tolist() # RFE tells that these variable are really significant
columns_to_drop = X_train.columns[~rfe.support_]
columns_to_drop 
X_train.drop(columns_to_drop,axis=1,inplace=True)
vif_calculate(X_train)
# using the statsmodel

X_train_sm = sm.add_constant(X_train)

lr = sm.OLS(y_train,X_train)

lr_model = lr.fit()
lr_model.summary()
X_train = X_train.drop('bedrooms',axis=1)
vif_calculate(X_train)
X_train_sm = X_train_sm.drop('bedrooms',axis=1)
lr = sm.OLS(y_train,X_train)
lr_model = lr.fit()
lr_model.summary()
