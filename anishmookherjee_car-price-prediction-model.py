import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')

df.head()
df.shape
df.info()
df.describe()


df.drop('car_ID',axis=1,inplace=True)

df.head()
print(df.isnull().values.any())
df.dtypes
df['CarName']=df['CarName'].str.split(' ',expand=True)

df['CarName'].head()
df['CarName'].unique()


df['CarName']=df['CarName'].replace({'maxda':'mazda',

                                     'nissan':'Nissan',

                                     'toyouta':'toyota',

                                     'porcshce':'porsche',

                                     'vokswagen':'volkswagen',

                                     'vw':'volkswagen'

                                     })
df['symboling']=df['symboling'].astype(str)

df['symboling'].head()
df.loc[df.duplicated()]
cat_col=df.select_dtypes(include='object').columns

num_col=df.select_dtypes(exclude='object').columns

df_cat=df[cat_col]

df_num=df[num_col]
df_cat.head(2)
df_num.head(2)
df['CarName'].value_counts()
plt.figure(figsize=(10, 10))

ax=df['CarName'].value_counts().plot(kind='bar')

plt.title(label='CarName')

plt.xlabel("Names of the Car",fontweight = 'bold')

plt.ylabel("Count of Cars",fontweight = 'bold')

plt.show()
ax=sns.pairplot(df[num_col])

plt.show()
plt.figure(figsize=(20, 15))

plt.subplot(3,3,1)

sns.boxplot(x = 'doornumber', y = 'price', data = df)

plt.subplot(3,3,2)

sns.boxplot(x = 'fueltype', y = 'price', data = df)

plt.subplot(3,3,3)

sns.boxplot(x = 'carbody', y = 'price', data = df)

plt.subplot(3,3,4)

sns.boxplot(x = 'drivewheel', y = 'price', data = df)

plt.subplot(3,3,5)

sns.boxplot(x = 'enginelocation', y = 'price', data = df)

plt.subplot(3,3,6)

sns.boxplot(x = 'cylindernumber', y = 'price', data = df)

plt.subplot(3,3,7)

sns.boxplot(x = 'enginetype', y = 'price', data = df)

plt.subplot(3,3,8)

sns.boxplot(x = 'fuelsystem', y = 'price', data = df)

plt.subplot(3,3,9)

sns.boxplot(x = 'aspiration', y = 'price', data = df)

plt.show()
ax=df.groupby(['CarName'])['price'].mean().sort_values(ascending=False)



plt.figure(figsize=(10, 10))

ax.plot.bar()

plt.title('Car Company Name vs Average Price')

plt.show()
ax=df.groupby(['carbody'])['price'].mean().sort_values(ascending=False)



plt.figure(figsize=(10, 10))

ax.plot.bar()

plt.title('Car Body Name vs Average Price')

plt.show()
df['price'] = df['price'].astype('int')

df_auto_temp = df.copy()

grouped = df_auto_temp.groupby(['CarName'])['price'].mean()

print(grouped)

df_auto_temp = df_auto_temp.merge(grouped.reset_index(), how='left', on='CarName')

bins = [0,10000,20000,40000]

label =['Budget_Friendly','Medium_Range','TopNotch_Cars']

df['Cars_Category'] = pd.cut(df_auto_temp['price_y'], bins, right=False, labels=label)

df.head()
sig_col = ['price','Cars_Category','enginetype','fueltype', 'aspiration','carbody','cylindernumber', 'drivewheel',

            'wheelbase','curbweight', 'enginesize', 'boreratio','horsepower', 

                    'citympg','highwaympg', 'carlength','carwidth']
df=df[sig_col]
sig_cat_col=['Cars_Category','enginetype','fueltype','aspiration','carbody','cylindernumber','drivewheel']
dummies=pd.get_dummies(df[sig_cat_col])

print(dummies.shape)

dummies.head()
dummies=pd.get_dummies(df[sig_cat_col],drop_first=True)

print(dummies.shape)

dummies.head()
df=pd.concat([df,dummies],axis=1)
df.drop(sig_cat_col,axis=1,inplace=True)

df.shape
df
np.random.seed(0) 



from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, train_size=0.7, test_size = 0.3, random_state = 100)
df_train.head()
from sklearn.preprocessing import StandardScaler 

scaler=StandardScaler()
sig_num_col = ['wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg','price']
df_train[sig_num_col]=scaler.fit_transform(df_train[sig_num_col])
df_train.head()
plt.figure(figsize=(20,20))

sns.heatmap(df_train.corr(), cmap= 'RdYlGn')

plt.show()
y_train=df_train.pop('price')
x_train=df_train
import statsmodels.api as sm



x_train_copy = x_train
x_train_copy1=sm.add_constant(x_train_copy['horsepower'])



#1st model

lr1=sm.OLS(y_train,x_train_copy1).fit()
lr1.params
print(lr1.summary())
from sklearn.linear_model import LinearRegression



lm = LinearRegression()

lm.fit(x_train, y_train)
from sklearn.feature_selection import RFE



rfe=RFE(lm,15)

rfe=rfe.fit(x_train,y_train)
list(zip(x_train.columns,rfe.support_,rfe.ranking_))
col_sup=x_train.columns[rfe.support_]

col_sup
x_train_rfe=x_train[col_sup]

x_train_rfe
import statsmodels.api as sm



x_train_rfec = sm.add_constant(x_train_rfe)

lm_rfe = sm.OLS(y_train,x_train_rfec).fit()



#Summary of linear model

print(lm_rfe.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = x_train_rfe.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe.values, i) for i in range(x_train_rfe.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe1=x_train_rfe.drop('cylindernumber_twelve',axis=1)



x_train_rfe1c=sm.add_constant(x_train_rfe1)

lm_rfe1=sm.OLS(y_train,x_train_rfe1c).fit()



print(lm_rfe1.summary())
vif = pd.DataFrame()

vif['Features'] = x_train_rfe1.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe1.values, i) for i in range(x_train_rfe1.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe2=x_train_rfe1.drop('cylindernumber_six',axis=1)



x_train_rfe2c=sm.add_constant(x_train_rfe2)

lm_rfe2=sm.OLS(y_train,x_train_rfe2c).fit()



print(lm_rfe2.summary())
vif = pd.DataFrame()

vif['Features'] = x_train_rfe2.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe2.values, i) for i in range(x_train_rfe2.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe3=x_train_rfe2.drop('carbody_hardtop',axis=1)



x_train_rfe3c=sm.add_constant(x_train_rfe3)

lm_rfe3=sm.OLS(y_train,x_train_rfe3c).fit()



print(lm_rfe3.summary())
vif = pd.DataFrame()

vif['Features'] = x_train_rfe3.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe3.values, i) for i in range(x_train_rfe3.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe4=x_train_rfe3.drop('enginetype_ohc',axis=1)



x_train_rfe4c=sm.add_constant(x_train_rfe4)

lm_rfe4=sm.OLS(y_train,x_train_rfe4c).fit()



print(lm_rfe4.summary())
vif = pd.DataFrame()

vif['Features'] = x_train_rfe4.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe4.values, i) for i in range(x_train_rfe4.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe5=x_train_rfe4.drop('cylindernumber_five',axis=1)



x_train_rfe5c=sm.add_constant(x_train_rfe5)

lm_rfe5=sm.OLS(y_train,x_train_rfe5c).fit()



print(lm_rfe5.summary())
vif = pd.DataFrame()

vif['Features'] = x_train_rfe5.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe5.values, i) for i in range(x_train_rfe5.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe6=x_train_rfe5.drop('enginetype_ohcv',axis=1)



x_train_rfe6c=sm.add_constant(x_train_rfe6)

lm_rfe6=sm.OLS(y_train,x_train_rfe6c).fit()



print(lm_rfe6.summary())
vif = pd.DataFrame()

vif['Features'] = x_train_rfe6.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe6.values, i) for i in range(x_train_rfe6.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe7=x_train_rfe6.drop('curbweight',axis=1)



x_train_rfe7c=sm.add_constant(x_train_rfe7)

lm_rfe7=sm.OLS(y_train,x_train_rfe7c).fit()



print(lm_rfe7.summary())
vif = pd.DataFrame()

vif['Features'] = x_train_rfe7.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe7.values, i) for i in range(x_train_rfe7.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe8=x_train_rfe7.drop('cylindernumber_four',axis=1)



x_train_rfe8c=sm.add_constant(x_train_rfe8)

lm_rfe8=sm.OLS(y_train,x_train_rfe8c).fit()



print(lm_rfe8.summary())
vif = pd.DataFrame()

vif['Features'] = x_train_rfe8.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe8.values, i) for i in range(x_train_rfe8.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe9=x_train_rfe8.drop('carbody_sedan',axis=1)



x_train_rfe9c=sm.add_constant(x_train_rfe9)

lm_rfe9=sm.OLS(y_train,x_train_rfe9c).fit()



print(lm_rfe9.summary())
vif = pd.DataFrame()

vif['Features'] = x_train_rfe9.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe9.values, i) for i in range(x_train_rfe9.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
x_train_rfe10=x_train_rfe9.drop('carbody_wagon',axis=1)



x_train_rfe10c=sm.add_constant(x_train_rfe10)

lm_rfe10=sm.OLS(y_train,x_train_rfe10c).fit()



print(lm_rfe10.summary())
vif = pd.DataFrame()

vif['Features'] = x_train_rfe10.columns

vif['VIF'] = [variance_inflation_factor(x_train_rfe10.values, i) for i in range(x_train_rfe10.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_pred=lm_rfe10.predict(x_train_rfe10c)
sns.distplot((y_train-y_train_pred),bins=20)

plt.title('Error Term Analysis')

plt.xlabel('Errors')

plt.show()
df_test[sig_num_col]=scaler.transform(df_test[sig_num_col])

df_test.shape
y_test=df_test.pop('price')

x_test=df_test
x_test_1=sm.add_constant(x_test)



x_test_new=x_test_1[x_train_rfe10c.columns]
y_pred=lm_rfe10.predict(x_test_new)
y_pred
from sklearn.metrics import r2_score



r2_score(y_test,y_pred)