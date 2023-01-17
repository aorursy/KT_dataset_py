
import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv("../input/dp-house-price/House_Price (1).csv", header = 0)
df.head()
df.shape
#always look for EDD properly before starting analysis.
df.describe()
sns.jointplot(x='n_hot_rooms', y ='price', data = df )
sns.jointplot(x='rainfall', y= 'price', data = df)
df.head()
sns.countplot(x='airport',data=df)
sns.countplot(x='waterbody', data=df)
sns.countplot( x= 'bus_ter', data=df)
#this variable data of no use as it wont impact the modle in any way.
df.info()
# Capping and Flooring metod on outliers
np.percentile(df.n_hot_rooms,[99])
np.percentile(df.n_hot_rooms, [99])[0]
uv = np.percentile(df.n_hot_rooms,[99])[0]
df[(df.n_hot_rooms >uv)]
# capping the Value
df.n_hot_rooms[(df.n_hot_rooms > 3*uv)] = 3*uv
np.percentile(df.rainfall,[1])[0]
lv = np.percentile(df.rainfall,[1])[0]
df[(df.rainfall < lv)]
df.rainfall[(df.rainfall <0.3*lv)] = 0.3*lv
df[(df.rainfall < lv)]
sns.jointplot(x="crime_rate", y='price', data=df)
df.describe()
df.info()
# imputing missing values
df.n_hos_beds = df.n_hos_beds.fillna(df.n_hos_beds.mean())
df.info()
# missing value imputation for all the columns (if needed)
# df=df.fillna(df.mean())
sns.jointplot( x= "crime_rate", y= "price", data= df)
# to make lgorithmic curve to tranform to linear relation betwem X & Y we would take log of X varilable to get the same.
# adding a value of 1 to crime rate.
df.crime_rate = np.log(1 + df.crime_rate) 

sns.jointplot( x= "crime_rate", y= "price", data= df)
# Creating Avg Variable to covey all figures of Distance "dist " in one.
df[ 'avg_dist']= (df.dist1+df.dist2+df.dist3+df.dist4)/4
df.describe()
del df['dist1']
df.describe()
del df['dist2']
del df['dist3']

df.describe()
del df['bus_ter']
df.head()
df = pd.get_dummies(df)
df.head()
del df['airport_NO']
del df['waterbody_None']
df.head()
# Coorelation Matrics to understand which variable is significant and which is not.
df.corr()
del df['parks']

df.head()
import statsmodels.api as sn
x = sn.add_constant(df['room_num'])
lm = sn.OLS(df['price'], x).fit()    #where OLS stands for ordinary least square
lm.summary()
from sklearn.linear_model import LinearRegression
y = df['price']
x = df[['room_num']]
lm2 = LinearRegression()
lm2.fit(x,y)
print(lm2.intercept_, lm2.coef_)
lm2.predict(x)
# help(sns.jointplot)
sns.jointplot(x = df['room_num'], y = df['price'], data =df, kind = 'reg')
#Multiple LR model using Stats model
x_multi = df.drop("price", axis=1 )
x_multi.head()
y_multi = df["price"]
y_multi.head()
x_multi_cons = sn.add_constant(x_multi)
x_multi_cons.head()
lm_multi = sn.OLS(y_multi, x_multi_cons).fit()
lm_multi.summary()
# Multiple LR model using SKLearn 

lm3 = LinearRegression()
lm3.fit(x_multi, y_multi)
print(lm3.intercept_, lm3.coef_)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x_multi, y_multi, test_size = 0.2, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
lm_a = LinearRegression()
lm_a.fit(x_train,y_train)
y_test_a = lm_a.predict(x_test) 
y_train_a = lm_a.predict(x_train)
from sklearn.metrics import r2_score
r2_score(y_test, y_test_a)
r2_score(y_train, y_train_a)
# we need to standardise our data before runing Rig & laso 
# we shall import Preprocessing from SKLearn
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)
from sklearn.linear_model import Ridge
lm_r = Ridge(alpha = 0.5)
lm_r.fit(x_train_s,y_train )
r2_score(y_test, lm_r.predict(x_test_s))
# importing Validation Curve from sklearnLibrary to find out highest r2 value
from sklearn.model_selection import validation_curve
#validation_curve?
param_range = np.logspace(-2, 8,100)
param_range
train_scores, test_scores = validation_curve(Ridge(),x_train_s, y_train, "alpha", param_range, scoring = 'r2')
print(train_scores)
print(test_scores)
train_mean = np.mean(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis=1)
train_mean
max(test_mean)
sns.jointplot(x=np.log(param_range), y=test_mean)
# to find the location of our value 0.7386...
np.where(test_mean == max(test_mean))

param_range[31]
lm_r_best = Ridge(alpha = param_range [31])
lm_r_best.fit(x_train_s, y_train)
r2_score(y_test, lm_r_best.predict(x_test_s))
r2_score(y_train, lm_r_best.predict(x_train_s))
from sklearn.linear_model import Lasso
lm_l = Lasso (alpha = 0.4)