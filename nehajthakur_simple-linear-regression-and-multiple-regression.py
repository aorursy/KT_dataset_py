# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')

df.head()
def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)
train_data ,test_data = train_test_split(df, train_size=0.8,random_state = 3 )
train_data.head()
lr = linear_model.LinearRegression()
X_train = np.array(train_data['sqft_living']).reshape(-1,1)
y_train = np.array(train_data['price']).reshape(-1,1)
lr.fit(X_train, y_train)
coefficient = lr.coef_
intercept = lr.intercept_
print("coefficient: ",lr.coef_)
print("Intercept: ",lr.intercept_)

X_test = np.array(test_data['sqft_living']).reshape(-1,1)
y_test = np.array(test_data['price']).reshape(-1,1)
pred = lr.predict(X_test)

RMSE = np.sqrt(metrics.mean_squared_error(y_test,pred))
print("Root mean square error: ", RMSE)
R2_training = lr.score(X_train,y_train)
R2_test = lr.score(X_test, y_test)
print("R square  for Training set: ",R2_training)
print("R square for Test set: ",R2_test)
# find SSR(Sum of squared regression)
# sum of sqaures of variation of predicted line from the mean of Y axis(y_bar)
Y_hat = pred
Y_bar = y_test.mean()

def SSR(Y_hat, Y_bar):
    return np.square(Y_hat - Y_bar).sum()

# finding SSE(Sum of squared error)
def SSE(Yi,Y_hat):
    return np.square(Yi - Y_hat).sum()
Rsquared = SSR(Y_hat, Y_bar)/ (SSR(Y_hat, Y_bar) + SSE(y_test.reshape(pred.shape),Y_hat))
Rsquared

sns.set(style='white', font_scale=1)
plt.figure(figsize =(10,8))
plt.scatter(X_test,y_test,color='green',label="prediction using linear regression")
plt.plot(X_test,lr.predict(X_test),color='red', label="predicted regression line")
plt.xlabel("Living Space (sqft)", fontsize=15)
plt.ylabel("Price ($)", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()
features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
            'view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
            'zipcode','lat','long','sqft_living15','sqft_lot15']
mask = np.zeros_like(df[features].corr(),dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(16,12))
plt.title ("Pearsons correlation matirx",fontsize=25)
sns.heatmap(df[features].corr(), linewidth=0.25, vmax=0.7, square= True, cmap="BuGn",linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
df_dm = df.copy()
df_dm.head()
# just take the year from the date columns
df_dm['sales_yr'] = df_dm['date'].astype(str).str[:4]

#find add of the building when house where sold = sales_year - built_year
df_dm['age'] = df_dm['sales_yr'].astype(int) - df_dm['yr_built']

# find age of the renovation when house were sold
df_dm['age_renov'] = 0
df_dm['age_renov'] = df_dm['sales_yr'][df_dm['yr_renovated']!=0].astype(int)-df_dm['yr_renovated'][df_dm['yr_renovated']!=0]
df_dm['age_renov'][df_dm['age_renov'].isnull()]=0
# partition the 'age' into bins
bins = [-2, 0,5, 10, 25, 50,75, 100, 100000]
labels = ['<1', '1-5','6-10','11-25', '26-50','51-75','76-100','>100']
df_dm['age_binned'] = pd.cut(df_dm['age'], bins=bins, labels=labels)

# partition 'age_renov' in to bins
bins = [-2, 0,5, 10, 25, 50, 75,100,100000]
labels = ['<1', '1-5','6-10','11-25', '26-50','51-75','76-100','>100']
df_dm['age_renov_binned'] = pd.cut(df_dm['age_renov'], bins=bins, labels=labels)

#histogram for binned columns

f, axes = plt.subplots(1,2,figsize = (15,5))
p1 = sns.countplot(df_dm['age_binned'], ax=axes[0])
p2 = sns.countplot(df_dm['age_renov_binned'],ax=axes[1])

axes[0].set(xlabel='Age')
axes[0].yaxis.tick_left()
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='Renovation Age');

# transform the factor values to be able to use in the model
df_dm = pd.get_dummies(df_dm, columns=['age_binned','age_renov_binned'])

train_data_dm, test_data_dm = train_test_split(df_dm, train_size = 0.8, random_state=3)
features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','grade','age_binned_<1', 
            'age_binned_1-5', 'age_binned_6-10','age_binned_11-25', 'age_binned_26-50',
            'age_binned_51-75','age_binned_76-100', 'age_binned_>100','age_renov_binned_<1',
            'age_renov_binned_1-5', 'age_renov_binned_6-10', 'age_renov_binned_11-25',
            'age_renov_binned_26-50', 'age_renov_binned_51-75', 'age_renov_binned_76-100',
            'age_renov_binned_>100','zipcode','lat','long','sqft_living15','sqft_lot15']
complex_model1 = linear_model.LinearRegression()
complex_model1.fit(train_data_dm[features],train_data_dm['price'])
print('Intercept:{}' .format(complex_model1.intercept_))
print('coefficient:{} '.format(complex_model1.coef_))

pred = complex_model1.predict(test_data_dm[features])
Root_mean_square_error = np.sqrt(metrics.mean_squared_error(test_data_dm['price'], pred))
R2_square_train = complex_model1.score(train_data_dm[features], train_data_dm['price']) # R2 square for training model
R2_square_test = complex_model1.score(test_data_dm[features], test_data_dm['price']) # R2 square for test model
print("R2_square for train data:",R2_square_train)
print("R2_square for test data:",R2_square_test)
R2_squareAdjusted_train = adjustedR2(complex_model1.score(train_data_dm[features], train_data_dm['price']),train_data_dm.shape[0],len(features)) # R2 squareAdjusted for training model
print("Rsquare adjusted value for training data:", R2_squareAdjusted_train)
