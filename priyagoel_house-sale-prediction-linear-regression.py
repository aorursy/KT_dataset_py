import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from sklearn import metrics
import scipy.stats as stats
import pylab
plt.rcParams['figure.figsize'] = 10, 7.5
plt.rcParams['axes.grid'] = True
df=pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
df.head()
df.info()
df['date'] = df.date.str.strip('T000000')
df['date'] = pd.to_datetime(df.date , format='%Y%m%d')
df.isnull().sum()
df.describe().T
sns.distplot(df.price)
df['log_price'] = np.log(df.price)
sns.distplot(df.log_price)
corr = df.corr()
corr.style.background_gradient()
plt.subplots(figsize=(17,14))
sns.heatmap(df.corr(),annot=True,linewidths=0.5,fmt="1.1f")
plt.title("Data Correlation",fontsize=50)
plt.show()
# Drop variables based on low correlation
df=df.drop(['id','condition','yr_built','yr_renovated','zipcode','long','date'],axis=1)
df.head()
feature_columns=df.columns.difference(['price','log_price'])
feature_columns
train, test= train_test_split(df,test_size=0.3,random_state=12345)
print('train data :: ',train.shape)
print('test data :: ',test.shape)
lm=smf.ols('log_price ~ bathrooms + bedrooms + floors + grade + lat + sqft_above + sqft_basement + sqft_living + sqft_living15 + sqft_lot + sqft_lot15 + view + waterfront',train).fit()
lm.summary()
train['pred_price'] = np.exp(lm.predict(train))
train['error'] = train['price'] - train['pred_price']
train.head()
test['pred_price'] = np.exp(lm.predict(test))
test['error'] = test['price'] - test['pred_price']
test.head()
# Accuracy metrices
MAPE_train = np.mean(np.abs(train.error) / train.price) * 100
MAPE_test = np.mean(np.abs(test.error) / test.price) * 100
print(MAPE_train)
print(MAPE_test)
lm.resid.hist(bins=10)
lm.resid.mean()
sns.distplot(lm.resid)
sns.distplot(test.error)
sns.jointplot(train.price,train.error)
stats.probplot(train.error,dist='norm',plot=pylab)
pylab.show()