import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
plt.show()
%matplotlib inline
# Import csv file into dataframe
df = pd.read_csv('../input/kc_house_data.csv')
df.head()
df.info()
df.describe()
df.corr()[1:2]
sns.heatmap(df[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'price']].corr(), cmap='coolwarm')
df.columns
import warnings
warnings.filterwarnings("ignore")
sns.jointplot(x='sqft_living',y='price',data=df)
sns.jointplot(x='grade',y='price',data=df)
sns.pairplot(df[['price', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view']])
sns.pairplot(df[['price','sqft_above', 'sqft_basement', 'yr_renovated','lat', 'sqft_living15']])
sns.set_style('whitegrid')
sns.regplot(df.sqft_living, df.price, order=1, ci=None, scatter_kws={'color':'r', 's':9})
plt.xlim(0,13540)
plt.ylim(ymin=0);
#Using all features to train model for Linear Regression 
X = df[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']]
y = df['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
print('Intercept:',lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
from sklearn.metrics import r2_score
print('R2 Score : ',r2_score(y_test, predictions))