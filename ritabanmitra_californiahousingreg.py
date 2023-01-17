import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
from pandas.plotting import scatter_matrix
cali = pd.read_csv('housing.csv')
cali.head()
cali.describe().T
sns.set()
cali.isna().sum().sort_values(ascending = True).plot(kind ='barh',figsize = (10,7))
#Dropping the missing values in our dataset
missing_data = cali.dropna(inplace=True)
cali.isna().sum()
#checking if the data has been cleaned
total_null = cali.isna().sum().sort_values(ascending=False)
percent = (cali.isna().sum()/cali.isna().count()).sort_values(ascending=False)
missing_data = pd.concat([total_null, percent], axis=1, keys=['Total', 'Percent'])
missing_data
cali.drop(['ocean_proximity'],axis = 1).head()
#Plotting histograms
cali.hist(bins = 50, figsize = (20,15))
plt.show();
#Plotting a Scatterplot
plt.figure(figsize = (10,8))
plt.scatter(cali.latitude,cali.longitude,alpha = 0.2,c = cali.median_house_value, s = cali.population/100)
plt.colorbar()
sns.pairplot(cali[["total_bedrooms","population","median_income","median_house_value"]],diag_kind = "kde")
y = cali['median_house_value']
x = cali[['median_income','total_rooms','housing_median_age']]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
#X = train['median_income'].values.reshape(-1,1)
#y = train['median_house_value']
train = x_train.join(y_train)
corr_mat = train.corr()
corr_mat['median_house_value'].sort_values(ascending = False)
fig = plt.subplots(figsize = (20,10))
sns.heatmap(train.corr(), annot = True)
linear_regression = LinearRegression()
linear_regression.fit(x,y)
y_pred = linear_regression.predict(x)
y_pred
acc = linear_regression.score(x_test, y_test)
acc_percentage = acc*100
acc_percentage
#measuring accuracy using mae and mse
mse = mean_squared_error(y_pred,y)
np.sqrt(mse)
mae = mean_absolute_error(y_pred,y)
np.sqrt(mae)
#in a linear regression y = mx + c ( m is the coefficient for generating the model) )
print('Coefficients: \n', linear_regression.coef_)
