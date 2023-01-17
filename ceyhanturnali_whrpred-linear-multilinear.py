import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df1=pd.read_csv('../input/world-happiness/2019.csv')
df1.head()
df = df1.rename(columns = {'GDP per capita': 'GDPpercapita', 'Social support': 'Socialsupport','Healthy life expectancy':'Healthylifeexpectancy'}, inplace = False)
cdf = df[['Score','GDPpercapita','Socialsupport','Healthylifeexpectancy']]
cdf.head(9)
plt.figure(figsize=(24,4))
plt.plot(cdf.Score,cdf.GDPpercapita,color="red") 
plt.plot(cdf.Score,cdf.Socialsupport,color="blue")
plt.xlabel("Score")
plt.ylabel("GDP vs Social Support")
plt.show()
plt.figure(figsize=(24,4))
plt.plot(cdf.Score,cdf.GDPpercapita,color="red") 
plt.xlabel("Score")
plt.ylabel("Healty Life Expectancy")
plt.show()
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
plt.scatter(train.GDPpercapita, train.Score,  color='blue')
plt.xlabel("GDP Per Capita")
plt.ylabel("Score")
plt.show()
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['GDPpercapita']])
train_y = np.asanyarray(train[['Score']])
regr.fit (train_x, train_y)

print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
plt.scatter(train.GDPpercapita, train.Score,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("GDP Per Capita")
plt.ylabel("Score")
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['GDPpercapita']])
test_y = np.asanyarray(test[['Score']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['GDPpercapita','Socialsupport','Healthylifeexpectancy']])
y = np.asanyarray(train[['Score']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)
y_hat= regr.predict(test[['GDPpercapita','Socialsupport','Healthylifeexpectancy']])
x = np.asanyarray(test[['GDPpercapita','Socialsupport','Healthylifeexpectancy']])
y = np.asanyarray(test[['Score']])

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - y)))
print("Residual sum of squares: %.2f"% np.mean((y_hat - y) ** 2))

