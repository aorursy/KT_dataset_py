import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = pd.read_csv('../input/headbrain.csv')
print('Shape:',data.shape)
data.head()

X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values
def train(X,Y):
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    n = len(X)
    numerator = 0
    denominator = 0
    for i in range(n):
        numerator += (X[i] - mean_X)*(Y[i] - mean_Y)
        denominator += ((X[i] - mean_X) ** 2)
    m = numerator/denominator  #slope = ((x - meanx)*(y=meany))/((x - meanx)**2)
    c = mean_Y - m*mean_X      #intercept formula is derived from the formula y = mx+c => c = y - m*x
    return (m,c)
def predict(test_data,m,c):
    predicted = []
    for i in range(len(test_data)):
        predicted.append(m*test_data[i] + c) #predicting the testset with the slope and intercept from the training set
    return (predicted)
def r2score(actual,predicted):
    #r2 score is a metric to find how good our regression line is fitted to the test data
    #if its value is 1 then all our test data points fits perfectly in our regression line
    mean = np.mean(actual)
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(len(actual)):
        ss_tot += (actual[i]-mean)**2
        ss_res += (actual[i]-predicted[i])**2
        
    r2 = 1 - (ss_res/ss_tot)
    return r2

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
(m,c) = train(X_train,y_train)
predict = predict(X_test,m,c)
r2_scor = r2score(y_test,predict)
print('The slope m is:',m)
print('The intercept c is:',c)
print('The r2 score is:',r2_scor)
plt.plot(X_test,predict,color='r',label='Regression Line')
plt.scatter(X,Y,color='b',label='Data')
plt.xlabel('Head size')
plt.ylabel('brain weight')
plt.legend()
plt.show()
X = X.reshape(len(X),1)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)

clf = LinearRegression()
clf.fit(X_train,y_train)  #sklearn package helps us to finish our task in a single line

sklearn_predict = clf.predict(X_test)
sklearn_r2  = r2_score(y_test, sklearn_predict)
print('Intercept sklearn:',clf.intercept_)
print('Slope sklearn:',clf.coef_)
print("R-squared Sklearn :",sklearn_r2)
plt.plot(X_test,sklearn_predict,color='r',label='Regression Line')
plt.scatter(X,Y,color='b',label='Data')
plt.xlabel('head size')
plt.ylabel('Brain weight')
plt.legend()
plt.show()