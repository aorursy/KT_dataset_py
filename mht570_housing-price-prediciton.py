import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as lr

from sklearn.model_selection import train_test_split as tts

%matplotlib inline
dt=pd.read_csv('../input/kc_house_data.csv')

dt.head()


dt.info()
plt.figure(figsize=(7, 7))
plt.scatter(x=dt['price'],y=dt['sqft_living'])
plt.xlabel('price')
plt.ylabel('sqft_living')
plt.show()
plt.figure(figsize=(7, 7))
plt.scatter(x=dt['price'],y=dt['grade'])
plt.xlabel('price')
plt.ylabel('grade')
plt.show()
dt.shape
dt.columns
#features are
fet=[ 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
       'lat', 'long']
x_fet=dt[fet]
y_tar=dt['price']


#features v/s target visualition using seaborn pairplot


sns.pairplot(dt,y_vars="price",x_vars=fet,kind = 'reg',palette='spring')

X_train, X_test, Y_train, Y_test = tts(x_fet, y_tar, test_size = 0.3, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#training the model by using LinearRegression

clf=lr()
clf.fit(X_train,Y_train)
accuracy=clf.score(X_test,Y_test)

"Accuracy: {}%".format(int(round(accuracy * 100)))
#using Ridge classifier
from sklearn.linear_model import Ridge as rd

clf1=rd(alpha=0.0001)
clf1.fit(X_train,Y_train)

accuracy1=clf1.score(X_test,Y_test)

"Accuracy1: {}%".format(int(round(accuracy * 100)))
