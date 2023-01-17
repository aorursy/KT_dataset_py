import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("../input/USA_Housing.csv")
data.head()
data.describe()
data.shape
data.info()
data.dropna(how="any",axis=0)
data.shape
data.columns
import seaborn as sns
sns.pairplot(data)

plt.show()
sns.distplot(data['Price'])
plt.show()
x=data["Avg. Area Income"]
y=data["Avg. Area House Age"]
x.head()
y.head()
sns.regplot(x="Avg. Area Income",y="Price",data=data)
plt.show()
data.head()
sns.regplot(x="Avg. Area House Age",y="Price",data=data)
plt.show()
sns.regplot(x="Avg. Area Number of Rooms",y="Price",data=data)
plt.show()
sns.regplot(x="Avg. Area Number of Bedrooms",y="Price",data=data)
plt.show()
sns.regplot(x="Area Population",y="Price",data=data)
plt.show()
sns.regplot(x="Price",y="Price",data=data)
plt.show()
sns.jointplot(x="Avg. Area House Age",y="Price",kind="reg",data=data)
plt.show()
sns.jointplot(x="Avg. Area Number of Rooms",y="Price",kind="reg",data=data)
plt.show()
data.corr()
corr=data.corr()
corr.nlargest(7,'Price')["Price"]
fig = plt.figure(figsize = (10,7))
sns.heatmap(data.corr(), annot = True,cmap = "coolwarm")
plt.show()

X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Area Population']]
y=data[['Price']]

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size = 0.4, random_state=101)

from sklearn.linear_model import LinearRegression
sv=LinearRegression()
sv.fit(X_train,Y_train)
sv.predict(X_train)
sv.score(X_train,Y_train)
test_score=sv.score(X_test,Y_test)
test_score
sv.coef_
print(sv.intercept_)

coef = pd.DataFrame(sv.coef_,X.columns, columns = ['coeffcient'])
coef

predict = sv.predict(X_test)
predict

