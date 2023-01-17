# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from sklearn.neighbors import KNeighborsClassifier 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!pip3 install sklearn
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt 
from pylab import rcParams

user_review = pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")
google_apps = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
google_apps.head()
google_apps.shape
total = google_apps.isnull().sum().sort_values(ascending=False)
percent = (google_apps.isnull().sum()/google_apps.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)
google_apps.dropna(how ='any', inplace = True)
total = google_apps.isnull().sum().sort_values(ascending=False)
percent = (google_apps.isnull().sum()/google_apps.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)
print(google_apps.shape)
google_apps.info()
google_apps.isnull().sum()
google_apps.Price = google_apps.Price.apply(lambda x: str(x).replace("$",""))
google_apps.Price.unique()

google_apps.Size = google_apps.Size.apply(lambda x: str(x).replace("M",""))
google_apps.Size.unique()

google_apps['Installs'] = google_apps.Installs.str.replace('+', '')
google_apps['Installs'] = google_apps.Installs.str.replace(',', '')

google_apps.Reviews = pd.to_numeric(google_apps.Reviews, errors='coerce')
google_apps.Price = pd.to_numeric(google_apps.Price, errors='coerce')
google_apps.Rating = pd.to_numeric(google_apps.Rating, errors='coerce')
google_apps.Size = pd.to_numeric(google_apps.Size, errors='coerce')
google_apps['Installs'] = pd.to_numeric(google_apps['Installs'], errors = 'coerce')
google_apps.dtypes

print(google_apps.shape)
google_apps = google_apps.drop_duplicates(subset=['App'], keep = 'first')
print(google_apps.shape)
plt.figure()
fig = sns.countplot(x='Installs', hue='Type', data=google_apps, palette='RdBu')
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show()
plt.figure()
fig = sns.countplot(x='Rating', hue='Type', data=google_apps, palette='RdBu')
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show()
rating=google_apps['Rating']
price=google_apps['Price']

x = np.array(rating).reshape(-1, 1)
y = np.array(price)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
pred = regressor.predict(xtest)
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Rating")
plt.ylabel("Price")
plt.show()
plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Rating")
plt.ylabel("Price")
plt.show()
installs=google_apps['Installs']
price=google_apps['Price']

x = np.array(installs).reshape(-1, 1)
y = np.array(price)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
joblib.dump(regressor, 'Random Forest Installs vs Price.pkl')
regressor.fit(x_train, y_train)
print(regressor.score(x_train, y_train))
print(regressor.score(x_test, y_test))
y_pred = regressor.predict(x_test)
plt.plot(y_test, color = 'red', label = 'Actual')
plt.plot(y_pred, color = 'blue', label = 'Predicted')
plt.xlabel('Number of Installation')
plt.ylabel('Pricing')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
reviews=google_apps['Reviews']
rating=google_apps['Rating']

x = np.array(reviews).reshape(-1, 1)
y = np.array(rating)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
regressor.fit(x_train, y_train)
joblib.dump(regressor, 'Random Forest Reviews vs Rating.pkl')
print(regressor.score(x_train, y_train))
print(regressor.score(x_test, y_test))
y_pred = regressor.predict(x_test)
plt.plot(y_test, color = 'red', label = 'Actual')
plt.plot(y_pred, color = 'blue', label = 'Predicted')
plt.xlabel('Number of Reviews')
plt.ylabel('Rating')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib 
installs=google_apps['Installs']
X = np.array(installs).reshape(-1, 1)
y = google_apps['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=7)
joblib.dump(knn, 'Knn Instalation and Type.pkl') 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test))
plt.plot(y_test, color = 'red', label = 'Actual')
plt.plot(y_pred, color = 'blue', label = 'Predicted')
plt.xlabel('Number of Installation')
plt.ylabel('Type')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


installs=google_apps['Installs']
X = np.array(installs).reshape(-1, 1)
y = google_apps['Reviews']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


reg_all = LinearRegression()

joblib.dump(reg_all, 'linear reggression Installation and Reviews.pkl')
reg_all.fit(X_train, y_train)


y_pred = reg_all.predict(X_test)

print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
plt.plot(y_test, color = 'red', label = 'Actual')
plt.plot(y_pred, color = 'blue', label = 'Predicted')
plt.xlabel('Number of Installation')
plt.ylabel('Reviews')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


for i, k in enumerate(neighbors):
 
    knn = KNeighborsClassifier(n_neighbors=k)
    joblib.dump(knn, 'This page ML Testing and Training Accuracy.pkl')
    knn.fit(X_train, y_train)

    train_accuracy[i] = knn.score(X_train, y_train)


    test_accuracy[i] = knn.score(X_test, y_test)
    
    
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()