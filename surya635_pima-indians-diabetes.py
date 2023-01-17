#load all need labraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
%matplotlib inline
#load dataset
data = pd.read_csv('../input/diabetes.csv')
#shape of data
data.shape
#let's some data
data.head()
#see the type of all attributes
data.info()
#describe
data.describe()
#check the group by values
data['Outcome'].value_counts()
#plot the outcome values
sb.countplot(x='Outcome', data=data)
plt.show()
#Histogram
data.hist(figsize=(12, 8))
plt.show()
#corralation between each column
corr = data.corr()
plt.figure(figsize=(12, 7))
sb.heatmap(corr, annot=True)
#Boxplot of each column
data.plot(kind='box', figsize=(12, 8), subplots=True, layout=(3, 3))
plt.show()
cols = data.columns[:8]
for item in cols:
    plt.figure(figsize=(10, 8))
    plt.title(str(item) + ' With' + ' Outcome')
    sb.violinplot(x=data.Outcome, y=data[item], data=data)
    plt.show()
#pair plot of each attributes
sb.pairplot(data, size=3, hue='Outcome', palette='husl',)
plt.show()
#let's seprate the data 
X = data.iloc[:, :8].values
y = data.iloc[:, 8].values #target variable
#standarize the data of X
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
#Split the into train and test
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=10)
from sklearn import linear_model

#apply algorithm
model = linear_model.LogisticRegression()

#fiting the model
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)

#accuracy
print("Accuracy -- >", model.score(X_test, y_test)*100)

#plot the confusion matrix
sb.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sb.heatmap(cm, annot=True, fmt='g')
plt.show()
from sklearn.ensemble import RandomForestClassifier

#apply algorithm
model = RandomForestClassifier(n_estimators=1000)

#fiting the model
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)

#accuracy
print("Accuracy -- >", model.score(X_test, y_test)*100)

#plot the confusion matrix
sb.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sb.heatmap(cm, annot=True, fmt='g')
plt.show()
from sklearn.svm import SVC

#applying algorithm
model = SVC(gamma=0.01)

#fiting the model
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)

#accuracy
print("Accuracy -- >", model.score(X_test, y_test)*100)

#plot the confusion matrix
sb.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sb.heatmap(cm, annot=True, fmt='g')
plt.show()
from sklearn.neighbors import KNeighborsClassifier

#applying algorithm
model = KNeighborsClassifier(n_neighbors=20)

#fiting the model
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)

#accuracy
print("Accuracy -- >", model.score(X_test, y_test)*100)

#plot the confusion matrix
sb.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sb.heatmap(cm, annot=True, fmt='g')
plt.show()
from sklearn.ensemble import GradientBoostingClassifier

#apply algorithm
model = GradientBoostingClassifier()

#fiting the model
model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)

#accuracy
print("Accuracy -- >", model.score(X_test, y_test)*100)

#plot the confusion matrix
sb.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sb.heatmap(cm, annot=True, fmt='g')
plt.show()
