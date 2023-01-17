import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv("../input/advertising/advertising.csv")
data.head()
data.info()
data.isnull()
sns.heatmap(data.isnull(),yticklabels=False,cbar=True,cmap='Accent')
data.describe()
plt.figure(figsize=(16,4))
sns.countplot(x='Age', data=data)
sns.jointplot(data['Area Income'], data['Age'])
sns.jointplot(data['Age'], data['Daily Time Spent on Site'])
sns.jointplot(data['Daily Time Spent on Site'], data['Daily Internet Usage'])
sns.pairplot(data, hue='Clicked on Ad')
data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1, inplace=True)
X = data.drop(['Clicked on Ad'], axis = 1)              
y = data['Clicked on Ad']                               
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
predictions

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions)
accuracy