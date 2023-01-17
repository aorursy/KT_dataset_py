import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
ad_data = pd.read_csv('../input/advertising.csv')
ad_data.head()
ad_data.info()
ad_data['Age'].plot.hist(bins=30)
sns.jointplot(x='Age',y='Area Income',data=ad_data)
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='red')
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,kind='kde',color='green')
sns.pairplot(ad_data,hue='Clicked on Ad')
from sklearn.cross_validation import train_test_split
X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


from sklearn.metrics import confusion_matrix


print(confusion_matrix(y_test,predictions))


