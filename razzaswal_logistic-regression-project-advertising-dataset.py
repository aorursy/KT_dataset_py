import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
ad_data = pd.read_csv('../input/advertising-dataset/advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()
sns.set()
sns.distplot(ad_data['Age'].dropna(),kde=False, color='navy', bins=20)

sns.jointplot(x='Age', y='Area Income', data=ad_data)

sns.jointplot(x = 'Age', y='Daily Time Spent on Site', kind='kde',space=0, color='b', data=ad_data)

sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',color='r',data=ad_data)

sns.pairplot(ad_data, hue="Clicked on Ad")
type(ad_data['Timestamp'][0])
ad_data['Timestamp'] = pd.to_datetime(ad_data['Timestamp'])
time = ad_data['Timestamp'].iloc[0]
ad_data['Hour'] = ad_data['Timestamp'].apply(lambda x: x.hour)
ad_data['Month'] = ad_data['Timestamp'].apply(lambda x: x.month)
ad_data['Day of Week'] = ad_data['Timestamp'].apply(lambda x: x.dayofweek)
ad_data['Year'] = ad_data['Timestamp'].apply(lambda x: x.year)
ad_data.head()
plt.figure(figsize=(10,10))
print(ad_data.groupby('Month')['Month'].count())
ad_data.groupby('Month')['Month'].count().plot(kind='bar')
plt.title('click count by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(10,10))
print(ad_data.groupby('Hour')['Hour'].count())
ad_data.groupby('Hour')['Hour'].count().plot(kind='bar')
plt.title('click count by Hour')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.show()

sns.heatmap(ad_data.corr(),annot=True)
ad_data.head(2)
from sklearn.model_selection import train_test_split
x = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male','Hour','Month','Day of Week','Year']]
y= ad_data['Clicked on Ad']
x, x_test, y, y_test = train_test_split(x,y,test_size=0.30, random_state=40)
from sklearn.linear_model import LogisticRegression
Logmodel = LogisticRegression()
Logmodel.fit(x,y)
predictions = Logmodel.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(predictions, y_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)