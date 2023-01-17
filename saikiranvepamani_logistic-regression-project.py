import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
ad_data = pd.read_csv('../input/salaries/advertising.csv')
ad_data.head()
ad_data.describe()
ad_data.info()
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')
sns.jointplot(x='Age',y='Area Income',data=ad_data)
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind="kde")
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data)
sns.pairplot(ad_data,hue='Clicked on Ad')
from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression 
model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,predictions))
print('MSE:',metrics.mean_squared_error(y_test,predictions))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))
model.score(X_test,y_test)