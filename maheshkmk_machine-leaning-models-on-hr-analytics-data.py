import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

hr = pd.read_csv('/kaggle/input/hr-analytics/HR_comma_sep.csv')

hr.head()
hr.isnull().sum()
plt.hist(hr.satisfaction_level)
hr_uns = hr[(hr['average_montly_hours'] > 220) & (hr['satisfaction_level'] < 0.5) & (hr.left == 1)]

hr_uns.head()

hr_uns_p = round((len(hr_uns)/len(hr))*100, 2) 

hr_uns_p
hr.groupby(['Department','left']).size().unstack('left').plot(kind='bar')

b=hr[(hr['last_evaluation']>0.7) & (hr['average_montly_hours']>220) & hr['left']==1]

sns.countplot(data=b, x=b.salary, palette= 'autumn', order =['low', 'medium', 'high'])
sns.countplot(data = hr, x = 'salary', hue ='left',palette = 'bwr')
salary = pd.get_dummies(hr.salary, drop_first = True)

hr1 = pd.concat([hr,salary], axis=1)

hr1.head()
hr1 = hr1.drop(['Department','salary'], axis=1)

hr1.head()
x = hr1.drop(['left'], axis=1)

y = hr1.left



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1 )



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

model.score(x_test, y_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
from sklearn.tree import DecisionTreeClassifier

model1=  DecisionTreeClassifier()

model1.fit(x_train, y_train)

y_pred1 = model1.predict(x_test)

model1.score(x_test, y_test)

from sklearn.ensemble import RandomForestClassifier

model2=  RandomForestClassifier(n_estimators=15)

model2.fit(x_train, y_train)

y_pred1 = model2.predict(x_test)

model2.score(x_test, y_test)
from sklearn.svm import SVC

model3=  SVC()

model3.fit(x_train, y_train)

y_pred1 = model3.predict(x_test)

model3.score(x_test, y_test)