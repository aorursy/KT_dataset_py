import numpy as np

import pandas as pd

from pandas import Series, DataFrame



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sb

from pylab import rcParams



import scipy

from scipy import stats

from scipy.stats import spearmanr

csv1= 'C:/Users/arul.arockia/Google Drive/1_Python_dump/train.csv'

train=pd.read_csv(csv1)

train.head()



csv2= 'C:/Users/arul.arockia/Google Drive/1_Python_dump/test.csv'

test=pd.read_csv(csv2)

test.head()
#feature engineering



d={'male':1, 'female':0}

train.Sex=train.Sex.map(d)

#take a clean version without nans



train=train[-train.Pclass.isnull()]

train=train[-train.Sex.isnull()]

train=train[-train.Age.isnull()]

train=train[-train.SibSp.isnull()]

train=train[-train.Parch.isnull()]

train=train[-train.Fare.isnull()]
X= train.iloc[:, [2,4,5,6,7,9]].values

y= train.iloc[:, 1].values
column_names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 0.25, random_state=0)
# Feature scaling

from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.transform(X_test)
# Fitting Logistic Regression to the training set

from sklearn.linear_model import LogisticRegression

classifier= LogisticRegression(random_state = 0)

classifier.fit(X_train,y_train)
# Predicting the Test set results

y_pred=classifier.predict(X_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, y_pred)
cm
coef=classifier.coef_
#joining coeficient data to column names



Coefficients=pd.DataFrame(np.transpose(coef),columns=['coefficients'])

Columnnames=pd.DataFrame(np.transpose(column_names),columns=['column_names'])

combined_table= pd.concat([Coefficients, Columnnames], axis=1)

combined_table=combined_table.sort_values(by='coefficients')
plt.figure(figsize=(15,6))

plt.bar(np.arange(len(combined_table)),combined_table.coefficients)

plt.xticks(np.arange(len(combined_table)),combined_table.column_names, rotation=45)

plt.ylabel('coefficients')

plt.title('Regression coefficients')

plt.show()
combined_table
cm_x= ['True Positive', 'True Negative']

cm_y= ['False Postive', 'False Negative']