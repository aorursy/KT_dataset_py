import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data.head()
print(data.dtypes)
print('\n')
print(data.info())
print('\n')
print(data.describe())
print('\n')
print(len(data.index))
data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()
print(len(data.index))
data.Survived.value_counts().plot(kind='bar')
plt.xlabel('Survived')
plt.show()
female = data.Survived[data.Sex == 'female'].value_counts().sort_index()
female.plot(kind='barh', color='blue', label='Female')
plt.show()
male = data.Survived[data.Sex == 'male'].value_counts().sort_index()
male.plot(kind='barh',label='Male', color='red')
plt.show()
Hclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
Hclass.plot(kind='bar',label='Hclass', color='red', alpha=0.6)
plt.show()
Lclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
Lclass.plot(kind='bar',label='Lclass', color='Blue', alpha=0.6)
plt.show()
y, X = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + Fare + C(Embarked)', data = data, return_type='dataframe')
# X.head()
y = np.ravel(y)
model = LogisticRegression()
model.fit(X, y)
print(model.score(X, y))
print(1 - y.mean()) # compare with all unservived, 0.2 higher accuracy
print(pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_)))))
test['Survived'] = 1
test.loc[np.isnan(test.Age), 'Age'] = np.mean(data['Age'])
ytest, Xtest = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + Fare + C(Embarked)', data = test, return_type='dataframe')
pred = model.predict(Xtest).astype(int)
# solution = pd.DataFrame({"PassengerID": list(range(1,len(pred)+1)),"Label": pred})
solution = pd.DataFrame(list(zip(test['PassengerId'], pred)), columns=['PassengerID', 'Survived'])
solution.to_csv('./my_prediction.csv', index = False)