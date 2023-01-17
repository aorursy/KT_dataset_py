#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
%matplotlib inline
#load dataset
data = pd.read_csv('../input/adult.csv')
#view some data
data.head()
data.shape
#information about data type
data.info()
#Count the values of target variables
print(data.income.value_counts())
sb.countplot(x='income', data=data)
plt.show()
#replace missing variable('?') into null variable using numpy
data = data.replace('?', np.NaN)
#let's count the how many variable missing
data.isnull().sum()
#plotting of Null variable
plt.figure(figsize=(10,6))
sb.heatmap(data.isnull())
plt.show()
#let's fill null variable 
var = data['native-country'].mode()
data['native-country'] = data['native-country'].replace(np.NaN,var[0])

var1 = data.workclass.mode()[0]
data.workclass = data.workclass.replace(np.NaN, var1)

var2 = data.occupation.mode()[0]
data.occupation = data.occupation.replace(np.NaN,var2)
#again check there is null value or not
print(list(data.isnull().sum()))
plt.figure(figsize=(10,6))
sb.heatmap(data.isnull())
plt.show()
#convert string into integer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = ['workclass', 'education','marital-status', 'occupation',
       'relationship', 'race', 'gender','native-country','income']
for col in cols:
    data[col] = le.fit_transform(data[col])
data.head()
data.describe()
#Correlation between attributes
corr = data.corr()
plt.figure(figsize=(20, 10))
sb.heatmap(corr, annot=True)
plt.show()
#Violin plot
plt.style.use('ggplot')
cols = ['age', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'gender']
result = {0:'<=50k', 1:'>50'}
print(result)
for col in cols:
    plt.figure(figsize=(12, 5))
    plt.title(str(col) +' with' + ' income')
    sb.violinplot(x=data.income, y=data[col], data=data)
    plt.show()
X = data.iloc[:,:14]
X = X.values
y = data['income'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
algo = {'LR': LogisticRegression(), 
        'DT':DecisionTreeClassifier(), 
        'RFC':RandomForestClassifier(n_estimators=100), 
        'SVM':SVC(gamma=0.01),
        'KNN':KNeighborsClassifier(n_neighbors=10)
       }

for k, v in algo.items():
    model = v
    model.fit(X_train, y_train)
    print('Acurracy of ' + k + ' is {0:.2f}'.format(model.score(X_test, y_test)*100))