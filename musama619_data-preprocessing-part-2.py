import pandas as pd
data = pd.read_csv('../input/dataset/Data.csv')
data
mean = data['Salary'].mean()
data['Salary'] = data['Salary'].fillna(mean)
median = data['Age'].median()
data['Age'] = data['Age'].fillna(median)
data
data['Salary'] = data['Salary'].astype(int)
data['Age'] = data['Age'].astype(int)
data
x_data = data.iloc[:, :-1]

y_data = data['Purchased']
from sklearn.preprocessing import OneHotEncoder
ohc = OneHotEncoder()
c = pd.DataFrame(ohc.fit_transform(data[['Country']]).toarray())
x_data = pd.concat([c, x_data], axis=1)
x_data
x_data = x_data.drop(['Country'], axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y_data)
y
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x_data, y, test_size=0.2, random_state=1)
xtrain
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
xtrain.iloc[:, 3:] = sc.fit_transform(xtrain.iloc[:, 3:])
xtrain
xtest.iloc[:, 3:] = sc.transform(xtest.iloc[:, 3:])