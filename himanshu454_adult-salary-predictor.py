import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix , accuracy_score
data = pd.read_csv("../input/adult-income-dataset/adult.csv")
data.head(10)
data.info()
data.describe()
data.isnull().sum()
sns.heatmap(data.isnull() , yticklabels=False)
data.isin(['?']).sum(axis=0)
data['native-country'] = data['native-country'].replace('?',np.nan)
data['workclass'] = data['workclass'].replace('?',np.nan)
data['occupation'] = data['occupation'].replace('?',np.nan)
data.isin(['?']).sum(axis=0)
data.isnull().sum()
data.dropna(how='any',inplace=True)
plt.figure(figsize = (16,9))
sns.countplot(data["workclass"])
plt.figure(figsize = (16,9))
sns.countplot(data["occupation"])
data["native-country"].value_counts()
data.columns
data.drop(['educational-num','age', 'hours-per-week', 'fnlwgt', 'capital-gain','capital-loss', 'native-country'], axis=1, inplace=True)
data.head(10)
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
l = LabelEncoder()
data["gender"] = l.fit_transform(data["gender"])
data["race"] = l.fit_transform(data["race"])
data["race"] = l.fit_transform(data["race"])
data["marital-status"] = l.fit_transform(data["marital-status"])
data["workclass"] = l.fit_transform(data["workclass"])
data["education"] = l.fit_transform(data["education"])
data["occupation"] = l.fit_transform(data["occupation"])
data["relationship"] = l.fit_transform(data["relationship"])


data.head(10)
sns.distplot(data["workclass"] )
sns.distplot(data["occupation"])
plt.figure(figsize  = (16,9))
sns.heatmap(data.corr(), annot = True)
plt.figure(figsize = (16,9))
sns.pairplot(data)
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
train_x , test_x, train_y , test_y = train_test_split(x,y,test_size = 0.2 , random_state = 42)
train_x
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)
train_x
model = KNeighborsClassifier(n_neighbors=12)
model.fit(train_x , train_y)
pred = model.predict(test_x)
accuracy_score(test_y, pred)
sns.heatmap(confusion_matrix(test_y , pred) , annot =True)

## select best K value
accuracy =[]
error_rate = []
for i in range(1,20):
    score = cross_val_score(KNeighborsClassifier(n_neighbors=i) ,x ,y, cv=5)
    accuracy.append(score.mean())
    error_rate.append(1-score.mean())
 
plt.scatter(range(1,20) , error_rate)
plt.plot(range(1,20) , error_rate)
plt.xlabel("K Value")
plt.ylabel("Error")
plt.show()


 

plt.scatter(range(1,20) , accuracy)
plt.plot(range(1,20) , accuracy)
plt.xlabel("K Value")
plt.ylabel("accuracy")



plt.show()
z = np.array([[2,1,4,6,3,2,1]])
print(model.predict(z))