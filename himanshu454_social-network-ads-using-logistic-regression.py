import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix
data = pd.read_csv("../input/logistic-regression/Social_Network_Ads.csv")
data.head(10)
data.info()
data.describe()
l = LabelEncoder()
data["Gender"] = l.fit_transform(data["Gender"])
data.head(5)
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,cmap='inferno')
data.drop(labels = ['User ID','Gender'], axis = 1, inplace = True)
data.info()
sum(data.duplicated())
data.drop_duplicates(keep = False, inplace = True)
sum(data.duplicated())
plt.figure(figsize=(20, 12))

plt.subplot(3,3,1)
sns.boxplot(data['Age'],color='yellow')
plt.subplot(3,3,2)
sns.boxplot(data['EstimatedSalary'], color='yellow')

plt.show()
plt.figure(figsize=(6, 4))
sns.countplot('Purchased', data=data)
plt.title('Class Distributions')
plt.show()
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
model = LogisticRegression()
model.fit(train_x , train_y)
pred = model.predict(test_x)
print(confusion_matrix(test_y , pred))
print(accuracy_score(test_y , pred))
sns.heatmap(confusion_matrix(test_y , pred) , annot = True)