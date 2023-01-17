import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
data = pd.read_csv('/kaggle/input/suv-data/suv_data.csv')
data.head()
data.tail()
df = pd.DataFrame(data.info())
df
sns.countplot(x='Gender',data=data)
sns.countplot(x='Gender',data=data[data['Purchased']==1])
plt.figure(figsize = (5,5))
sns.distplot(data['Age'],color='maroon')
plt.figure(figsize = (5,5))
sns.distplot(data[data['Purchased']==1]['Age'],color='maroon')
sns.pairplot(data=data,hue='Gender')
plt.figure(figsize = (15,7))
sns.barplot(x=data['Age'],y=data['Purchased'])
plt.figure(figsize = (15,7))
sns.lineplot(x=data['EstimatedSalary'],y=data['Purchased'])
data['EstimatedSalary'].max()
data['EstimatedSalary'].min()
data[data['Purchased'] ==1 ]['EstimatedSalary'].max()
data[data['Purchased'] ==1 ]['EstimatedSalary'].min()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Gender"] = le.fit_transform(data['Gender'])
data.head()
x = data.drop(['Purchased'],axis=1).values
y = data['Purchased'].values
x_train ,x_test,y_train,y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
lr =LogisticRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)
sns.heatmap(cm,annot=True)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)