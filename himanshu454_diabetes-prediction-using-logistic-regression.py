import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score
data = pd.read_csv("../input/diabetes-dataset/diabetes2.csv")
data.head(10)
data.info()
data.describe()
sns.heatmap(data.isnull() , yticklabels=False)
sns.countplot(x='Outcome',data=data)
sns.distplot(data['Age'].dropna(),kde=True)
data.corr()
plt.figure(figsize = (16,9))
sns.heatmap(data.corr()  ,annot = True)
sns.pairplot(data)
plt.figure(figsize = (16,9))
sns.boxplot(x='Age', y='BMI', data=data)
x = data.drop('Outcome',axis=1)
y = data['Outcome']
train_x , test_x , train_y , test_y = train_test_split(x,y,test_size = 0.3 , random_state = 42)
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)
model = LogisticRegression()
model.fit(train_x ,train_y)
pred = model.predict(test_x)
print(confusion_matrix(test_y , pred))
print(classification_report(test_y , pred))
sns.heatmap(confusion_matrix(test_y , pred) , annot = True)
print(accuracy_score(test_y , pred))