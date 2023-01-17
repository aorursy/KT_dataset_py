import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline
df = pd.read_csv('../input/diabetes.csv')
df.head()
df.info()
df.describe()
sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Outcome',hue='Outcome',data=df,palette='RdBu_r')
sns.pairplot(data=df,hue='Outcome')
plt.scatter(x='Outcome',y='Age',data=df)
plt.xlabel('Outcome')
plt.ylabel('Age')
sns.distplot(df['Age'],kde=False,color='darkblue',bins=20)
df['Age'].hist(bins=40,color='Green',alpha=0.6)
sns.distplot(df['BloodPressure'],kde=False,color='darkblue',bins=20)
sns.jointplot(x='Age',y='BloodPressure',data=df)
import cufflinks as cf 
cf.go_offline()
df['BMI'].iplot(kind='hist',bins=40,color='red')
from sklearn.model_selection import train_test_split
df.head()
X = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
y = ['Output']
df2 = pd.DataFrame(data=df)
df2.head()
X_train, X_test, y_train, y_test = train_test_split(df.drop('Outcome',axis=1),df['Outcome'],
                                                    test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))