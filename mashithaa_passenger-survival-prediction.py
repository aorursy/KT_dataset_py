import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
df.head()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='Survived',data=df,hue='Sex')
sns.countplot(x='Survived',hue='Category',data=df,palette='rainbow')
sns.distplot(df['Age'],kde=False,bins=30)
plt.subplots(figsize=(20,5))
sns.countplot(x='Country',data=df,hue='Survived')
plt.figure(figsize=(12, 7))
sns.boxplot(x='Category',y='Age',data=df,palette='winter')
df.head()
df['Category']=df['Category'].map({'P':1,'C':2})
df['Sex']=df['Sex'].map({'M':1,'F':2})
df.head()
df.drop(['Country','Firstname','Lastname'],axis=1,inplace=True)
df.head()
sns.heatmap(df.corr(),annot=True)
df.shape
from sklearn.model_selection import train_test_split
X=df.drop('Survived',axis=1)
y=df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, 
                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))