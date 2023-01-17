import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv("../input/Dataset_spine.csv")
train.info()
train.head()
train.describe()
train.columns
sns.set_style('whitegrid')
sns.countplot(x='Class_att',data=train,palette='RdBu_r')
plt.figure(figsize=(20,10))
c=train.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
from sklearn.model_selection import train_test_split
X= train [['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6',]]
y= train['Class_att']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, 
                                                    random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))