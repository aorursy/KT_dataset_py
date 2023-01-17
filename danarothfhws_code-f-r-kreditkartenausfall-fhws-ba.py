import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df=pd.read_excel('TaiwanCreditDefaults.xls',header=1)
df.head()

df.columns
df.info()
df.describe()
df.isnull().sum()
df['default payment next month'].value_counts()

sns.countplot(df['default payment next month'])
sns.countplot(df['SEX'])
sns.countplot(df['default payment next month'],hue=df['SEX'])
sns.countplot(df['default payment next month'],hue=df['MARRIAGE'])

fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(13.7, 10.27)
sns.barplot(df['AGE'],df['LIMIT_BAL'])

sns.distplot(np.log(df['LIMIT_BAL']),bins=100)

sns.distplot(df['LIMIT_BAL'],bins=100)

plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)

X=df.drop('default payment next month',axis=1)
Y=df['default payment next month']

X.corrwith(Y).plot.bar(figsize=(20,10),
                                        title="Correalation with response variable",
                                        fontsize=15,rot=45,grid=True)


X_new=X.drop(['ID','MARRIAGE','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5'],axis=1)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
X_new=sc.fit_transform(X_new)

Y=np.array(Y)
X.shape,Y.shape

x=X[:1000,]
y=Y[:1000,]

from sklearn import model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(X,Y, test_size= 0.2)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score

#parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
clf_tree=DecisionTreeClassifier()
#clf=GridSearchCV(clf_tree,parameters)
clf_tree.fit(x_train,y_train)
y_pred=clf_tree.predict(x_test)

cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm,annot=True,fmt='d')

ac=accuracy_score(y_pred,y_test)
ac

alg1=RandomForestClassifier()
alg1.fit(x_train,y_train)
y_pred1=alg1.predict(x_test)

cm=confusion_matrix(y_pred1,y_test)
sns.heatmap(cm,annot=True,fmt='d')

ac=accuracy_score(y_pred,y_test)
ac

alg1=RandomForestClassifier()
alg1.fit(x_train,y_train)
y_pred1=alg1.predict(x_test)
cm=confusion_matrix(y_pred1,y_test)
sns.heatmap(cm,annot=True,fmt='d')

ac=accuracy_score(y_pred1,y_test)
ac



