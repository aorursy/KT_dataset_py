import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
datapath = "../input/jm1.csv"
df=pd.read_csv(datapath,sep=",",encoding = 'latin')
df.head()
df.tail()
df.info()
df.loc[df['uniq_Op'] == "?"]
df = df.drop([143,358,1598,4214,8279])
df.info()
print(df.shape)
df.describe()
ozNitelikler = df.iloc[:, :-7].values
np.corrcoef(ozNitelikler)
p = sns.countplot(x="loc", data = df, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90) 
df['Loc_Pass'] = np.where(df['loc']<10, 'F', 'P')
df.Loc_Pass.value_counts()
p = sns.countplot(x='loc', data = df, hue='Loc_Pass', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) 
x = df.drop(["defects","branchCount","total_Opnd","total_Op","uniq_Opnd","uniq_Op","Loc_Pass"],axis=1)
x.head()
y = df[["defects"]]
y.head()
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
x_train.shape, y_train.shape
x_test.shape, y_test.shape
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(x_train,y_train)
naive_bayes_pred = naive_bayes_model.predict(x_test)
naive_bayes_score = accuracy_score(naive_bayes_pred,y_test)*100
naive_bayes_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
naive_bayes_cv_model = GaussianNB()
naive_bayes_cv_score = cross_val_score(naive_bayes_cv_model,x,y,cv=k_fold,scoring = 'accuracy')*100
naive_bayes_cv_score.mean()
naive_bayes_cv_model.fit(x,y)
naive_bayes_cv_pred = naive_bayes_cv_model.predict(x)
naive_bayes_cv_score = accuracy_score(naive_bayes_cv_pred,y)*100
naive_bayes_cv_score
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 0)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_train)
cm
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()
tree_cv_score = cross_val_score(tree_model,x,y,cv=k_fold,scoring = 'accuracy')*100
tree_cv_score
tree_cv_score.mean()