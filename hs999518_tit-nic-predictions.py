import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

tr=pd.read_csv("../input/titanic/train.csv",encoding='ISO-8859-1')

tr

ts=pd.read_csv("../input/titanic/test.csv",encoding='ISO-8859-1')

ts



ts
data=ts.Ticket.value_counts()

data

import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')

fig,ax=plt.subplots(figsize=(50,20))

 # the width of the bars 





plt.bar(data.index,data.values)



ax.set_xticklabels(data.index,minor=False,size=10,rotation=90)

plt.title("Ticket",size=20)

plt.show()
tr.Age.max()

tr.query("Age=='80.0'")



plt.hist(tr["Age"])

plt.show()
sur=tr["Survived"].value_counts()

sur

plt.bar(sur.index,sur.values)

plt.show()
tr.Sex.max()

#fg.query("Sex=='male'")

sx=tr['Sex'].value_counts()

sx

plt.bar(sx.index,sx.values)

plt.title('Sex')





plt.show()

tr.Ticket.min()



tr.query("Ticket=='110152'")

tk=tr['Ticket'].value_counts()

tk

sns.distplot(tk,rug=True,kde=True)

plt.show()

X_train=tr.loc[:,['Name','Sex','Age','SibSp','Parch','Ticket']]

y_train=tr.iloc[:,1]

X_test=ts.loc[:,['Name','Sex','Age','SibSp','Parch','Ticket']]



X_train

y_train

X_test
X_train
from sklearn.preprocessing import LabelEncoder

la=LabelEncoder()

X_test.iloc[:,0]=la.fit_transform(X_test.iloc[:,0])

X_test

X_test.iloc[:,1]=la.fit_transform(X_test.iloc[:,1])

X_test

X_test.iloc[:,2]=la.fit_transform(X_test.iloc[:,2])

X_test

X_test.iloc[:,5]=la.fit_transform(X_test.iloc[:,5])

X_test



X_train.iloc[:,0]=la.fit_transform(X_train.iloc[:,0])

X_train

X_train.iloc[:,1]=la.fit_transform(X_train.iloc[:,1])

X_train

X_train.iloc[:,2]=la.fit_transform(X_train.iloc[:,2])

X_train

X_train.iloc[:,5]=la.fit_transform(X_train.iloc[:,5])

X_train





from sklearn.preprocessing import Imputer



missingValueImputer = Imputer (missing_values = 'NaN', strategy = 'mean', 

                               axis = 0)  

missingValueImputer = missingValueImputer.fit (X_test.loc[:,['Age']])



X_test.loc[:,['Age']] = missingValueImputer.transform(X_test.loc[:,['Age']])

X_test



missingValueImputer = missingValueImputer.fit (X_train.loc[:,['Age']])



X_train.loc[:,['Age']] = missingValueImputer.transform(X_train.loc[:,['Age']])

X_train
# using MInMaxScaler

from sklearn.preprocessing import MinMaxScaler

mms=MinMaxScaler()

X_trainNorm=mms.fit_transform(X_train)

X_testNorm=mms.transform(X_test)

X_trainNorm

X_testNorm





#using StandardScaler

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_trainNorm2=sc.fit_transform(X_train)

X_trainNorm2

X_testNorm2=sc.transform(X_test)

X_testNorm2



# using PCA

from sklearn.decomposition import PCA

pa=PCA(n_components=2)

X_trainNorm1=pa.fit_transform(X_trainNorm2)

X_trainNorm1

X_testNorm1=pa.transform(X_testNorm2)

X_testNorm1



print(pa.explained_variance_ratio_)
#Using Logistic with Standard Scaler

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

hr=lr.fit(X_trainNorm2,y_train)

pred=lr.predict(X_testNorm2)





#using gaussian 

from sklearn.naive_bayes import GaussianNB

gb=GaussianNB()

hr1=gb.fit(X_trainNorm2,y_train)

pred1=gb.predict(X_testNorm2)







#using knn

from sklearn.neighbors import KNeighborsClassifier

kc=KNeighborsClassifier()

hr2=kc.fit(X_trainNorm2,y_train)



pred2=kc.predict(X_testNorm2)





#using Svm

from sklearn.svm import SVC

sc=SVC()

hr3=sc.fit(X_trainNorm2,y_train)

pred3=sc.predict(X_testNorm2)





#using Random Forest

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=10,random_state=0,max_depth=5,criterion='entropy')

rf.fit(X_trainNorm2,y_train)

pred4=rf.predict(X_testNorm2)





#using DecisionTree

from sklearn.tree import DecisionTreeClassifier

dc = DecisionTreeClassifier(criterion = "entropy", max_depth = 3,random_state = 100)



dc.fit(X_trainNorm2,y_train)

pred5=dc.predict(X_testNorm2)

#using Logistic Regression using PCA

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

ax=lr.fit(X_trainNorm1,y_train)

ax

pred6=lr.predict(X_testNorm1)

pred6

#using GaussianNB

from sklearn.naive_bayes import GaussianNB

gb=GaussianNB()

ax1=gb.fit(X_trainNorm1,y_train)

ax1

pred7=gb.predict(X_testNorm1)

pred7



#using KNN

from sklearn.neighbors import KNeighborsClassifier

kc=KNeighborsClassifier()

ax2=kc.fit(X_trainNorm1,y_train)

ax2

pred8=kc.predict(X_testNorm1)

pred8





#using SVM

from sklearn.svm import SVC

sc=SVC()

ax3=sc.fit(X_trainNorm1,y_train)

ax3

pred9=sc.predict(X_testNorm1)

pred9







#using random forest

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=10,random_state=0,max_depth=5,criterion='entropy')

rf.fit(X_trainNorm1,y_train)

pred10=rf.predict(X_testNorm1)

pred10



#using Decison tree

from sklearn.tree import DecisionTreeClassifier

dc = DecisionTreeClassifier(criterion = "entropy", max_depth = 3,random_state = 100)

dc.fit(X_trainNorm1,y_train)

pred11=dc.predict(X_testNorm1)

pred11
#using Logistic Regression using MinMaxScaler

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

dt=lr.fit(X_trainNorm,y_train)

dt

pre=lr.predict(X_testNorm)

pre





#using GaussianNB

from sklearn.naive_bayes import GaussianNB

gb=GaussianNB()

dt1=gb.fit(X_trainNorm,y_train)

dt1

pre1=gb.predict(X_testNorm)

pre1







#using KNN



from sklearn.neighbors import KNeighborsClassifier

kc=KNeighborsClassifier()

dt2=kc.fit(X_trainNorm,y_train)

dt2

pre2=kc.predict(X_testNorm)

pre2



#using Svm



from sklearn.svm import SVC

sm=SVC()

dt3=sm.fit(X_trainNorm,y_train)

dt3

pre3=sm.predict(X_testNorm)

pre3





#using RandomForest

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=10,random_state=0,max_depth=5,criterion='entropy')

rf.fit(X_trainNorm,y_train)

pre4=rf.predict(X_testNorm)

pre4





#usingDecision tree



from sklearn.tree import DecisionTreeClassifier

dc = DecisionTreeClassifier(criterion = "gini", max_depth = 3,random_state = 100)



dc.fit(X_trainNorm,y_train)

pre5=dc.predict(X_testNorm)

#Using Logistic Regression

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

data=lr.fit(X_train,y_train)

data

pred=lr.predict(X_test)

pred



















#using Gussian naive bayes

from sklearn.naive_bayes import GaussianNB

gb=GaussianNB()

data1=gb.fit(X_train,y_train)

data1

pred1=gb.predict(X_test)

pred1









#using KNN

from sklearn.neighbors import KNeighborsClassifier

kc=KNeighborsClassifier()

data2=kc.fit(X_train,y_train)

data2

pred2=kc.predict(X_test)

pred2







#using SVM

from sklearn.svm import SVC

sc=SVC()

data3=sc.fit(X_train,y_train)

data3

pred3=sc.predict(X_test)

pred3









#using RAndom_forest

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=10,random_state=0,max_depth=5,criterion='entropy')

rf.fit(X_train,y_train)

pred=rf.predict(X_test)

pred







#using  Decision_Tree



from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(criterion = "gini", max_depth = 3,random_state = 100)

decision_tree.fit(X_train, y_train)



predictValues =decision_tree.predict(X_test)



predictValues



estimators=rf.estimators_[5]

labels=['Pclass','Name','Sex','Age','SibSp','Parch']

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display

graph = Source(tree.export_graphviz(estimators, out_file=None, feature_names=labels, filled = True))

display(SVG(graph.pipe(format='svg')))
data_feature_names = ['Pclass','Name','Sex','Age','SibSp','Parch']



from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(decision_tree, out_file=None, feature_names=data_feature_names, filled = True,rounded=True))





display(SVG(graph.pipe(format='svg')))