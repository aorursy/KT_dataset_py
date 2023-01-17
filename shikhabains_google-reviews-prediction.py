#import files

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

a=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

#read files

b=pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')

#merge files

c=pd.merge(a,b,on='App',how='inner')

a["Rating"].nunique()
c['Rating']=c['Rating'].fillna(3.5)

c['Sentiment']=c['Sentiment'].fillna('Positive')



c["Installs"]=c["Installs"].str.replace("+","")

c["Installs"]=c["Installs"].str.replace(",","").astype(int)

d=c.groupby("App")["Installs"].sum().sort_values(ascending=False)

d=c['Rating'].value_counts().sort_values(ascending=False)

d1=c.groupby("Reviews")["App"].count().sort_values(ascending=False)

d3=c.groupby("Rating")["App"].count().sort_values(ascending=False).head()



d4=c.groupby("Last Updated")["App"].count().sort_values(ascending=False)

d5=c.groupby("Sentiment")["App"].count().sort_values(ascending=False)

d5







plt.bar(d5.index,d5.values)



plt.title('People views about play store')



#plt.legend(loc='upper right')

plt.legend(loc=[1,0.9])



plt.show()

import matplotlib.pyplot as plt

plt.pie(d3.values,labels=d3.index,autopct='%.1f%%',startangle=90,shadow=True)



plt.title('People views about play store')



#plt.legend(loc='upper right')

plt.legend(loc=[1,0.9])



plt.show()

plt.bar(d3.index,d3.values)



plt.title('People views about play store')

plt.show
c["Current Ver"]=c["Current Ver"].str.replace("Varies with devic","")
c["Current Ver"]=c["Current Ver"].str.replace("e","")
x=c.iloc[:,[0,1,3,5,6,8,14]]

y=c.loc[:,['Rating']]

x.isnull().sum()

from sklearn.preprocessing import LabelEncoder

la=LabelEncoder()

y['Rating']=la.fit_transform (y['Rating'])
la=LabelEncoder()

x['App']=la.fit_transform (x['App'])

x['Category']=la.fit_transform (x['Category'])

x['Type']=la.fit_transform (x['Type'])

x['Sentiment']=la.fit_transform (x['Sentiment'])

x['Content Rating']=la.fit_transform (x['Content Rating'])

x

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.20,random_state=0)







from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)



knn.fit(train_x,train_y)





y_pred = knn.predict(test_x)

print(y_pred)



from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(test_y, y_pred))



y_pred

from sklearn import datasets

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()

gnb.fit(train_x, train_y)



prediction = gnb.predict(test_x)



from sklearn.metrics import accuracy_score

print(accuracy_score(test_y, prediction))

prediction



#print(metrics.confusion_matrix(test_y, prediction))

#prediction





from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion = "gini", max_depth = 3,random_state = 80)

dt.fit(train_x, train_y)

predictvalues=dt.predict(test_x)

predictvalues



from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(test_y, predictvalues))



predictvalues









data = [0,1,3,5,6,8,14]



from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(dt, out_file=None, feature_names=data, filled = True,rounded=True))





display(SVG(graph.pipe(format='svg')))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=10, random_state=80, max_depth=5, criterion = 'gini')

rfc.fit(train_x, train_y)

predictvalues=rfc.predict(test_x)

predictvalues



from sklearn.metrics import accuracy_score

print(accuracy_score(test_y, predictvalues))

predictvalues
estimators=rfc.estimators_[2]

labels= [ 0,1,3,5,6,8,14]

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(estimators, out_file=None

   , feature_names=labels

   , filled = True))

display(SVG(graph.pipe(format='svg')))

from sklearn.preprocessing import MinMaxScaler       

mms = MinMaxScaler()

train_x.np = mms.fit_transform (train_x) #fit and transform

test_x.np = mms.transform (test_x) # only transform





print(train_x.np)

print(test_x.np)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)



knn.fit(train_x.np,train_y)





y_pred = knn.predict(test_x.np)

print(y_pred)



from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(test_y, y_pred))



y_pred





from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()

gnb.fit(train_x.np, train_y)



prediction = gnb.predict(test_x.np)



from sklearn.metrics import accuracy_score

print(accuracy_score(test_y, prediction))

prediction

dt=DecisionTreeClassifier(criterion = "gini", max_depth = 3,random_state = 80)

dt.fit(train_x.np, train_y)

predictvalues=dt.predict(test_x.np)

predictvalues



from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(test_y, predictvalues))



predictvalues

rfc=RandomForestClassifier(n_estimators=10, random_state=80, max_depth=5, criterion = 'gini')

rfc.fit(train_x.np, train_y)

predictvalues=rfc.predict(test_x.np)

predictvalues



from sklearn.metrics import accuracy_score

print(accuracy_score(test_y, predictvalues))

predictvalues
from sklearn.preprocessing import StandardScaler      

mms = StandardScaler()

train_x.np1 = mms.fit_transform (train_x) 

test_x.np1 = mms.transform (test_x) 





print(train_x.np1)



print(test_x.np1)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)



knn.fit(train_x.np1,train_y)





y_pred = knn.predict(test_x.np1)

print(y_pred)



from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(test_y, y_pred))



y_pred





from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()

gnb.fit(train_x.np1, train_y)



pred= gnb.predict(test_x.np1)



from sklearn.metrics import accuracy_score

print(accuracy_score(test_y, pred))

pred
dt=DecisionTreeClassifier(criterion = "gini", max_depth = 3,random_state = 80)

dt.fit(train_x.np1, train_y)

predictvalues=dt.predict(test_x.np1)

predictvalues



from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(test_y, predictvalues))



predictvalues

rfc=RandomForestClassifier(n_estimators=10, random_state=80, max_depth=5, criterion = 'gini')

rfc.fit(train_x.np1, train_y)

predictvalues=rfc.predict(test_x.np1)

predictvalues



from sklearn.metrics import accuracy_score

print(accuracy_score(test_y, predictvalues))

predictvalues