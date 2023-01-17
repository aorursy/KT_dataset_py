import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

crises=pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')

crises
crises.info()
x=crises.drop(['banking_crisis','cc3','gdp_weighted_default'],axis=1)

x
y=crises.iloc[:,13]

y
sns.countplot(x=y,data=crises,palette='hls')

plt.show()
pd.crosstab(x.country,y).plot(kind='bar')

plt.title('crisis frequency for african country')

plt.xlabel('country')

plt.ylabel('frequency of banking crisis')

plt.show()
pd.crosstab(x.year,y).plot(kind='bar')

plt.title('crisis frequency per year')

plt.xlabel('year')

plt.ylabel('frequency of banking crises')

plt.show()
pd.crosstab(x.year,y).plot(kind='bar')

plt.title('crisis frequency for systemic_crisis')

plt.xlabel('systemic crisis')

plt.ylabel('frequency of banking crises')

plt.show()
pd.crosstab(x.year,y).plot(kind='bar')

plt.title('crisis frequency for exch_usd')

plt.xlabel('exch_usd')

plt.ylabel('frequency of banking crises')

plt.show()
pd.crosstab(x.year,y).plot(kind='bar')

plt.title('crisis frequency for inflation_annual_cpi')

plt.xlabel('inflation_annual_cpi')

plt.ylabel('frequency of banking crises')

plt.show()
pd.crosstab(x.year,y).plot(kind='bar')

plt.title('crisis frequency for independence')

plt.xlabel('independence')

plt.ylabel('frequency of banking crises')

plt.show()
pd.crosstab(x.year,y).plot(kind='bar')

plt.title('crisis frequency for inflation_crises')

plt.xlabel('inflation_crises')

plt.ylabel('frequency of banking crises')

plt.show()
pd.crosstab(x.year,y).plot(kind='bar')

plt.title('crisis frequency for currency_crises')

plt.xlabel('currency_crises')

plt.ylabel('frequency of banking crises')

plt.show()
pd.crosstab(x.year,y).plot(kind='bar')

plt.title('crisis frequency for domestic_debt_in_default')

plt.xlabel('domestic_debt_in_default')

plt.ylabel('frequency of banking crises')

plt.show()
pd.crosstab(x.year,y).plot(kind='bar')

plt.title('crisis frequency for sovereign_external_debt_default')

plt.xlabel('sovereign_external_debt_default')

plt.ylabel('frequency of banking crises')

plt.show()
x.exch_usd.hist()
x.country.hist()
#data=x.drop(['country'],axis=1)

#data
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

x['country']=le.fit_transform(x['country'])

y=le.fit_transform(y)

    
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=0)

y_test
from sklearn.preprocessing import MinMaxScaler

mms=MinMaxScaler()

x_trainmin=mms.fit_transform(x_train)

                         

x_testmin=mms.transform(x_test)
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

x_trainstd=ss.fit_transform(x_train)

x_teststd=ss.transform(x_test)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

pred1=lr.predict(x_test)

pred1



from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_test,pred1))
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_trainmin,y_train)

pred2=lr.predict(x_testmin)

pred2



from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_test,pred2))
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_trainstd,y_train)

pred3=lr.predict(x_teststd)

pred3



from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_test,pred3))
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

knn.fit(x_train,y_train)

pred1=knn.predict(x_test)

pred1



from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_test,pred1))
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

knn.fit(x_trainmin,y_train)

pred2=knn.predict(x_testmin)

pred2



from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_test,pred2))
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

knn.fit(x_trainstd,y_train)

pred3=knn.predict(x_teststd)

pred3



#from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_test,pred3))
from sklearn.svm import SVC

svc=SVC()

svc.fit(x_train,y_train)

pred1=svc.predict(x_test)

pred1



#from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_test,pred1))
from sklearn.svm import SVC

svc=SVC()

svc.fit(x_trainmin,y_train)

pred2=svc.predict(x_testmin)

pred2



#from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_test,pred2))
from sklearn.svm import SVC

svc=SVC()

svc.fit(x_trainstd,y_train)

pred3=svc.predict(x_teststd)

pred3



#from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_test,pred3))
from sklearn.tree import DecisionTreeClassifier

dt1=DecisionTreeClassifier(splitter='best',max_depth=3,random_state=56)

dt1.fit(x_train,y_train)

pred1=dt1.predict(x_test)

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred1)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test,pred1))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, pred1))

fet=['case','country','year','systemic_crisis','exch_usd','domestic_debt_in_default','sovereign_external_debt_default','inflation_annual_cpi','independence','currency_crises','inflation_crises']

#distance	speed	temp_inside	gas_type	AC	rain	sun

from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(dt1, out_file=None, feature_names=fet, filled = True,rounded=True))





display(SVG(graph.pipe(format='svg')))

plt.scatter(y_test,pred1)

plt.xlabel('Values')

plt.ylabel('pred1')

plt.show()
from sklearn.tree import DecisionTreeClassifier

dt2=DecisionTreeClassifier(splitter='best',max_depth=3,random_state=56)

dt2.fit(x_trainmin,y_train)

pred2=dt2.predict(x_testmin)

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred2)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test,pred2))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, pred2))

fet=['case','country','year','systemic_crisis','exch_usd','domestic_debt_in_default','sovereign_external_debt_default','inflation_annual_cpi','independence','currency_crises','inflation_crises']



from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(dt2, out_file=None, feature_names=fet, filled = True,rounded=True))





display(SVG(graph.pipe(format='svg')))

plt.scatter(y_test,pred2)

plt.xlabel('Values')

plt.ylabel('pred2')

plt.show()
from sklearn.tree import DecisionTreeClassifier

dt3=DecisionTreeClassifier(splitter='best',max_depth=3,random_state=56)

dt3.fit(x_trainstd,y_train)

pred3=dt2.predict(x_teststd)

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred3)))

print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test,pred3))

print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, pred3))

fet=['case','country','year','systemic_crisis','exch_usd','domestic_debt_in_default','sovereign_external_debt_default','inflation_annual_cpi','independence','currency_crises','inflation_crises']



from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(dt3, out_file=None, feature_names=fet, filled = True,rounded=True))





display(SVG(graph.pipe(format='svg')))

plt.scatter(y_test,pred3)

plt.xlabel('Values')

plt.ylabel('pred3')

plt.show()
from sklearn.ensemble import RandomForestClassifier

model1 = RandomForestClassifier(n_estimators=5, random_state=50,criterion='entropy',max_depth=4)



model1.fit(x_train, y_train)

pred1=model1.predict(x_test)

pred1
estimators=model1.estimators_[3]

labels=['case','country','year','systemic_crisis','exch_usd','domestic_debt_in_default','sovereign_external_debt_default','inflation_annual_cpi','independence','currency_crises','inflation_crises']

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(estimators, out_file=None

   , feature_names=labels

   , filled = True))

display(SVG(graph.pipe(format='svg')))

sns.distplot((y_test-pred1),bins=50)

plt.show()
from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(n_estimators=5, random_state=50,criterion='entropy',max_depth=4)



model2.fit(x_trainmin, y_train)

pred2=model1.predict(x_testmin)

pred2
estimators=model2.estimators_[3]

labels=['case','country','year','systemic_crisis','exch_usd','domestic_debt_in_default','sovereign_external_debt_default','inflation_annual_cpi','independence','currency_crises','inflation_crises']

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(estimators, out_file=None

   , feature_names=labels

   , filled = True))

display(SVG(graph.pipe(format='svg')))

sns.distplot((y_test-pred2),bins=50)

plt.show()
from sklearn.ensemble import RandomForestClassifier

model3= RandomForestClassifier(n_estimators=5, random_state=50,criterion='entropy',max_depth=4)



model3.fit(x_trainstd, y_train)

pred3=model1.predict(x_teststd)

pred3
estimators=model3.estimators_[3]

labels=['case','country','year','systemic_crisis','exch_usd','domestic_debt_in_default','sovereign_external_debt_default','inflation_annual_cpi','independence','currency_crises','inflation_crises']

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(estimators, out_file=None

   , feature_names=labels

   , filled = True))

display(SVG(graph.pipe(format='svg')))

sns.distplot((y_test-pred3),bins=50)

plt.show()
from sklearn.naive_bayes import GaussianNB

model1=GaussianNB()

model1.fit(x_train,y_train)



pred1=model1.predict(x_test)

pred1



from sklearn.metrics import accuracy_score

#from sklearn.metrics import confusion_matrix

print(accuracy_score(y_test, pred1))



from sklearn.naive_bayes import GaussianNB

model2=GaussianNB()

model2.fit(x_trainmin,y_train)



pred2=model2.predict(x_testmin)

pred2



from sklearn.metrics import accuracy_score

#from sklearn.metrics import confusion_matrix

print(accuracy_score(y_test, pred2))



from sklearn.naive_bayes import GaussianNB

model3=GaussianNB()

model3.fit(x_trainstd,y_train)



pred3=model3.predict(x_teststd)

pred3



from sklearn.metrics import accuracy_score

#from sklearn.metrics import confusion_matrix

print(accuracy_score(y_test, pred3))



import pandas as pd

african_crises = pd.read_csv("../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv")