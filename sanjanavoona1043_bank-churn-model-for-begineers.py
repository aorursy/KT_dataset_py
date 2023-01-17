import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
#data data
df =pd.read_csv('../input/bank-churn/Bank_churn_modelling.csv')
df.shape
df.head()
df.columns
df.info()
df.describe()
df.Gender.unique()
df.Geography.unique()
df.profile_report()
#check for duplicate entries
df.duplicated().sum()
# check for missing values
df.isnull().sum()
df.columns
#df.drop(["Surname","RowNumber","CustomerId"],axis=1,inplace=True)
df.columns
x=df[['CreditScore', 'Geography', 'Gender', 'Age','Balance','NumOfProducts', 'IsActiveMember']]
y=df["Exited"]
x.head()
from sklearn.preprocessing import LabelEncoder
lel = LabelEncoder()
x["Gender"]= lel.fit_transform(x["Gender"])
x.head()
y.head()
#here in Gender column female changed to 0 and male changed to 1
x.head(8)
#onehotencoding for geography
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("encoder",OneHotEncoder(),[1])],remainder="passthrough")# here 1 represents column geography
#here we can use remainder="drop" ,here it only represents geography
x = ct.fit_transform(x)
x.shape
x=pd.DataFrame(x)
x.head(10)
#here 0-france 1-germany 2-spain (in alaphabetical model)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x=sc.fit_transform(x)
#splitting data info train and test set
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts = train_test_split(x,y,test_size=0.2)
print(x.shape)
print(xtr.shape)
print(xts.shape)
from sklearn.linear_model import LogisticRegression
model =LogisticRegression()
#train the model -using training data -xtr,ytr
model.fit(xtr,ytr)
#france, cs=580, age=58, Male,numofprod=3,isactmember=0,balance=456782
new_customer=[[1,0,0,580,1,58,456782,3,0]]
model.predict(new_customer)
#here [1] is exited
#check perfomance of model on test data
# getting prediction for test data
ypred = model.predict(xts)
from sklearn import metrics
metrics.accuracy_score(yts,ypred)
#here the accuracy which we got is 0.79 which is not good 
#after adding standardization it become 80% 0r 0.80
#calcuate recall
metrics.recall_score(yts,ypred)
#here we used feature scaling to get features in same range
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors = 3)#no of neighbors is hpyer parameter
model2.fit(xtr,ytr)

ypred2=model2.predict(xts)
metrics.accuracy_score(yts,ypred2)#here it checks for both class 1 and class 0( here 0 and 1  are told as class)
metrics.recall_score(yts,ypred2)#here it ckecks only for class 1
from sklearn.tree import DecisionTreeClassifier
model3= DecisionTreeClassifier(criterion="gini")
#here we are facing the problem of overfitting
#train the model
model3.fit(xtr,ytr)
ypred3=model3.predict(xts)
metrics.accuracy_score(yts,ypred3)
metrics.recall_score(yts,ypred3)
from sklearn.tree import DecisionTreeClassifier
model3= DecisionTreeClassifier(criterion="entropy")
#train the model
model3.fit(xtr,ytr)
ypred3=model3.predict(xts)
metrics.accuracy_score(yts,ypred3)
metrics.recall_score(yts,ypred3)
metrics.recall_score(ytr,model3.predict(xtr))
from sklearn.tree import DecisionTreeClassifier
model3= DecisionTreeClassifier(criterion="gini",max_depth=8,min_samples_leaf=10)
#here max_depth and min_samples_leaf is used to control overfitting
#train the model
model3.fit(xtr,ytr)
ypred3=model3.predict(xts)
metrics.accuracy_score(yts,ypred3)
metrics.recall_score(yts,ypred3)
metrics.recall_score(ytr,model3.predict(xtr))
import graphviz
from sklearn import tree
fname=['France','Germany','Spain','CreditScore','Gender','Age','Balance','NumofProducts','IsActiveMember']
cname=['Not Excited','Excited']
graph_data = tree.export_graphviz(model3,out_file=None,feature_names=fname,class_names=cname,filled=True,rounded=True)
graph=graphviz.Source(graph_data)
from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(n_estimators=50,criterion='gini',max_depth=10,min_samples_leaf=20)
model4.fit(xtr,ytr)
ypred4=model4.predict(xts)
metrics.accuracy_score(yts,ypred4)
metrics.recall_score(yts,ypred4)
metrics.recall_score(ytr,model4.predict(xtr))