import numpy as np
import pandas as pd
data=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data
data.info()
## check columnwise missing values
data.isnull().sum()
data.drop(['salary',"sl_no"], axis = 1,inplace=True)
data.describe(include='object')
#Import matplotlib and seaborn for data visualisation
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
fig, ax1 = plt.subplots(figsize=(12,5))
graph=sns.countplot(x='status',data=data,order=data.status.value_counts().index)
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
i=0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,
        data['status'].value_counts()[i],ha="center")
    i += 1
sns.countplot(data["workex"])
# Change target varaible to numeric(0 and1)
data["status"]=data.apply(lambda x: 1 if x.status=="Placed" else 0,axis=1)
# variable 'workex'
data.groupby("workex")["status"].mean()
sns.barplot(x='workex',y='status',data=data)
data.groupby("gender")["status"].mean()
data.groupby("ssc_b")["status"].mean()
#### similarly check hsc_b(board of eduction in hsc)
data.groupby("hsc_b")["status"].mean()
#### hsc_s (hsc specialization)
data.groupby("hsc_s")["status"].mean()

sns.barplot(x='hsc_s',y='status',data=data)
data['hsc_s']=data['hsc_s'].apply(lambda x: "com/sci" if (x=="Commerce" or x=="Science") else "Arts")
data.groupby("degree_t")["status"].mean()
data['degree_t']=data['degree_t'].apply(lambda x: "Com/sci" if (x=="Comm&Mgmt" or x=="Sci&Tech") else "Others")
data.groupby("specialisation")["status"].mean()
data.describe()
#### correlation with target
data.corr()
plt.figure(figsize=(8,5))
sns.heatmap(data.corr(),
            vmin=-1,
            cmap='coolwarm',
            annot=True);
sns.pairplot(data,hue='status')
##lets check ssc_p
sns.distplot(data["ssc_p"])
sns.boxplot("status","ssc_p",data=data)

#### Boxplot clearly shows that higher mean ssc_p are in Placed status(1)
sns.factorplot(x='gender',y='ssc_p' , col='workex', data=data , hue='status' , kind = 'box', palette=['r','g'])
sns.factorplot(x='gender',y='ssc_p' , col='workex', data=data , hue='status' , kind = 'violin', palette=['r','g'])
data = pd.get_dummies( data,drop_first=True)
X = data.drop('status', axis=1)
y = data['status']
from sklearn.model_selection import train_test_split
# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, predictions)
auc
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))


auc = roc_auc_score(y_test, rfc_pred)
auc