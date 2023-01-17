import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Load the data-set.
data = pd.read_csv('../input/winequality-red.csv')
data.head()
data.describe()
data.info()
data[data.isnull()].count()
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True)
data['quality'].unique()
from collections import Counter
Counter(data['quality'])
fig = plt.figure(figsize = (10,6))
plt.hist(data["quality"].values, range=(1, 10))
plt.xlabel('Ratings of wines')
plt.ylabel('Amount')
plt.title('Distribution of wine ratings')
plt.show()
Quality_count=[681,638,199,53,18,10]
Quality_labels=['5','6','7','4','8','3']
plt.pie(Quality_count,labels=Quality_labels,radius=2,autopct='%0.1f%%',shadow=True)
sns.pairplot(data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = data)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = data)
fig = plt.figure(figsize = (10,6))
sns.pointplot(x=data['pH'].round(1),y='residual sugar',color='green',data=data)
fig = plt.figure(figsize = (10,6))
sns.pointplot(y=data['pH'].round(1),x='quality',color='green',data=data)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
 
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score 

bins = (2, 6.5, 8)
group_names = ['bad', 'good']
data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
data['quality'] = label_quality.fit_transform(data['quality'])
x_train,x_test,y_train,y_test=train_test_split(data.drop('quality',axis=1),data['quality'],test_size=0.25,random_state=42)

models=[LogisticRegression(),
        LinearSVC(),
        SVC(kernel='rbf'),
        KNeighborsClassifier(),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        GradientBoostingClassifier(),
        GaussianNB()]

model_names=['LogisticRegression',
             'LinearSVM',
             'rbfSVM',
             'KNearestNeighbors',
             'RandomForestClassifier',
             'DecisionTree',
             'GradientBoostingClassifier',
             'GaussianNB']

acc=[]
d={}

for model in range(len(models)):
    classification_model=models[model]
    classification_model.fit(x_train,y_train)
    pred=classification_model.predict(x_test)
    acc.append(accuracy_score(pred,y_test))
     
d={'Modelling Algorithm':model_names,'Accuracy':acc}
d
acc_table=pd.DataFrame(d)
acc_table
sns.barplot(y='Modelling Algorithm',x='Accuracy',data=acc_table)
sns.factorplot(x='Modelling Algorithm',y='Accuracy',data=acc_table,kind='point',size=4,aspect=3.5)
def func(x_train,x_test,y_train,y_test,name_scaler):
    models=[LogisticRegression(),LinearSVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),
        DecisionTreeClassifier(),GradientBoostingClassifier(),GaussianNB()]
    acc_sc=[]
    for model in range(len(models)):
        classification_model=models[model]
        classification_model.fit(x_train,y_train)
        pred=classification_model.predict(x_test)
        acc_sc.append(accuracy_score(pred,y_test))
     
    acc_table[name_scaler]=np.array(acc_sc)
scalers=[MinMaxScaler(),StandardScaler()]
names=['Acc_Min_Max_Scaler','Acc_Standard_Scaler']
for scale in range(len(scalers)):
    scaler=scalers[scale]
    scaler.fit(data)
    scaled_data=scaler.transform(data)
    X=scaled_data[:,0:11]
    Y=data['quality'].values
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
    func(x_train,x_test,y_train,y_test,names[scale])
acc_table
sns.barplot(y='Modelling Algorithm',x='Accuracy',data=acc_table)
sns.barplot(y='Modelling Algorithm',x='Acc_Standard_Scaler',data=acc_table)
sns.barplot(y='Modelling Algorithm',x='Acc_Min_Max_Scaler',data=acc_table)

