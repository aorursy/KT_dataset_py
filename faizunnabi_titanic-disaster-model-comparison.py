import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.head()
df.info()
df.describe()
sns.heatmap(df.isnull(),cmap='viridis',cbar=False,yticklabels=
           False)
df1 = df.drop('Cabin',axis=1)
sns.heatmap(df1.isnull(),cmap='viridis',cbar=False,yticklabels=
           False)
sns.countplot(x="Survived",data=df1)
df1.columns
sns.set_style('darkgrid')
sns.boxplot(x='Pclass',y='Age',data=df1)
sns.countplot(x='Survived',hue='Sex',data=df1)
plt.figure(figsize=(14,6))
sns.violinplot(x='SibSp',y='Age',hue='Survived',data=df1)
df1.columns
df2=df1.drop(['PassengerId','Name','Ticket'],axis=1)
df2.head()
df2.info()
sns.boxplot(x='Embarked',y='Fare',data=df2)
sns.countplot(x='Embarked',data=df2)
df2.groupby('Embarked')['Fare'].mean()
df2[df2['Embarked'].isnull()]
df2.loc[61,'Embarked']='C'
df2.loc[829,'Embarked']='C'
df2.groupby('Pclass')['Age'].mean()
def compute_age(cols):
    age=cols[0]
    pc=cols[1]
    if pd.isnull(age):
        if pc==1:
            return 38
        elif pc==2:
            return 30
        elif pc==3:
            return 25
    else:
        return age
df2['Age'] = df2[['Age','Pclass']].apply(compute_age,axis=1)
sns.heatmap(df2.isnull(),cmap='viridis',yticklabels=False,cbar=False)
sex = pd.get_dummies(df2['Sex'],drop_first=True)
embark = pd.get_dummies(df2['Embarked'],drop_first=True)
df3=df2.drop(['Sex','Embarked'],axis=1)
df3 = pd.concat([df3,sex,embark],axis=1)
df3.head()
X=df3.drop('Survived',axis=1)
y=df3['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(X_train,y_train)
preds=lgr.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,preds))
from sklearn.svm import SVC
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid_predictions = grid.predict(X_test)
print(classification_report(y_test,grid_predictions))
from sklearn.neighbors import KNeighborsClassifier
error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=12)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('\n')
print(classification_report(y_test,pred))
df_test = pd.read_csv('../input/test.csv')
df_test.head()
df_test = df_test.drop('Cabin',axis=1)
df_test=df_test.drop(['PassengerId','Name','Ticket'],axis=1)
df_test.info()
df_test[df_test['Fare'].isnull()]
plt.figure(figsize=(14,6))
sns.boxplot(x='Pclass',y='Fare',hue='Embarked',data=df_test)
df_test.groupby(['Pclass','Embarked'])['Fare'].mean()
df_test.loc[152,'Fare']=14
df_test[df_test['Fare'].isnull()]
df_test['Age'] = df_test[['Age','Pclass']].apply(compute_age,axis=1)
df_test.info()
se = pd.get_dummies(df_test['Sex'],drop_first=True)
emb = pd.get_dummies(df_test['Embarked'],drop_first=True)
dft=df_test.drop(['Sex','Embarked'],axis=1)
dft = pd.concat([dft,se,emb],axis=1)
dft.head()
predictions=lgr.predict(dft)
sub_df = pd.read_csv('../input/test.csv')
sub_id=sub_df['PassengerId'].values
sbdf = pd.DataFrame({"PassengerId":sub_id,"Survived":predictions})
sbdf.head(10)