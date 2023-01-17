import numpy as np 
import pandas as pd 

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os
print(os.listdir("../input"))
np.random.seed(0)
train = pd.read_csv("../input/train.csv",index_col='PassengerId')
test = pd.read_csv("../input/test.csv",index_col='PassengerId')#
train.isna().sum()
test.isna().sum()
train.shape,test.shape
train.head()
def replaceGen(sex):
    gen =0
    if sex=='male':
        gen=0
    elif sex=='female':
        gen=1
    return gen
    
train['Sex'] = train['Sex'].apply(replaceGen)
test['Sex'] = test['Sex'].apply(replaceGen)
train['Age'].hist(figsize=(10, 4));
train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)
test[test['Fare'].isna()]
Age_mean = train[(train['Pclass']==3) & (train['Embarked']=='S') & (train['Age']>55) & (train['Sex']==0)]['Fare'].mean()
test['Fare'].fillna(Age_mean, inplace=True)
X =train.drop(['Survived','Name','Ticket','Cabin','Embarked'],axis=1)
y =pd.DataFrame(train['Survived'])
test_f =test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
X.shape,y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train.shape,y_train.shape
X.head()
model1 = DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()
model4 = RandomForestClassifier()
model5 = GradientBoostingClassifier()
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)
model4.fit(X_train,y_train)
model5.fit(X_train,y_train)
pred1=model1.predict(X_test)
pred2=model2.predict(X_test)
pred3=model3.predict(X_test)
pred4=model4.predict(X_test)
pred5=model5.predict(X_test)
print("Modelo 1 DecisionTreeClassifier",model1.score(X_test,y_test))
print("Modelo 2 KNeighborsClassifier",model2.score(X_test,y_test))
print("Modelo 3 LogisticRegression",model3.score(X_test,y_test))
print("Modelo 4 RandomForestClassifier",model4.score(X_test,y_test))
print("Modelo 5 GradientBoostingClassifier",model5.score(X_test,y_test))
def Stacking(model,train,y,test,n_fold,t=1):
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred=np.empty((0,1),float)
    train_pred=np.empty((0,1),float)
    for train_indices,val_indices in folds.split(train,y.values):
        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]
        if t==1:
            model.fit(X=x_train,y=y_train)
        else:
            model.train(x_train,y_train)
        train_pred=np.append(train_pred,model.predict(x_val))
    test_pred=np.append(test_pred,model.predict(test))
    return test_pred.reshape(-1,1),train_pred

#Number of Folds
nfolds = 5
model1 = DecisionTreeClassifier()

test_pred1 ,train_pred1=Stacking(model=model1,n_fold=nfolds, train=X_train,test=X_test,y=y_train)

train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)
model2 = KNeighborsClassifier()

test_pred2 ,train_pred2=Stacking(model=model2,n_fold=nfolds,train=X_train,test=X_test,y=y_train)

train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)
model3 = RandomForestClassifier()

test_pred3 ,train_pred3=Stacking(model=model3,n_fold=nfolds,train=X_train,test=X_test,y=y_train)

train_pred3=pd.DataFrame(train_pred3)
test_pred3=pd.DataFrame(test_pred3)
model4 = GradientBoostingClassifier()

test_pred4 ,train_pred4=Stacking(model=model4,n_fold=nfolds,train=X_train,test=X_test,y=y_train)

train_pred4=pd.DataFrame(train_pred4)
test_pred4=pd.DataFrame(test_pred4)
df = pd.concat([train_pred1, train_pred2, train_pred3, train_pred4], axis=1)
df_test = pd.concat([test_pred1, test_pred2, test_pred3, test_pred4], axis=1)

model = LogisticRegression(random_state=1)
model.fit(df,y_train)
#y_test = model.predict(df_test)
##df_test
#y_test
print("HiperModelo LogisticRegression",model.score(df_test, y_test))
test_pred1 ,train_pred1=Stacking(model=model1,n_fold=nfolds, train=X,test=test_f,y=y,t=1)
test_pred2 ,train_pred2=Stacking(model=model2,n_fold=nfolds,train=X,test=test_f,y=y,t=1)
test_pred3 ,train_pred3=Stacking(model=model3,n_fold=nfolds,train=X,test=test_f,y=y,t=1)
test_pred4 ,train_pred4=Stacking(model=model4,n_fold=nfolds,train=X,test=test_f,y=y,t=1)

train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)

train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)

train_pred3=pd.DataFrame(train_pred3)
test_pred3=pd.DataFrame(test_pred3)

train_pred4=pd.DataFrame(train_pred4)
test_pred4=pd.DataFrame(test_pred4)

df = pd.concat([train_pred1, train_pred2, train_pred3, train_pred4], axis=1)
df_test = pd.concat([test_pred1, test_pred2, test_pred3, test_pred4], axis=1)

model = LogisticRegression(random_state=1)
model.fit(df,y)
y_target = model.predict(df_test)
test_salida = pd.DataFrame( { 'PassengerId': test_f.index , 'Survived': y_target } )
#Show Output
test_salida.head(20)
#Generate file
test_salida.to_csv( 'titanic_pred.csv' , index = False )