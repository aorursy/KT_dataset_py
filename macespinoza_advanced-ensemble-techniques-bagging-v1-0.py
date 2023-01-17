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
from sklearn.metrics import accuracy_score
import os
print(os.listdir("../input"))
np.random.seed(42)
def get_samples_bagging(data,target,size_col):
    features = np.random.choice(data.columns, size_col, replace=True)
    data = data[features]
    indices = list(set(np.random.randint(data.shape[0], size=data.shape[0])))
    data = data.iloc[indices,:]
    targ = target.iloc[indices,:]
    return data, targ,features
train = pd.read_csv("../input/train.csv",index_col='PassengerId')
test = pd.read_csv("../input/test.csv",index_col='PassengerId')#
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
X.head()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
x_bag1,y_bag1,feat = get_samples_bagging(x_train,y_train,6)
print(x_bag1.head(5))
print(x_bag1.shape)
x_bag1,y_bag1,feat = get_samples_bagging(x_train,y_train,6)
print(x_bag1.head(5))
print(x_bag1.shape)
x_bag1,y_bag1,feat = get_samples_bagging(x_train,y_train,6)
print(x_bag1.head(5))
print(x_bag1.shape)
np.random.seed(2)
x_bag1,y_bag1,feact = get_samples_bagging(x_train,y_train,6)
tmodel1 = DecisionTreeClassifier()
tmodel1.fit(x_bag1, y_bag1)
pd1 =tmodel1.predict_proba(x_test[feact])
pd.DataFrame(tmodel1.predict_proba(x_test[feact])).head(5)
np.random.seed(1767)
x_bag1,y_bag1,feact = get_samples_bagging(x_train,y_train,6)
tmodel2 = DecisionTreeClassifier()
tmodel2.fit(x_bag1, y_bag1)
pd2 =tmodel2.predict_proba(x_test[feact])
pd.DataFrame(tmodel2.predict_proba(x_test[feact])).head(5)
probs = []
probs.append(pd1)
probs.append(pd2)
meand_target = np.mean(probs, axis=0)
pd.DataFrame(np.mean(probs, axis=0)).head()
pd.DataFrame(np.argmax(meand_target, axis=1)).head()
probs = []
model = []

for i in range(10):
    np.random.seed(19+i)
    x_bag1,y_bag1,feact = get_samples_bagging(x_train,y_train,6)
    model.append(DecisionTreeClassifier())
    model[i].fit(x_bag1, y_bag1)
    probs.append(model[i].predict_proba(x_test[feact]))

t = np.mean(probs, axis=0)
y_target =np.argmax(t, axis=1).reshape(-1,1)
accuracy_score(y_test, y_target)
probs = []
model = []
for i in range(100):
    np.random.seed(19+i)
    x_bag1,y_bag1,feact = get_samples_bagging(x_train,y_train,6)
    model.append(DecisionTreeClassifier())
    model[i].fit(x_bag1, y_bag1)
    probs.append(model[i].predict_proba(x_test[feact]))

t = np.mean(probs, axis=0)
y_target =np.argmax(t, axis=1).reshape(-1,1)
accuracy_score(y_test, y_target)
probs = []
model = []
for i in range(1000):
    np.random.seed(19+i)
    x_bag1,y_bag1,feact = get_samples_bagging(x_train,y_train,6)
    model.append(DecisionTreeClassifier())
    model[i].fit(x_bag1, y_bag1)
    probs.append(model[i].predict_proba(x_test[feact]))

t = np.mean(probs, axis=0)
y_target =np.argmax(t, axis=1).reshape(-1,1)
accuracy_score(y_test, y_target)
probs = []
model = []
for i in range(10):
    x_bag1,y_bag1,feact = get_samples_bagging(x_train,y_train,6)
    model.append(DecisionTreeClassifier())
    model[i].fit(x_bag1, y_bag1)
    probs.append(model[i].predict_proba(test_f[feact]))

t = np.mean(probs, axis=0)
pd.DataFrame(t).head()
y_target =np.argmax(t, axis=1).reshape(-1,1)
d_ytarget = pd.DataFrame(y_target)
test_f_salida = pd.DataFrame( { 'PassengerId': test_f.index , 'Survived': d_ytarget[0]} )
#Show Output
test_f_salida.head(20)
test_f_salida.to_csv( 'titanic_pred.csv' , index = False )