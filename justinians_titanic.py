# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# the first 3 entries
train_df.head(5)
ntr=int(train_df.shape[0]*0.6)
ncv=int(train_df.shape[0]*0.2)
ntt=train_df.shape[0]-ntr-ncv
nsub=test_df.shape[0]
# print info 
train_df.info()
print('_'*40)
test_df.info()
# shuffle the dataset, split X and Y, pick the first 60% rows to form the training set
train_df = train_df.sample(frac=1).reset_index(drop=True)

Y_tot=train_df['Survived']
X_tot=train_df.drop('Survived',axis=1).append(test_df,ignore_index=True)
X_train=X_tot.head(ntr) #only take the first ntr rows to be the training set
# non-digital features are Name, Sex, Ticket, Cabin, Embarked
# the columns need to be filled are Age(714), Cabin(204), Embarked(889) for the training set (891)
#                                   Age(332), Fare(417), Cabin(91) for the test set (418)

# drop passenger id, name and ticket number 
X_tot=X_tot.drop(['PassengerId','Name','Ticket'],axis=1)

# map the sex into 0(female) and 1 (male)
X_tot['Sex']=X_tot['Sex'].map({'female':0, 'male':1})
# fill in the nan for column embarked and map the values to numbers
portmax=X_train['Embarked'].dropna().mode()[0]
X_tot['Embarked']=X_tot['Embarked'].fillna(portmax)

ports=X_train.Embarked.dropna().unique()
ports_map={}
for i in range(len(ports)):
    ports_map.update({ports[i]:i})
X_tot['Embarked']=X_tot['Embarked'].map(ports_map)
# add a column primarycabin
X_train['PrimaryCabin']=X_train['Cabin'].fillna('N').str.split().apply(sorted).astype(str).str[2] # on the training set
X_tot['PrimaryCabin']=X_tot['Cabin'].fillna('N').str.split().apply(sorted).astype(str).str[2]       # both training and test set
# generate random cabins for null cabin
cabinlist=X_train['PrimaryCabin'].tolist()  # extract list
cabinlist=[ele for ele in cabinlist if ele != 'N'] # remove 'N
rnd.shuffle(cabinlist)     # shuffle

# fill null cabin in X_tot
for i in range(X_tot.shape[0]):
    X_tot.loc[i,'PrimaryCabin']=(X_tot.loc[i,'PrimaryCabin']=='N')*rnd.choice(cabinlist)+(X_tot.loc[i,'PrimaryCabin']!='N')*X_tot.loc[i,'PrimaryCabin']
# map the cabins to integers
#X_tot['PrimaryCabin']=X_tot['PrimaryCabin'].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7})
X_tot['PrimaryCabin']=X_tot['PrimaryCabin'].apply(lambda x:ord(x))
# drop passenger id, name and ticket number 
X_tot=X_tot.drop(['Cabin'],axis=1)
# fill in null for Age and Fare
# train_df['Age'].hist(bins=100)
agelist=X_train['Age'].fillna(-1).tolist()
X_tot['Age']=X_tot['Age'].fillna(-1)

agelist=[age for age in agelist if age!=-1] 
rnd.shuffle(agelist)
for i in range(X_tot.shape[0]):
    X_tot.loc[i,'Age']=(X_tot.loc[i,'Age']<0)*rnd.choice(agelist)+(X_tot.loc[i,'Age']>0)*X_tot.loc[i,'Age']
# fill in null for fare
# train_df['Fare'].hist(bins=10)
farelist=X_train['Fare'].fillna(-1).tolist()
X_tot['Fare']=X_tot['Fare'].fillna(-1)

farelist=[fare for fare in farelist if fare!=-1]
rnd.shuffle(farelist)
for i in range(X_tot.shape[0]):
    X_tot.loc[i,'Fare']=(X_tot.loc[i,'Fare']<0)*rnd.choice(farelist)+(X_tot.loc[i,'Fare']>0)*X_tot.loc[i,'Fare']
# feature scaling: outliers exist, use robust scaling
#scaler = MinMaxScaler()
scaler = RobustScaler()

X_scaled = scaler.fit_transform(X_tot)
X_scaled = pd.DataFrame(X_scaled, columns=X_tot.columns.values)


# split the data
X=X_scaled.head(ntr)
X_cv=X_scaled.loc[ntr:ntr+ncv-1]
X_tt=X_scaled.loc[ntr+ncv:ntr+ncv+ntt-1]
X_sub=X_scaled.tail(nsub)
#X=X_tot.head(ntr)
#X_cv=X_tot.loc[ntr:ntr+ncv]
#X_tt=X_tot.loc[ntr+ncv:ntr+ncv+ntt]

Y=Y_tot.head(ntr)
Y_cv=Y_tot.loc[ntr:ntr+ncv-1]
Y_tt=Y_tot.tail(ntt)
# naive bayes with Gaussian
gaussian = GaussianNB()
gaussian.fit(X, Y)
#Y_pred = gaussian.predict(X_test)
#acc_gaussian = round(gaussian.score(X, Y) * 100, 2)
#acc_gaussian
Y_cv_gaussian=gaussian.predict(X_cv)
acc_gaussian=(Y_cv_gaussian==Y_cv).value_counts(True)[True]
acc_gaussian
# support vector machine
# vary the regularization parameter to achieve best accuracy

Clist=10**np.linspace(-1,2,100)
acc_svc_list=[]
for Cvalue in Clist:
    svc = SVC(C=Cvalue)
    svc.fit(X, Y)
    Y_cv_svc=svc.predict(X_cv)
    acc_svc_list.append((Y_cv_svc==Y_cv).value_counts(True)[True])
    
plt.figure()
plt.xscale('log')
plt.plot(Clist,acc_svc_list,'x')
plt.show()

Cbest=Clist[acc_svc_list.index(max(acc_svc_list))]
svc=SVC(C=Cbest)
svc.fit(X,Y)
Y_cv_svc=svc.predict(X_cv)
print((Y_cv_svc==Y_cv).value_counts(True)[True])
# decision tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X, Y)
#Y_pred = decision_tree.predict(X_test)

Y_cv_dt=decision_tree.predict(X_cv)
acc_dt=(Y_cv_dt==Y_cv).value_counts(True)[True]
acc_dt
#  Logistic Regression

Clist=10**np.linspace(-1,0,100)
acc_lr_list=[]
for Cvalue in Clist:
    logreg = LogisticRegression(C=Cvalue)
    logreg.fit(X, Y)
    Y_cv_lr=logreg.predict(X_cv)
    acc_lr_list.append((Y_cv_lr==Y_cv).value_counts(True)[True])

    
plt.figure()
plt.xscale('log')
plt.plot(Clist,acc_lr_list,'x')
plt.show()


Cbest=Clist[acc_lr_list.index(max(acc_lr_list))]
logreg=LogisticRegression(C=Cbest)
logreg.fit(X,Y)
Y_cv_lr=logreg.predict(X_cv)
print(logreg.score(X_cv,Y_cv))
# neural network

alist=10**np.linspace(-6,0,20)

acc_mlp_list=[]
for avalue in alist:
    mlp = MLPClassifier(hidden_layer_sizes=(10,5),max_iter=2000,alpha=avalue)
    mlp.fit(X,Y)
    acc_mlp_list.append(mlp.score(X_cv,Y_cv))
    
    
plt.figure()
plt.xscale('log')
plt.plot(alist,acc_mlp_list,'x')
plt.show()


abest=alist[acc_mlp_list.index(max(acc_mlp_list))]
mlp=MLPClassifier(hidden_layer_sizes=(10,10),max_iter=2000,alpha=abest)
mlp.fit(X,Y)
#Y_cv_mlp=mlp.predict(X_cv)
print(mlp.score(X_cv,Y_cv))
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X, Y)
#Y_pred = knn.predict(X_test)
print(knn.score(X_cv, Y_cv) )
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, Y)
print(random_forest.score(X_cv, Y_cv))
# it seems SVM gives best accura
print([gaussian.score(X_tt,Y_tt),svc.score(X_tt,Y_tt),decision_tree.score(X_tt,Y_tt),logreg.score(X_tt,Y_tt),mlp.score(X_tt,Y_tt),knn.score(X_tt,Y_tt),random_forest.score(X_tt,Y_tt)])
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": logreg.predict(X_sub)
    })
submission.to_csv('submission.csv', index=False)
