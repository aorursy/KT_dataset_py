import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm as tq

from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler, RobustScaler

%matplotlib inline
df = pd.read_csv("train.csv")
df.corr()
numerical_features = df.columns

numerical_features = ['chem_0','chem_1','chem_4','chem_5','chem_6','attribute']

X = df[numerical_features]

y = df["class"]
from sklearn.model_selection import train_test_split

data = []

for i in range(10):

    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=i)  #Checkout what does random_state do

    data.append((X_train,X_val,y_train,y_val))
#TODO

from sklearn.preprocessing import StandardScaler



scaler = RobustScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

X_val[numerical_features] = scaler.transform(X_val[numerical_features])  



# It is important to scale tain and val data separately because val is supposed to be unseen data on which we test our models. If we scale them together, data from val set will also be considered while calculating mean, median, IQR, etc



X_train[numerical_features].head()
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

#from xgboost.sklearn import XGBClassifier

from sklearn import *

# Initialize and train

acc1 = []

acc2 = []

acc3 = []

acc4 = []

acc5 = []

for i in range(10):

    X_train,X_val,y_train,y_val = data[i]

    clf1 = AdaBoostClassifier().fit(X_train,y_train)

    clf2 = RandomForestClassifier().fit(X_train,y_train)

    clf3 = ExtraTreesClassifier(n_estimators=60,max_depth=18,random_state=42).fit(X_train,y_train)

    clf4 = GradientBoostingClassifier().fit(X_train,y_train)

    clf5 = neural_network.MLPClassifier().fit(X_train,y_train)



    y_pred_1 = clf1.predict(X_val)

    y_pred_2 = clf2.predict(X_val)

    y_pred_3 = clf3.predict(X_val)

    y_pred_4 = clf4.predict(X_val)

    y_pred_5 = clf5.predict(X_val)



    acc1 += [accuracy_score(y_pred_1,y_val)*100]

    acc2 += [accuracy_score(y_pred_2,y_val)*100]

    acc3 += [accuracy_score(y_pred_3,y_val)*100]

    acc4 += [accuracy_score(y_pred_4,y_val)*100]

    acc5 += [accuracy_score(y_pred_5,y_val)*100]

    

    



print(sum(acc1)/10)

print(sum(acc2)/10)

print(sum(acc3)/10)

print(sum(acc4)/10)

print(sum(acc5)/10)



#print("1:",sum(acc1)/10)
from sklearn.metrics import accuracy_score  #Find out what is accuracy_score



y_pred_1 = clf1.predict(X_val)

y_pred_2 = clf2.predict(X_val)

y_pred_3 = clf3.predict(X_val)

y_pred_4 = clf4.predict(X_val)

y_pred_5 = clf5.predict(X_val)



acc1 = accuracy_score(y_pred_1,y_val)*100

acc2 = accuracy_score(y_pred_2,y_val)*100

acc3 = accuracy_score(y_pred_3,y_val)*100

acc4 = accuracy_score(y_pred_4,y_val)*100

acc5 = accuracy_score(y_pred_5,y_val)*100



print("Accuracy score of clf1: {}".format(acc1))

print("Accuracy score of clf2: {}".format(acc2))

print("Accuracy score of clf3: {}".format(acc3))

print("Accuracy score of clf4: {}".format(acc4))

print("Accuracy score of clf5: {}".format(acc5))


n_estimators_l = [5,50,100]

max_depth_l = [5,15,20,35,50,75,100]




score_train_XT = []

score_test_XT = []

mi=0

bi=0

bj=0

for i in tq(range(90,110)):

    for j in range(5,20):

        acc = []

        for k in range(10):

            X_train,X_val,y_train,y_val = data[k]

            xt = ExtraTreesClassifier(n_estimators=i, max_depth=j, random_state = 42)

            xt.fit(X_train,y_train)

            acc += [accuracy_score(y_val,xt.predict(X_val))]

        rmse_test = sum(acc)/10

        if (rmse_test>mi):

            mi=rmse_test

            bi=[i]

            bj=[j]

        

print(mi,bi,bj)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier()

parameters = {'random_state':[1234,1331,9999],'n_estimators':[1996,1998,2000,2002]}

scorer = make_scorer(accuracy_score,greater_is_better=True) 

grid_obj = GridSearchCV(rf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X,y)

bestrf = grid_fit.best_estimator_  

out = pd.read_csv("test.csv")
#TODO

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X[numerical_features] = scaler.fit_transform(X[numerical_features])

out[numerical_features] = scaler.transform(out[numerical_features])  

#y_pred_1 = clf1.predict(out[numerical_features])

#clf2 = ExtraTreesClassifier(n_estimators=60,max_depth=18).fit(X,y)

clf2 =  ExtraTreesClassifier().fit(X,y)   #sol1

clf2 = bestrf  #sol2                        

y_pred_2 = clf2.predict(out[numerical_features])
y_pred = y_pred_2

y_pred
look =([5, 1, 7, 3, 7, 7, 2, 5, 1, 2, 1, 7, 2, 2, 2, 2, 1, 1, 2, 7, 1, 1, 2,

       7, 3, 2, 1, 2, 2, 2, 5, 2, 2, 2, 3, 1, 1, 2, 1, 2, 7, 5, 2, 2, 2, 1,

       7, 1, 1, 7, 2, 2, 2, 6, 2, 2, 7, 1, 2, 2, 6, 1, 7, 1, 7, 2, 1, 2, 2,

       2, 1, 2, 2, 1, 2, 2, 7, 1, 1, 2])
upload = pd.concat([out.id,pd.DataFrame(data=y_pred)],axis=1)

upload.columns = ['id','class']

upload.to_csv('submit.csv',index=False)
upload.groupby(['class']).count()