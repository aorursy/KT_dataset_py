import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import numpy as np
exemplo=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
data=pd.read_csv('/kaggle/input/titanic/train.csv',index_col='PassengerId')
dataTest=pd.read_csv('/kaggle/input/titanic/test.csv',index_col='PassengerId')
data=data.dropna(axis=0)
data.head()
#Prediction Target
y=data.Survived
#Features choose
columnas=['Pclass','Sex','Age']
X=data[columnas]
X.head()
#data must be numerical
X['Sex']=X.Sex.map(lambda x: 1 if x=='male' else 0)
X
data_model=DecisionTreeRegressor(random_state=1)
data_model.fit(X,y)

prediccions=data_model.predict(X)
z=[x1-x2 for (x1,x2) in zip(prediccions,list(y))]
#Success ratio 
z.count(0)/len(z)
Xtest=dataTest[columnas]
Xtest['Sex']=Xtest.Sex.map(lambda x: 1 if x=='male' else 0)
Xtest
#poñemos
#dataTest.loc[np.isnan(dataTest.Age)]
# Temos que poñer a idade dos NaN -> media
ageM=data.Age.mean()
Xtest=Xtest.fillna(ageM)
pred=data_model.predict(Xtest)
predR=[int(round(x)) for x in pred]
len(predR), predR.count(1)
submit=pd.DataFrame({'Survived':predR}, index=dataTest.index)
#submit.to_csv('/home/manu/Programacion/kaggle/titanic/TreeSKlearn1.csv')
from sklearn.model_selection import train_test_split
train_X,val_X,train_y, val_y=train_test_split(X,y,random_state=1)

for mln in [2,5,8,10,20,100]:
    train_X_model=DecisionTreeRegressor(max_leaf_nodes=mln,random_state=2)
    train_X_model.fit(train_X,train_y)
    pred=train_X_model.predict(val_X)
    predR=[int(round(x)) for x in pred]
    z=[x1-x2 for (x1,x2) in zip(predR,list(val_y))]
    #Success ratio 
    sR=z.count(0)/len(z)
    print("Max Leaf Nodes "+str(mln)+"\t\t Success Ratio: "+"{0:.0%}".format(sR))
    
mln=10
data_model=DecisionTreeRegressor(max_leaf_nodes=mln,random_state=2)
data_model.fit(X,y)
pred=train_X_model.predict(Xtest)
predR=[int(round(x)) for x in pred]
len(predR), predR.count(1)
submit=pd.DataFrame({'Survived':predR}, index=dataTest.index)
#submit.to_csv('/home/manu/Programacion/kaggle/titanic/TreeSKlearnMLN.csv')