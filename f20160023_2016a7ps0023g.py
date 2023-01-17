import pandas as pd

import numpy as np

import sklearn as sk

%matplotlib inline

pd.set_option('display.max_columns', 100)
fi = "../input/eval-lab-1-f464-v2/train.csv"
data = pd.read_csv(fi)
data.head()
data.dropna( inplace=True)
data.describe()
data["type"]=data.type.eq("new").mul(1)
data.head()
from sklearn.model_selection import train_test_split
x = data.drop(["id", "rating"], axis = 1)

y = data["rating"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

d = RandomForestClassifier(n_estimators=450,random_state=0)
d.fit(x_train,y_train)
sol=d.predict(x_test)

for i in range(len(sol)):

    sol[i]=round(sol[i])

sol = sol.astype(np.int64)



from sklearn.metrics import mean_squared_error

from math import sqrt

rms=sqrt(mean_squared_error(y_test,sol))

print(rms)
x = data.drop(["id", "rating"], axis = 1)

y = data["rating"]

LR= RandomForestClassifier(n_estimators=450,random_state=0)
LR.fit(x,y)
testfile = "../input/eval-lab-1-f464-v2/test.csv"

testdata = pd.read_csv(testfile)

testdata["type"]=testdata.type.eq("new").mul(1)

testdata.fillna( testdata.mean(),inplace=True)

Outtest=testdata

testdata = testdata.drop(["id"], axis = 1)
out=LR.predict(testdata)
out
for i in range(len(out)):

    out[i]=round(out[i])



out=out.astype(np.int64)

len(out)
Outtest_id=Outtest["id"]

Output=list(zip(Outtest["id"],out))

out=list(out)

Outtest_id=list(Outtest_id)

   

  

dic = {'id': Outtest_id, 'rating': out}  

     

df = pd.DataFrame(dic) 

  



df.to_csv('Output1.csv',index=False) 



#2nd solution



import pandas as pd

import numpy as np

import sklearn as sk

%matplotlib inline

pd.set_option('display.max_columns', 100)







fi = "../input/eval-lab-1-f464-v2/train.csv"



data = pd.read_csv(fi)



data.head()



data.dropna( inplace=True)



data.describe()



data["type"]=data.type.eq("new").mul(1)



data.head()



from sklearn.model_selection import train_test_split



x = data.drop(["id", "rating"], axis = 1)

y = data["rating"]



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)



from sklearn.ensemble import RandomForestClassifier

d = RandomForestClassifier(n_estimators=100,random_state=0)



d.fit(x_train,y_train)

sol=d.predict(x_test)





for i in range(len(sol)):

    sol[i]=round(sol[i])





sol = sol.astype(np.int64)







from sklearn.metrics import mean_squared_error

from math import sqrt

rms=sqrt(mean_squared_error(y_test,sol))

print(rms)



x = data.drop(["id", "rating"], axis = 1)

y = data["rating"]

LR= RandomForestClassifier(n_estimators=100,random_state=0)



LR.fit(x,y)



testfile = "../input/eval-lab-1-f464-v2/test.csv"

testdata = pd.read_csv(testfile)

testdata["type"]=testdata.type.eq("new").mul(1)

testdata.fillna( testdata.mean(),inplace=True)

Outtest=testdata

testdata = testdata.drop(["id"], axis = 1)



out=LR.predict(testdata)



out



for i in range(len(out)):

    out[i]=round(out[i])



out=out.astype(np.int64)

len(out)

Outtest_id=Outtest["id"]

Output=list(zip(Outtest["id"],out))

out=list(out)

Outtest_id=list(Outtest_id)

   

  

dic = {'id': Outtest_id, 'rating': out}  

     

df = pd.DataFrame(dic) 

  



df.to_csv('Output2.csv',index=False) 
