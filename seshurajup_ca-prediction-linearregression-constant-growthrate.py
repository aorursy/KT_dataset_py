import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



from sklearn.linear_model import LinearRegression



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sub=pd.read_csv("../input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv")

test=pd.read_csv("../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv")

train=pd.read_csv("../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv")
print(train.shape)

train.head()
#Only taking data with confirmed cases

train=train[train.ConfirmedCases>0]

print(train.shape)

train.head()
sns.lineplot(train.Id, train.ConfirmedCases)
sns.regplot(train.Id, np.log(train.ConfirmedCases))
model_1= LinearRegression()

x1=np.array(train.Id).reshape(-1,1)

y1=np.log(train.ConfirmedCases)

model_1.fit(x1,y1)

print("R-squared score : ",model_1.score(x1,y1))



gr=np.power(np.e, model_1.coef_[0])

print("Growth Factor : ", gr)

print(f"Growth Rate : {round((gr-1)*100,2)}%")
sns.regplot(train.ConfirmedCases,train.Fatalities)
model_2= LinearRegression()

x2=np.array(train.ConfirmedCases).reshape(-1,1)

y2=train.Fatalities

model_2.fit(x2,y2)

print("R-Squared Score= ",model_2.score(x2,y2))
test.head()
#Making Id as unique key between test and train

test["Id"]=50+test.ForecastId

test.head()
test["LogConf"]=model_1.predict(np.array(test.Id).reshape(-1,1))

test["ConfirmedCases"]=np.exp(test.LogConf)//1

test["Fatalities"]=model_2.predict(np.array(test.ConfirmedCases).reshape(-1,1))//1

test
#Wherever confirmed cases and fatalities are available in train data, update it into test data

for id in train.Id:

    test.ConfirmedCases[test.Id==id]=train.ConfirmedCases[train.Id==id].sum()

    test.Fatalities[test.Id==id]=train.Fatalities[train.Id==id].sum()

test
### Prepare submission file
sub.ConfirmedCases=test.ConfirmedCases

sub.Fatalities=test.Fatalities

sub.to_csv("submission.csv", index=False)