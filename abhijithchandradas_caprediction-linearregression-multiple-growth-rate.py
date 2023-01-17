import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LinearRegression 



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

model_2.score(x2,y2)
test.head()
#Making Id as unique key between test and train

test["Id"]=50+test.ForecastId

test.head()
test["LogConf"]=model_1.predict(np.array(test.Id).reshape(-1,1))

test["ConfirmedCases"]=np.exp(test.LogConf)//1

test["Fatalities"]=model_2.predict(np.array(test.ConfirmedCases).reshape(-1,1))//1
#Wherever confirmed cases and fatalities are available in train data, update it into test data

for id in train.Id:

    test.ConfirmedCases[test.Id==id]=train.ConfirmedCases[train.Id==id].sum()

    test.Fatalities[test.Id==id]=train.Fatalities[train.Id==id].sum()
test["Conf_d1"]=test.ConfirmedCases

test["Fat_d1"]=test.Fatalities



rate=gr-1

for i in range(train.shape[0]-2,test.shape[0]):

    rate*=0.99

    test.Conf_d1[i]=(1+rate)*test.Conf_d1[i-1]//1

    test.Fat_d1[i]=model_2.predict(np.array(test.Conf_d1[i]).reshape(-1,1))[0]//1
test["Conf_d5"]=test.ConfirmedCases

test["Fat_d5"]=test.Fatalities



rate=gr-1

for i in range(train.shape[0]-2,test.shape[0]):

    rate*=0.95

    test.Conf_d5[i]=(1+rate)*test.Conf_d5[i-1]//1

    test.Fat_d5[i]=model_2.predict(np.array(test.Conf_d5[i]).reshape(-1,1))[0]//1
test["Conf_i1"]=test.ConfirmedCases

test["Fat_i1"]=test.Fatalities



rate=gr-1

for i in range(train.shape[0]-2,test.shape[0]):

    rate*=1.01

    test.Conf_i1[i]=(1+rate)*test.Conf_i1[i-1]//1

    test.Fat_i1[i]=model_2.predict(np.array(test.Conf_i1[i]).reshape(-1,1))[0]//1
test["Conf_15"]=test.ConfirmedCases

test["Fat_15"]=test.Fatalities



rate=1.15

for i in range(train.shape[0]-2,test.shape[0]):

    test.Conf_15[i]=rate*test.Conf_15[i-1]//1

    test.Fat_15[i]=model_2.predict(np.array(test.Conf_15[i]).reshape(-1,1))[0]//1
test["Conf_20"]=test.ConfirmedCases

test["Fat_20"]=test.Fatalities



rate=1.20

for i in range(train.shape[0]-2,test.shape[0]):

    test.Conf_20[i]=rate*test.Conf_20[i-1]//1

    test.Fat_20[i]=model_2.predict(np.array(test.Conf_20[i]).reshape(-1,1))[0]//1
test["Conf_25"]=test.ConfirmedCases

test["Fat_25"]=test.Fatalities



rate=1.25

for i in range(train.shape[0]-2,test.shape[0]):

    test.Conf_25[i]=rate*test.Conf_25[i-1]//1

    test.Fat_25[i]=model_2.predict(np.array(test.Conf_25[i]).reshape(-1,1))[0]//1
test["Conf_30"]=test.ConfirmedCases

test["Fat_30"]=test.Fatalities



rate=1.30

for i in range(train.shape[0]-2,test.shape[0]):

    test.Conf_30[i]=rate*test.Conf_30[i-1]//1

    test.Fat_30[i]=model_2.predict(np.array(test.Conf_30[i]).reshape(-1,1))[0]//1
plt.figure(figsize=(12,8))

sns.lineplot(test.Id, test.Conf_20, label="Constant 20%")

sns.lineplot(test.Id, test.Conf_25, label="Constant 25%")

sns.lineplot(test.Id, test.Conf_30, label="Constant 30%", dashes=True)

sns.lineplot(test.Id, test.Conf_15, label="Constant 15%")

sns.lineplot(test.Id, test.Conf_i1, label="Increasing by 1%")

sns.lineplot(test.Id, test.Conf_d1, label="Decreasing by 1%")

sns.lineplot(test.Id, test.Conf_d5, label="Decreasing by 5%")

sns.lineplot(test.Id, test.ConfirmedCases, label="Current Rate")

sns.lineplot(train.Id, train.ConfirmedCases, label="Train Data")

plt.legend()

plt.show()
plt.figure(figsize=(12,8))

sns.lineplot(test.Id, test.Fat_25, label="Constant 25%")

sns.lineplot(test.Id, test.Fat_20, label="Constant 20%")

sns.lineplot(test.Id, test.Fat_30, label="Constant 30%")

sns.lineplot(test.Id, test.Fat_15, label="Constant 15%")

sns.lineplot(test.Id, test.Fat_i1, label="Increasing by 1%")

sns.lineplot(test.Id, test.Fat_d1, label="Decreasing by 1%")

sns.lineplot(test.Id, test.Fat_d5, label="Decreasing by 5%")

sns.lineplot(test.Id, test.Fatalities, label="Current Rate")

sns.lineplot(train.Id, train.Fatalities, label="Train Data")

plt.legend()

plt.show()
sub.ConfirmedCases=test.ConfirmedCases

sub.Fatalities=test.Fatalities

sub.to_csv("submission.csv", index=False)