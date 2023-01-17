# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
salaries=pd.read_csv("/kaggle/input/nba-salaries/salaries_1985to2018.csv")

players=pd.read_csv("/kaggle/input/nba-salaries/players.csv")

salaries.head()

players.head()
fulldata=pd.merge(salaries,players, left_on='player_id',right_on='_id')

fulldata.shape
fulldata.head()
fulldata.isnull().sum()
after2k=fulldata[fulldata["draft_year"].fillna(0).astype(int)>=2000]

after2k.shape
after2k.columns
sortedbysalary=after2k.sort_values(by="salary",ascending=False)

sortedbysalary[["salary","name","draft_pick"]][0:20]
sortedbysalary[["salary","name","career_WS"]][0:20]
after2k["career_WS"]=after2k["career_WS"].astype(float)

sortedbywinsh=after2k.sort_values(by="career_WS",ascending=False)

sortedbywinsh[["salary","name","career_WS"]][0:20]
set(sortedbywinsh["name"][0:120])
after2k[after2k["name"]=="Hedo Turkoglu"][["name","salary","season"]]
after2k[after2k["name"]=="Mehmet Okur"][["name","salary","season"]]
total_earnings=after2k.groupby("name")["salary"].sum()

total_earnings.sort_values(ascending=False)[0:25]
total_earnings.sort_values(ascending=False)[0:30]
total_earnings.to_csv("totalearningsnba2000s.csv")
total_earnings.sort_values(ascending=False)[30:60]
total_earnings_df=pd.DataFrame({"name":list(total_earnings.index),

              "salaries":total_earnings.get_values()

             })
tempo=after2k[["name","career_PER","career_PTS","career_AST","career_TRB"]]
newdata=pd.merge(total_earnings_df,tempo,on="name",how='left')
newdata=newdata.drop_duplicates()
newdata
y=newdata["salaries"]

x=newdata[["career_PER","career_PTS","career_AST","career_TRB"]]
def isnumber(x):

    try:

        float(x)

        return True

    except:

        return False



x=x[x.applymap(isnumber)]

x=x.fillna(0)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
from sklearn.linear_model import LinearRegression

import sklearn.metrics as metrik





linreg=LinearRegression()

linreg.fit(x_train,y_train)

ypred=linreg.predict(x_test)

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)
import seaborn as sns

import matplotlib.pyplot as plt

newdata["career_TRB"]=newdata["career_TRB"].astype(float)

correlation=newdata.corr()

sns.heatmap(correlation,annot=True)
from tpot import TPOTRegressor

tpot = TPOTRegressor(verbosity=2,max_time_mins=229)

tpot.fit(x_train, y_train)

tpot.score(x_test,y_test)
ypred=tpot.predict(x_test)

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)
newdata.to_csv("lastdata.csv")
newdata.head()
sadeceper=newdata["career_PER"]

sadeceper=pd.DataFrame(sadeceper)

sadeceper=sadeceper[sadeceper.applymap(isnumber)]

sadeceper=sadeceper.fillna(0)

newdata["career_PER"]=sadeceper.astype(float)

sortedbyper=newdata.sort_values(by="career_PER",ascending=False)

sortedbyper[0:30]
sortedbyper[30:60]
sortedbyper[60:120]
print(sortedbyper[sortedbyper["name"]=="Hedo Turkoglu"])

print(sortedbyper[sortedbyper["name"]=="Mehmet Okur"])

print(sortedbyper[sortedbyper["name"]=="Ersan Ilyasova"])
sortedbyper.to_csv("playerefficencyrating.csv")