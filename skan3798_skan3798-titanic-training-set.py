# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%pylab inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.columns
df.Survived.value_counts()
###PClass vs. Survived ###
df.Pclass.value_counts().plot(kind='bar')
df[df.Pclass.isnull()]
df.Pclass.hist()
#since unable to compare string to series values, convert Pclass to str
df['Pclass'] = df['Pclass'].astype(str)
#now compare Pclass 3 to survival rate
df[df.Pclass=="3"].Survived.value_counts()
119/(372+119)
#comparing Pclass 2 to survival rate
df[df.Pclass=="2"].Survived.value_counts()
87/(97+87)
#comparing Pclass 1 to survival rate
df[df.Pclass=="1"].Survived.value_counts()
136/(136+80)
###Sex vs. Survival###
df.Sex.value_counts().plot(kind='bar')
df[df.Sex.isnull()] #no missing values
#comparing Male to survival rate
df[df.Sex=="male"].Survived.value_counts()
109/(468+109)
#comparing Female to survival rate
df[df.Sex=="female"].Survived.value_counts()
233/(233+81)
###Age vs. Survival ###
df.Age.value_counts().plot(kind='bar')
df[df.Age.isnull()] #there is missing data for age
df[df.Age.isnull()].Survived.value_counts()
125/(52+125)
#test, filling the null Ages with average age and checking survival rate impact
avgAge = df.Age.mean()
print(avgAge)
df.Age = df.Age.fillna(value=avgAge)
df[df.Age.isnull()] #check that there are now no null ages
df   #from looking at the data table, we see any null ages are changed to 	29.6991176471
df[df.Age <"15"].Survived.value_counts()
19/(19+11)
df[(df.Age >= "15") & (df.Age <"29.6991176471")].Survived.value_counts()
114/(114+202+114)
df[(df.Age <= "50") & (df.Age >"29.6991176471")].Survived.value_counts()
123/(123+153)
df[(df.Age > "50") & (df.Age <"70")].Survived.value_counts()
29/(44+29)
df[df.Age >= "70"].Survived.value_counts()
5/(5+14)
df[df.Age =="29.6991176471"].Survived.value_counts()
52/(125+52)
###Fare vs. Survival###
df.Fare.value_counts().plot(kind='barh')
df[df.Fare.isnull()] #no null entries for fare :)
df[df.Fare==0].Survived.value_counts()
1/15
df[(df.Fare >0)&(df.Fare <10)].Survived.value_counts()
66/(255+66)
df[(df.Fare >=10)&(df.Fare <20)].Survived.value_counts()
76/(103+76)
df[(df.Fare >=20)&(df.Fare <30)].Survived.value_counts()
58/(58+78)
df[(df.Fare >=30)&(df.Fare <40)].Survived.value_counts()
28/(36+28)
df[(df.Fare >=40)&(df.Fare <50)].Survived.value_counts()
4/15
df[(df.Fare >=50)&(df.Fare <60)].Survived.value_counts()
27/(27+12)
df[(df.Fare >=60)&(df.Fare <70)].Survived.value_counts()
6/17
df[(df.Fare >=70)].Survived.value_counts()
76/(76+29)
df[(df.Fare >=60)&(df.Fare < 70)].Sex.value_counts()
df[(df.Age > "70")].Sex.value_counts()
df[(df.Age <= "50") & (df.Age >"29.6991176471")].Sex.value_counts()
df[df.Pclass=='1'].Sex.value_counts()
df[df.Pclass=='2'].Sex.value_counts()
df[(df.Pclass=='3')&(df.Sex=='female')].Survived.value_counts()
df[(df.Pclass=='2')&(df.Sex=='female')].Survived.value_counts()
df[(df.Pclass=='1')&(df.Sex=='female')].Survived.value_counts()
df[(df.Age >='50')&(df.Age <'70')&(df.Sex=='female')].Survived.value_counts()
df[(df.Age >'29.6991176471')&(df.Age <'50')&(df.Sex=='female')].Survived.value_counts()
df[(df.Age =='29.6991176471')&(df.Sex=='female')].Survived.value_counts()
df[(df.Age <'15')& (df.Sex=='female')].Survived.value_counts()
#predicting using TEST.CSV
df_test = pd.read_csv('../input/test.csv')
df_test.columns
df_test.insert(2,'Survived',0)

df_test.columns
df_test.loc[(df_test.Sex == 'female')&((df_test.Pclass == 1)|(df_test.Pclass == 2)|((df_test.Age >= 50)&(df_test.Age < 70)))]
df_test.loc[((df_test.Sex == 'female')&((df_test.Pclass == 1)|(df_test.Pclass == 2)|((df_test.Age >= 50)&(df_test.Age < 70)))),'Survived'] = 1
df_test.loc[(df_test.Sex == 'female')&((df_test.Pclass == 1)|(df_test.Pclass == 2)|((df_test.Age >= 50)&(df_test.Age < 70)))]
submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": df_test["Survived"]
})
submission.to_csv('titanic_submission.csv',index=False)
print