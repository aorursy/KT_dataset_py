# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame, Series

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print(train.info())

print("---------------------------------------")

print(test.info())
#Removing NaN values and features

train = train.drop(["Ticket", "Cabin"], axis=1)

test = test.drop(["Ticket", "Cabin"], axis=1)



train = train.dropna()

train.info()
#Visualizations



fig = plt.figure(figsize=(12,8), dpi=1000)

alpha_scatterplot = 0.2

alpha_bar_chart=0.55



ax1 = plt.subplot2grid((2,3), (0,0))

train.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)

plt.grid(b=True, axis='y')

ax1.set_xlim(-1,2)

plt.title("Distribution of Survival, (1 = Survived)")



plt.subplot2grid((2,3),(0,1))

plt.scatter(train.Survived, train.Age)

plt.ylabel("Age")

plt.xlabel("Survived")

plt.grid(b=True, which='major', axis='y')

plt.title("Survival by Age (1 =  Survived)")



plt.subplot2grid((2,3),(0,2))

train.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)

plt.ylabel("Pclass")

plt.title("Class Distribution")

plt.grid(b=True,axis='x')



plt.subplot2grid((2,3),(1,0), colspan=2)

train.Age[train.Pclass==1].plot(kind="kde")

train.Age[train.Pclass==2].plot(kind="kde")

train.Age[train.Pclass==3].plot(kind="kde")

plt.title("Age Distribution by Class")

plt.legend(['Class1','Class2','Class3'], loc='best')

plt.xlabel("Age")



plt.subplot2grid((2,3),(1,2))

train.Embarked.value_counts().plot(kind="bar", alpha=alpha_bar_chart)

plt.xlabel("Embarked")

plt.grid(b=True,axis='y')

plt.title("Distribution of Embarked")
plt.figure(figsize=(6,4))

#fig, ax=plt.subplots()

train.Survived.value_counts().plot(kind='barh', alpha=alpha_bar_chart)

plt.grid(b=True,axis='x')
fig = plt.figure(figsize=(18,4))

train_male = train.Survived[train.Sex=='male'].value_counts().sort_index()

train_female = train.Survived[train.Sex=='female'].value_counts().sort_index()



#ax1 = plt.subplot2grid((1,2),(0,0))

ax1 = fig.add_subplot(121)

train_male.plot(kind="barh",alpha=alpha_bar_chart, label='Male', color="blue")

train_female.plot(kind="barh",alpha=alpha_bar_chart, label="Female", color='red')

plt.title("Who Survived? with respect to Gender, (raw value counts) "); plt.legend(loc='best')

ax1.set_ylim(-1, 2) 

plt.grid(b=True)



#ax2 = plt.subplot2grid((1,2),(0,1))

ax2 = fig.add_subplot(122)

(train_male/float(train_male.sum())).plot(kind="barh",alpha=alpha_bar_chart, label='Male', color="blue")

(train_female/float(train_female.sum())).plot(kind="barh",alpha=alpha_bar_chart, label="Female", color='red')

plt.title("Who Survived? with respect to Gender, (proportionally) "); plt.legend(loc='best')

ax2.set_ylim(-1, 2)

ax2.set_xlim(0,1)

plt.xticks(np.arange(0,1,0.1))

plt.grid(b=True, which='Major')
fig = plt.figure(figsize=(9,7),dpi=1600)



train_l_male = train.Survived[train.Sex=='male'][train.Pclass == 3].value_counts().sort_index()

train_l_female = train.Survived[train.Sex=='female'][train.Pclass == 3].value_counts().sort_index()

train_h_male = train.Survived[train.Sex=='male'][train.Pclass != 3].value_counts().sort_index()

train_h_female = train.Survived[train.Sex=='female'][train.Pclass != 3].value_counts().sort_index()



ax1 = fig.add_subplot(221)

train_l_male.plot(kind="bar", alpha=0.50, color ='blue')

plt.grid(True,axis='y')

ax1.set_xticklabels(["Died", "Survived"], rotation=0)

plt.title("Low Class Males Survival")

ax1.set_ylim(0,250)



ax2 = fig.add_subplot(222)

train_h_male.plot(kind="bar", alpha=0.50, color ='blue')

plt.grid(True,axis='y')

ax2.set_xticklabels(["Died", "Survived"], rotation=0)

plt.title("High Class Males Survival")

ax2.set_ylim(0,250)



ax3 = fig.add_subplot(223)

train_l_female.plot(kind="bar", alpha=0.50, color ='pink')

plt.grid(True,axis='y')

ax3.set_xticklabels(["Died", "Survived"], rotation=0)

plt.title("Low Class Females Survival")

ax3.set_ylim(0,250)



ax4 = fig.add_subplot(224)

train_h_female.plot(kind="bar", alpha=0.50, color ='pink')

plt.grid(True,axis='y')

ax4.set_xticklabels(["Died", "Survived"], rotation=0)

plt.title("High Class Females Survival")

ax4.set_ylim(0,250)
import statsmodels.api as sm

from patsy import dmatrices



formula = "Survived ~ C(Pclass) + C(Sex) + Age + SibSp + C(Embarked)"

results={}



#Separate the columns based on the data

y, x = dmatrices(formula, data = train, return_type='dataframe' )



#Instantiate the model

model = sm.Logit(y,x)



#Train the model

res = model.fit()

results["Logit"]=[res, formula]

res.summary()
#Plot predictions vs actual

plt.figure(figsize=(10,4))

ax1=plt.subplot(121, axisbg="#DBDBDB")

ypred=res.predict(x)

plt.plot(x.index, ypred, 'bo', x.index, y, 'mo', alpha=.25);

plt.grid(color='white', linestyle='dashed')

plt.title('Logit predictions, Blue: \nFitted/predicted values: Red');



ax2=plt.subplot(122, axisbg="#DBDBDB")

plt.plot(res.resid_dev, 'r-')

plt.grid(color='white', linestyle='dashed')

ax2.set_xlim(-1, len(res.resid_dev))

plt.title('Logit Residuals');