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



import math

import random

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from collections import Counter

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR

from scipy import stats

import scipy.stats as stats

import pymc3 as pm

import arviz as az

from sklearn.utils import shuffle

from sklearn.model_selection import cross_val_score





from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_df.info()
new_train_df = train_df.drop(columns=['Survived'], axis=1)

total_df = pd.concat([new_train_df,test_df], sort=False, ignore_index=True)

total_df.isnull().sum()
total_df.head()
total_df['Title'] = total_df.Name.str.extract('([A-Za-z]+)\.',expand=False)

pd.crosstab(total_df['Title'],total_df['Sex'])
list1 = ['Master','Mr']

list2 = ['Miss','Mrs','Ms','Mlle']



def AgeHist1(list1,list2,dataset):

    fig = plt.figure(figsize=(10,4))

    title1 = []

    title2 = []

    

    ax1 = fig.add_subplot(121)

    for i in np.arange(len(list1)):

        ax1 = sns.distplot(dataset['Age'].loc[dataset['Title']==list1[i]].dropna(),kde=False)

        ax1.set_ylabel('Counts')

        ax1.set_xlabel('Age')

        title1.append(list1[i])

    ax1.legend(labels=title1,loc='upper right',fontsize='small')

    ax2 = fig.add_subplot(122)

    for i in np.arange(len(list2)):

        ax2 = sns.distplot(dataset['Age'].loc[dataset['Title']==list2[i]].dropna(),kde=False)

        ax2.set_ylabel('Counts')

        ax2.set_xlabel('Age')

        title2.append(list2[i])

    ax2.legend(labels=title2,loc='upper right',fontsize='small')

        

AgeHist1(list1,list2,total_df)
total_df['Title'] = total_df.Name.str.extract('([A-Za-z]+)\.',expand=False)

total_df['Title'] = total_df['Title'].replace(['Rev','Dr','Sir','Major','Countess'],'Special')

total_df['Title'] = total_df['Title'].replace(['Mlle','Miss','Lady'],'Miss')

total_df['Title'] = total_df['Title'].replace(['Mme','Ms','Mrs'],'Mrs')

total_df['Title'] = total_df['Title'].replace(['Col','Don','Dona','Jonkheer','Capt'],'The others')



pd.crosstab(total_df['Title'],total_df['Sex'])
pd.crosstab(total_df['Title'],total_df['Age'].isna())
list = ['Master','Miss','Mr','Mrs']



def AgeHist2(list,dataset):

    fig = plt.figure(figsize=(10,6))

    

    for i in np.arange(len(list)):

        

        plt.subplot(math.ceil(len(list)/2),2,i+1)

        ax = sns.distplot(dataset['Age'].loc[dataset['Title']==list[i]].dropna(),kde=False)

        median = dataset['Age'].loc[dataset['Title']==list[i]].dropna().median()

        mean = dataset['Age'].loc[dataset['Title']==list[i]].dropna().mean()

        ax.axvline(mean, color='r', linestyle='--')

        ax.axvline(median, color='g', linestyle='--')

        ax.set_title('{}'.format(list[i]))

        plt.legend(labels=['mean','median'],loc='upper right',fontsize='small')

        plt.subplots_adjust(wspace=0.5, hspace=0.8)

        

AgeHist2(list,total_df)
def AgeBayesPredictor(title,dataset):

    

    missing = dataset['Age'].loc[dataset['Title']== title].isnull().sum()

    

    def AgeExtractor(title,dataset):

    

        Age = dataset['Age'].loc[dataset['Title']==title]

        Age = Age.dropna()

        

        return Age

    

    Age = AgeExtractor(title,dataset)

    



    with pm.Model() as model:

        

        upper = max(Age)-(1/4)*(max(Age)-min(Age))

        lower = min(Age)+(1/4)*(max(Age)-min(Age))

    

        #Set the prior

        mu = pm.Uniform('mu',upper = upper ,lower= lower)

        sigma = pm.HalfNormal('sigma',sd=10)

        

    

        #Liklihood

        observed = pm.Gamma('obs', mu=mu,sigma=sigma,observed=Age)

        

        

    with model:

        

        start = pm.find_MAP()

        

        #Trace

        trace = pm.sample(8000, start=start)

    

    #Sampling

    sampling = pm.sample_ppc(trace[1000:], model=model,samples=missing)

    sampling=[random.choice(sampling['obs'][i]) for i in np.arange(start=0, stop=missing)]

    return sampling



def imputeAge(title,dataset):

    

    for i in title:

        

        imputing_Age = AgeBayesPredictor(i,dataset)

        idx = dataset['Age'].loc[dataset['Title']==i].isnull()

        missing = dataset['Age'].loc[dataset['Title']==i][idx]

        

        for j in np.arange(len(missing)):

            missing.iloc[j] = imputing_Age[j]



        dataset.update(missing)

    return dataset



total_df = imputeAge(['Master','Miss','Mr','Mrs'],total_df)
list = ['Master','Miss','Mr','Mrs']

AgeHist2(list,total_df)
train_Cabin_df = train_df.dropna(subset=["Cabin"])

#Cabin grouped by Initial Alphabats

def CategorizeCabin(data):

    

    for i in ['A','B','C','D','E','F','G','T']:

        Index = data["Cabin"].str.find(i)==0

        

        data["Cabin"][Index] = i

    

    return data



train_Cabin_df=CategorizeCabin(train_Cabin_df)

total_df=CategorizeCabin(total_df)

train_Cabin_df["Cabin"].unique()



#Update train data

train_df.update(train_Cabin_df)
fig = plt.plot()

sns.countplot(data=train_Cabin_df, x = "Pclass",hue ="Cabin")

plt.show()
fig = plt.figure

sns.countplot(data=train_Cabin_df, x ="Cabin", hue="Survived")

plt.show()

sns.barplot( x ="Cabin", y="Pclass",data=train_Cabin_df)

plt.show()

sns.countplot(data=train_Cabin_df, x ="Pclass", hue="Survived")

plt.show()

#sns.countplot(data=train_df, x ="Pclass", hue="Survived")

#plt.show()
corr = total_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)]= True



cmap = sns.diverging_palette(220,10,as_cmap = True)





sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#Fill missing NaN

total_df['Fare'] = total_df['Fare'].fillna(total_df['Fare'][total_df['Pclass']==3].mean())

total_df['Age'] = total_df['Age'].fillna(total_df['Age'][total_df['Title']=='Mr'].mean())

total_df['Embarked'] = total_df['Embarked'].fillna('S')
total_df.isnull().sum()
bins = np.linspace(min(total_df['Age'])-1,max(total_df['Age'])+1,num=6)

labels = ['Kid','Young Adult','Adult','Older Adult','Senior']

total_df['AgeGroup'] = pd.cut(total_df['Age'], bins=bins, labels=labels, right=False)
pd.crosstab(total_df['AgeGroup'],total_df['Title'])
features_drop = ['PassengerId','Name', 'Ticket', 'Parch','Cabin','Age','Fare']

features = total_df.drop(features_drop, axis=1)

features.head()
#one-hot encoding

features = pd.get_dummies(features)

#separate train and label

train_label = train_df['Survived']

train_data = features.head(len(train_df))

test_data = features.tail(len(test_df))

train_data.head()
train_data, train_label = shuffle(train_data, train_label, random_state = 5)
def train_and_test(model):

    model.fit(train_data, train_label)

    prediction = model.predict(test_data)

    accuracy = round(model.score(train_data, train_label) * 100, 2)

    print("Accuracy : ", accuracy, "%")

    return prediction
# Logistic Regression

log_pred = train_and_test(LogisticRegression())

# SVM

svm_pred = train_and_test(SVC())

#kNN

knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors = 4))

# Random Forest

rf_pred = train_and_test(RandomForestClassifier(n_estimators=50))

# Navie Bayes

nb_pred = train_and_test(GaussianNB())
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission2 = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':rf_pred.astype(int)})



#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Titanic Predictions2.csv'



submission2.to_csv(filename,index=False)



print('Saved file: ' + filename)
cverror = []

trerror = []

for i in np.arange(5, 105, 5):

    clf = RandomForestClassifier(n_estimators=i)

    clf.fit(train_data, train_label)

    error1 = cross_val_score(clf,train_data,train_label, cv=5).mean()

    error2 = clf.score(train_data, train_label)

    cverror.append(1-error1)

    trerror.append(1-error2)

cverror = pd.DataFrame(cverror)

cverror.columns = ["cv-error"]

cverror["train-error"] = trerror

ax1=sns.lineplot(data=cverror)

ax1.set_title("5-fold cross validation")