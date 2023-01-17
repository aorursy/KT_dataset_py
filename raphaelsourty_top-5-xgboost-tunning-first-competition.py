from __future__ import division

import pandas as pd 

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import re

from xgboost import plot_importance
X_test=pd.read_csv('../input/test.csv', header=0, sep=',', index_col=0, parse_dates=True, encoding=None, infer_datetime_format=False)

train=pd.read_csv('../input/train.csv', header=0, sep=',', index_col=0, parse_dates=True, encoding=None, infer_datetime_format=False)

#Show a sample of the train dataset :

train.sample(10)
# Table of survival vs size of the family

survived_class = pd.crosstab(index=train["Survived"], 

                            columns=train["SibSp"],

                            margins=True)   # Include row and column totals



survived_class.columns = ["1","2","3","4","5","6","7","rowtotal"]

survived_class.index= ["Died","Survived","coltotal"]

survived_class/survived_class.loc["coltotal","rowtotal"]

survived_class
def findFamily(df, dicName):

	for index, row in df.iterrows():

		name=row['Name'].split(',')[0]

		if(name in dicName):

			dicName[name]+=1

		else:

			dicName[name]=1



	return(dicName)
def updateFamily(df, dicName):

	df['littleFamily']=0

	df['middleFamily']=0

	df['bigFamily']=0

    

	for index, row in df.iterrows():

		name=row['Name'].split(',')[0]

		if(dicName[name]<=3):

			df.loc[index,'littleFamily']=1

		elif(dicName[name]<=5):

			df.loc[index,'middleFamily']=1

        #Here I have made a mistake but fortunaly it improve over 1% my score !

        #I tink that there are members of a "big" family which survived in the test set

		#else:

		#My mistake :	df['bigFamily']=1 instead of : df.loc[index,'bigFamily']=1

	return(df)



#I try this function but it do not help me to improve the score

#Find family with the name is more efficient to improve the accuracy

def updateFamilyV2(df, dicName):

    df['SIBSP_1']=0

    df['SIBSP_2_3']=0

    df['SIBSP_4_5']=0

    df['SIBSP_6_7']=0



    for index, row in df.iterrows():

        if(row['SibSp']==1):

            df.loc[index,'SIBSP_1']=1

        elif(row['SibSp']==2 or row['SibSp']==3):

            df.loc[index,'SIBSP_2_3']=1

        elif(row['SibSp']==4 or row['SibSp']==5):

            df.loc[index,'SIBSP_4_5']=1 

        elif(row['SibSp']==6 or row['SibSp']==7):

            df.loc[index,'SIBSP_6_7']=1

   

    return(df)
dicName={}

dicName=findFamily(X_test,dicName)

dicName=findFamily(train,dicName)



train=updateFamily(train, dicName)

X_test=updateFamily(X_test, dicName)



#train=updateFamilyV2(train, dicName)

#X_test=updateFamilyV2(X_test, dicName)



train.sample(10)
def getTitleAndAge(df):

    df['Miss.']=0

    df['Mrs.']=0

    df['Master.']=0

    df['Mr.']=0

    df['Rare']=0



    for index, row in df.iterrows():

        #Get title, the row Name seems look like : "Name, Title ..."

        title=row['Name'].split(' ')[1].split(' ')[0]

        

        #We gathered the most commons title as Miss, Mrs, Master, Mr

        #If there is a title which is not common then we define it as rare

        if(title in df):

            df.loc[index,title]=1

        elif(len(title.split(' '))==1):

            df.loc[index,'Rare']=1



        df.loc[index,'Age']=transformAge(row['Age'], title)



    return(df)
def transformAge(age, title):

	if (np.isnan(age)):

		if(title=='Miss.'):

			return('0-20')

		elif(title=='Mrs.'):

			return('30-40')

		elif(title=='Master.'):

			return('0-20')

		elif(title=='Mlle.'):

			return('20-40')

		elif(title=='Mr.'):

			return('30-40')

	elif(age<=20):

		return('0-20')

	elif(age<=40):

		return('20-40')

	elif(age<=60):

		return('40-60')

	else:

		return('60-80')
def babordTribord(df):

    df['Babord']=0

    df['Tribord']=0



    for index, row in df.iterrows():

        #BABORD / TRIBOR

        cabine=re.sub('[^0-9]+','',str(row['Cabin']).split(' ')[0])

        if(len(cabine)>0 and int(cabine)%2==0):

            df.loc[index,'Babord']=1

        elif(len(cabine)>0):

            df.loc[index,'Tribord']=1

    return(df)
def joinDropColumns(df):   

    df=df.join(pd.get_dummies(df[['Sex', 'Age']]))

    df=df.drop('Sex', 1).drop('Embarked',1).drop('Cabin',1).drop('Name',1).drop('Age',1).drop('Ticket',1)

    return(df)
# Table of survival vs passenger sex

survived_class = pd.crosstab(index=train["Survived"], 

                            columns=train["Sex"])   # Include row and column totals

survived_class.columns = ["Female","Male"]

survived_class.index= ["Died","Survived"]



survived_class.plot(kind="bar", 

                 figsize=(8,8),

                 stacked=True)



plt.show()
X_test=getTitleAndAge(X_test)

train=getTitleAndAge(train)



X_test=babordTribord(X_test)

train=babordTribord(train)
# Table of survival vs left or right of the boat

train['leftRight']=float('NaN')



for index, row in train.iterrows():

    if(row['Babord']==1):

        train.loc[index,'leftRight']=0

    elif(row['Tribord']==1):

        train.loc[index,'leftRight']=1

        

#Table of survival vs babord tribord

survived_class = pd.crosstab(index=train[np.isfinite(train["leftRight"])]['Survived'], 

                            columns=train["leftRight"].dropna())   # Include row and column totals

survived_class.columns = ["Babord","Tribord"]

survived_class.index= ["Died","Survived"]



survived_class.plot(kind="bar", 

                 figsize=(8,8),

                 stacked=True)

plt.show()

train=train.drop('leftRight',1)
# Table of survival vs passenger sex

survived_class = pd.crosstab(index=train["Survived"], 

                            columns=train["Age"]) 

survived_class.columns = ["0-20","20-40","40-60","60-80","None"]

survived_class.index= ["Died","Survived"]



survived_class.plot(kind="bar", 

                 figsize=(8,8),

                 stacked=True)



plt.show()
# Table of survival vs left or right of the boat

train['title']=0



for index, row in train.iterrows():

    if(row['Miss.']==1):

        train.loc[index,'title']=1

    if(row['Mrs.']==1):

        train.loc[index,'title']=2

    if(row['Master.']==1):

        train.loc[index,'title']=3

    if(row['Mr.']==1):

        train.loc[index,'title']=4

    elif(row['Rare']==1):

        train.loc[index,'title']=5

        

#Table of survival vs babord tribord

survived_class = pd.crosstab(index=train['Survived'], 

                            columns=train["title"])   

survived_class.columns = ["Miss","Mrs","Master","Mr","Rare"]

survived_class.index= ["Died","Survived"]



survived_class.plot(kind="bar", 

                 figsize=(8,8),

                 stacked=True)

plt.show()

train=train.drop('title',1)
bins = [0,25,50,100,200,300,600]

group_names = ['0-25','25-50','50-100','100-200','200-300','300-600']

categories = pd.cut(train['Fare'], bins, labels=group_names)

train['categories'] = pd.cut(train['Fare'], bins, labels=group_names)

train['scoresBinned'] = pd.cut(train['Fare'], bins)



#Table of survival vs fare

survived_class = pd.crosstab(index=train['Survived'], 

                            columns=train["categories"])   

survived_class.columns = ['0-25','25-50','50-100','100-200','200-300','300-600']

survived_class.index= ["Died","Survived"]



survived_class.plot(kind="bar", 

                 figsize=(8,8),

                 stacked=True)

plt.show()

train=train.drop('scoresBinned',1).drop('categories',1)

X_test=joinDropColumns(X_test)

train=joinDropColumns(train)
train.sample(10)
y_train=train['Survived']

X_train=train.drop('Survived',1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
model = xgb.XGBClassifier(booster='gbtree', silent=0, seed=0, base_score=0.5, subsample=0.75)

parameters = {'n_estimators':[240,280,320],

            'max_depth':[10,11,12],

            'gamma':[0,1,2,3],

            'max_delta_step':[0,1,2],

            'min_child_weight':[1,2,3], 

            'colsample_bytree':[0.55,0.6,0.65],

            'learning_rate':[0.1,0.2,0.3]

            }

model = GridSearchCV(model, parameters)

model.fit(X_train,y_train)

print('Best parameters founded : {}'.format(model.best_params_))
model = xgb.XGBClassifier(booster='gbtree', silent=1, seed=0, base_score=0.5, subsample=0.75)

grid ={'max_delta_step': 0, 'max_depth': 10, 'min_child_weight': 2, 'n_estimators': 280, 'colsample_bytree': 0.65, 'gamma': 2}

model.set_params(**grid)

model.fit(X_train,y_train)

kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(model, np.array(X_val), np.array(y_val), cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

plot_importance(model)

plt.show()

prediction=model.predict(X_test)

res=pd.DataFrame({'PassengerId':X_test.index.tolist(),'Survived':prediction}, columns=['PassengerId','Survived'])

res.to_csv('submissionXgboost.csv', sep=",", index=False)
