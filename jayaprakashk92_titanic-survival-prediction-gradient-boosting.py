%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))
########### setting style ##########

sns.set(style="ticks", color_codes=True)
################ data extarction #################

gender = pd.read_csv('../input/gender_submission.csv')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
############### vizualization ################

sns.stripplot(x="Pclass", y="Age",hue="Survived", data=train, jitter=True);
kws = dict(s=50, linewidth=.5, edgecolor="w")



g = sns.FacetGrid(train, col="Pclass", hue="Sex",hue_order=["male", "female"],hue_kws=dict(marker=["^", "v"]))

g = (g.map(plt.scatter, "Survived","Age",**kws) .add_legend())



################# data processing ############



################# age mean calculation ##############

agemean=int(np.mean(train[train['Name'].str.contains(r'Master') & train['Age'].notnull()]['Age']))

adultagemean=int(np.mean(train[train['Name'].str.contains(r'Mr\.') & train['Name'].notnull()]['Age']))

missagemean=int(np.mean(train[train['Name'].str.contains(r'Miss\.') & (train['Parch'] <3 ) & (train['Parch'] >0 )  & train['Age'].notnull()]['Age']))

mrsagemean=int(np.mean(train[train['Name'].str.contains(r'Mrs\.') & train['Age'].notnull()]['Age']))

cmnagemean=int(np.mean(train[train['Age'].notnull()]['Age']))



train.loc[train.Name.str.match(r'.*Miss.*') & (train['Parch'] <3 ) & (train['Parch'] >0 ) & train.Age.isnull(),'Age']=int(missagemean)

train.loc[train.Name.str.match(r'.*Master.*') & train.Age.isnull(),'Age']=int(agemean)

train.loc[train.Name.str.match(r'.*Mr\..*') & train.Age.isnull(),'Age']=int(adultagemean)

train.loc[train.Name.str.match(r'.*Mrs.*') & train.Age.isnull(),'Age']=int(mrsagemean)

train.loc[train.Age.isnull(),'Age']=int(cmnagemean)



test.loc[test.Name.str.match(r'.*Miss.*') & (test['Parch'] <3 ) & (test['Parch'] >0 ) & test.Age.isnull(),'Age']=int(missagemean)

test.loc[test.Name.str.match(r'.*Master.*') & test.Age.isnull(),'Age']=int(agemean)

test.loc[test.Name.str.match(r'.*Mr\..*') & test.Age.isnull(),'Age']=int(adultagemean)

test.loc[test.Name.str.match(r'.*Mrs.*') & test.Age.isnull(),'Age']=int(mrsagemean)

test.loc[test.Age.isnull(),'Age']=int(cmnagemean)
################# agewise classification #################

train['child']=0

train['teenage']=0

train['adults']=0

train['senior citizens']=0



test['child']=0

test['teenage']=0

test['adults']=0

test['senior citizens']=0

train.loc[train['Age']<13.0 , 'child']=1

train.loc[(train['Age']>12.0) & (train['Age']<20.0) , 'teenage']=1

train.loc[(train['Age']>19) & (train['Age']<55), 'adults']=1

train.loc[(train['Age']>55), 'senior citizens']=1



test.loc[test['Age']<13.0 , 'child']=1

test.loc[(test['Age']>12.0) & (test['Age']<20.0) , 'teenage']=1

test.loc[(test['Age']>19) & (test['Age']<55), 'adults']=1

test.loc[(test['Age']>55), 'senior citizens']=1
############ trainset and testset preperation #############

trainset=train[['PassengerId','Pclass','Sex','child','teenage','adults','senior citizens','Survived']].copy()

testset=test[['PassengerId','Pclass','Sex','child','teenage','adults','senior citizens']].copy()



############ convertion non numeric columns to numeric columns ############333



lb = preprocessing.LabelBinarizer()

trainset['Sex']=lb.fit_transform(trainset['Sex'])

testset['Sex']=lb.transform(testset['Sex'])

########### model ################

model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400,max_depth=2)

model = model.fit(trainset[['Pclass','Sex','child','teenage','adults','senior citizens']],trainset['Survived'])

############# Prediction and Accuracy ################

predicted_output = model.predict(testset[['Pclass','Sex','child','teenage','adults','senior citizens']])



my_submission = pd.DataFrame({'PassengerId': testset.PassengerId, 'Survived': predicted_output})

# you could use any filename. We choose submission here

print (predicted_output)

my_submission.to_csv('submission.csv', index=False)




