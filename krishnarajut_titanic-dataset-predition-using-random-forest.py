# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier



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



#test['Survied']=gender['Survived'].copy()

combinedDF=pd.merge(test, gender, on='PassengerId')

#test.head()

train.head()
train['Embarked'].unique()
test.head()
############### vizualization ################

#sns.pairplot(data=train,hue="Survived",markers=["o", "s"])

ax = sns.barplot(x="Survived", y="Pclass", data=train)

bx = sns.barplot(x="Pclass", y="Survived", data=train)



sns.stripplot(x="Pclass", y="Age",hue="Survived", data=train, jitter=True);



#cx=sns.scatterplot(x="Pclass", y="Survived",hue='Sex',data=train)

#g = sns.FacetGrid(train, row="Sex", col="Pclass")

#g.map(plt.hist, "Survived", color="steelblue", bins=train)



#g = sns.FacetGrid(tips, col="smoker", col_order=["Yes", "No"])

#g = g.map(plt.hist, "total_bill", bins=bins, color="m")

kws = dict(s=50, linewidth=.5, edgecolor="w")

g = sns.FacetGrid(train, col="Pclass", hue="Sex",hue_order=["male", "female"],hue_kws=dict(marker=["^", "v"]))

g = (g.map(plt.scatter, "Survived","SibSp",**kws) .add_legend())



g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,size=6, kind="bar", palette="muted")

g = sns.factorplot(x="Sex", y="Pclass", hue="Survived", data=train,size=6, kind="bar", palette="muted")

################# data processing ############



agemean=int(np.mean(train[train['Name'].str.contains(r'Master') & train['Age'].notnull()]['Age']))

adultagemean=int(np.mean(train[train['Name'].str.contains(r'Mr\.') & train['Name'].notnull()]['Age']))

missagemean=int(np.mean(train[train['Name'].str.contains(r'Miss\.') & (train['Parch'] <3 ) & (train['Parch'] >0 )  & train['Age'].notnull()]['Age']))

mrsagemean=int(np.mean(train[train['Name'].str.contains(r'Mrs\.') & train['Age'].notnull()]['Age']))

cmnagemean=int(np.mean(train[train['Age'].notnull()]['Age']))





train.loc[train.Name.str.match(r'.*Miss.*') & (train['Parch'] <3 ) & (train['Parch'] >0 ) 

          & train.Age.isnull(),'Age']=int(missagemean)

train.loc[train.Name.str.match(r'.*Master.*') & train.Age.isnull(),'Age']=int(agemean)

train.loc[train.Name.str.match(r'.*Mr\..*') & train.Age.isnull(),'Age']=int(adultagemean)

train.loc[train.Name.str.match(r'.*Mrs.*') & train.Age.isnull(),'Age']=int(mrsagemean)

train.loc[train.Age.isnull(),'Age']=int(cmnagemean)



combinedDF.loc[combinedDF.Name.str.match(r'.*Miss.*') & (combinedDF['Parch'] <3 ) & (combinedDF['Parch'] >0 ) & combinedDF.Age.isnull(),'Age']=int(missagemean)

combinedDF.loc[combinedDF.Name.str.match(r'.*Master.*') & combinedDF.Age.isnull(),'Age']=int(agemean)

combinedDF.loc[combinedDF.Name.str.match(r'.*Mr\..*') & combinedDF.Age.isnull(),'Age']=int(adultagemean)

combinedDF.loc[combinedDF.Name.str.match(r'.*Mrs.*') & combinedDF.Age.isnull(),'Age']=int(mrsagemean)

combinedDF.loc[combinedDF.Age.isnull(),'Age']=int(cmnagemean)



train['agecateg']=""



train.loc[train['Age']<13.0 , 'agecateg']='child'

train.loc[(train['Age']>12.0) & (train['Age']<20.0) , 'agecateg']='teenage'

train.loc[(train['Age']>19) & (train['Age']<55), 'agecateg']='adults'

train.loc[(train['Age']>55), 'agecateg']='senior citizens'



combinedDF.loc[combinedDF['Age']<13.0 , 'agecateg']='child'

combinedDF.loc[(combinedDF['Age']>12.0) & (combinedDF['Age']<20.0) , 'agecateg']='teenage'

combinedDF.loc[(combinedDF['Age']>19) & (combinedDF['Age']<55), 'agecateg']='adults'

combinedDF.loc[(combinedDF['Age']>55), 'agecateg']='senior citizens'





trainset=train[['Name','Pclass','Sex','agecateg','Survived']].copy()

combinedDFset=combinedDF[['Name','Pclass','Sex','agecateg','PassengerId']].copy()



lb = preprocessing.LabelBinarizer()

trainset['Sex']=lb.fit_transform(trainset['Sex'])

combinedDFset['Sex']=lb.transform(combinedDFset['Sex'])





model = RandomForestClassifier(n_estimators=100,max_depth=2)

model = model.fit(trainset.iloc[:,1:3],trainset.iloc[:,4])





output = model.predict(combinedDFset.iloc[:,1:3])

outputDf = pd.DataFrame(output,columns=['Survived'])

outputDf['PassengerId'] = combinedDFset['PassengerId'].copy()

outputDf = outputDf[['PassengerId','Survived']]

print (outputDf)

print (accuracy_score(output,combinedDF.iloc[:,-2])*100)



# Any results you write to the current directory are saved as output.
