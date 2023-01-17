import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import datasets, linear_model

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
global test_df

global train



train = pd.DataFrame.from_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
def theview(train):

    for i in train:

        print ('\n' + i + ': ' + str(train[i].isnull().sum()) + ' nulls')

        print(str(len(train[i].unique())) + ' uniques' )

        #print(str(len(train[i].unique()))

        if len(train[i].unique()) <10:

            print (train[i].unique())

    print ('\n' + str(train.shape))
#Cleanup



#Cabin



train = pd.concat( [train , pd.get_dummies(train['Cabin'].str[:1],dummy_na=False)] , axis=1)

test_df = pd.concat( [test_df , pd.get_dummies(test_df['Cabin'].str[:1],dummy_na=False)] , axis=1)



for i in train.columns:

    if len(str(i))<2:

        train.rename(columns={i: 'Cabin_'+i}, inplace=True)

for i in test_df.columns:

    if len(str(i))<2:

        test_df.rename(columns={i: 'Cabin_'+i}, inplace=True)   



'''

def age_impute(train, test_df):

    for i in [train, test_df]:

        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)

        data = train.groupby(['Pclass'])['Age']

        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))

    return train, test_df



train, test_df = age_impute(train, test_df)        

'''



#Deleting Cabin_T since doesn't exist in test.csv

del train['Cabin_T']        





###Name Adjustments



def names(train, test_df):

    for i in [train, test_df]:

        i['Name_Len'] = i['Name'].apply(lambda x: len(x))

        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

        del i['Name']

    return train, test_df



train, test_df = names(train, test_df)

train = pd.concat( [train , pd.get_dummies(train['Name_Title'])] , axis=1)

test_df = pd.concat( [test_df, pd.get_dummies(test_df['Name_Title'])] , axis=1)







for i in train.columns[1:].tolist():

    if i not in test_df.columns[0:].tolist():

        print(i)

        del train[i]



del test_df['Dona.']







#Embarked

train = pd.concat( [train , pd.get_dummies(train['Embarked'])] , axis=1)

test_df = pd.concat( [test_df, pd.get_dummies(test_df['Embarked'])] , axis=1)



for i in train.columns:

    if len(str(i))<2:

        train.rename(columns={i: 'Embarked_'+i}, inplace=True)

for i in test_df.columns:

    if len(str(i))<2:

        test_df.rename(columns={i: 'Embarked_'+i}, inplace=True)      

        

train = pd.concat( [train , pd.get_dummies(train['Pclass'])] , axis=1)

test_df = pd.concat( [test_df, pd.get_dummies(test_df['Pclass'])] , axis=1)





for i in train.columns:

    if len(str(i))<2:

        train.rename(columns={i: 'Class_'+str(i)}, inplace=True)

for i in test_df.columns:

    if len(str(i))<2:

        test_df.rename(columns={i: 'Class_'+str(i)}, inplace=True)  



train.replace('male', 1,inplace=True)

train.replace('female', 0,inplace=True)

#train['Sex'] = train.where(train['Sex']=='male',1,0)



#train = train.drop(['Cabin','Embarked','Name','Ticket'], axis=1)

#test_df = test_df.drop(['Cabin','Embarked','Name','Ticket'], axis=1)



train = train.drop(['Cabin','Embarked','Ticket'], axis=1)

test_df = test_df.drop(['Cabin','Embarked','Ticket'], axis=1)



#could fill ages instead of drop

#train.dropna(subset = ['Age'],inplace=True)

#simply pluggin with mean







test_df['Age'].fillna(test_df['Age'].mean(),inplace=True)

train['Age'].fillna(train['Age'].mean(),inplace=True)



train['Sex'] = train['Sex'].astype(int)



del train['Pclass']

del test_df['Pclass']

del train['Name_Title']

del test_df['Name_Title']
train['Age']
#Still need to replacena for the 86 null ages and 1 null fare

test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)





#still need to convert male/female to 1/0

test_df.replace('male', 1,inplace=True)

test_df.replace('female', 0,inplace=True)

test_df['Sex'] = test_df['Sex'].astype(int)

test_df.set_index('PassengerId',inplace=True)
bins = [0, 10,20,30,40,50,60,120]

group_names = ['age0','age1','age2','age3','age4','age5','age6']

train['AgeBin'] = pd.cut(train['Age'], bins, labels=group_names)

test_df['AgeBin'] = pd.cut(test_df['Age'], bins, labels=group_names)



train = pd.concat( [train , pd.get_dummies(train['AgeBin'])] , axis=1)

test_df = pd.concat( [test_df, pd.get_dummies(test_df['AgeBin'])] , axis=1)



del train['Age']

del test_df['Age']

del train['AgeBin']

del test_df['AgeBin']



train = train.drop(['Embarked_C','Embarked_Q','Embarked_S'], axis=1)

test_df = test_df.drop(['Embarked_C','Embarked_Q','Embarked_S'], axis=1)



#Drop male feature when age <10

train['Sex'] = np.where(train['age0']==1, 0, train['Sex'])

train.where(train['age0']==1).dropna().head(5)
for i in [train, test_df]:

    i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo', \

        np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))

    

train = pd.concat( [train , pd.get_dummies(train['Fam_Size'])] , axis=1)

test_df = pd.concat( [test_df, pd.get_dummies(test_df['Fam_Size'])] , axis=1)    



del train['Parch']

del test_df['Parch']

del train['SibSp']

del test_df['SibSp']

del train['Fam_Size']

del test_df['Fam_Size']

splittrain, splittest = train_test_split(train, test_size = 0.2,random_state=0)
#for i in train.columns:

#    print(i + ' ' + str(train[i].dtype))

#print ('\n'+'\n')

#for i in test_df.columns:

#    print(i + ' ' + str(test_df[i].dtype))
#For testing

Xsplittrain = splittrain.drop('Survived', 1)

Ysplittrain = splittrain['Survived']



Xsplittest = splittest.drop('Survived', 1)

Ysplittest = splittest['Survived']



#For fitting all the data

Xtrain = train.drop('Survived', 1)

Ytrain = train['Survived']
regr = linear_model.LinearRegression()

random_forest = RandomForestClassifier(n_estimators=100)

gaussian = GaussianNB()

logreg = linear_model.LogisticRegression()



regr.fit(Xsplittrain, Ysplittrain)

random_forest.fit(Xsplittrain, Ysplittrain)

gaussian.fit(Xsplittrain, Ysplittrain)

logreg.fit(Xsplittrain, Ysplittrain)

#
rf = RandomForestClassifier(criterion='gini', 

                             n_estimators=700,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

rf.fit(Xsplittrain, Ysplittrain)
def runscore(x):

    print('Test Score: ' + str(x.score(Xsplittest, Ysplittest)))

    #print('\n')

    print('Train Score: ' + str(x.score(Xsplittrain, Ysplittrain)))
print('RandomForest')

runscore(random_forest)

print('\nLogistic')

runscore(logreg)

print('\nLinear')

runscore(regr)

print('\nGaussian')

runscore(gaussian)

print('\nRF')

runscore(rf)
#for i in range(10,40):

#    print(i)

#    random_forest = RandomForestClassifier(n_estimators=i)

#    random_forest.fit(Xsplittrain, Ysplittrain)

#    runscore(random_forest)   
# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



# preview

coeff_df
plt.matshow(train.corr())
train.corr()
logreg.fit(Xtrain, Ytrain)

rf.fit(Xtrain, Ytrain)
submission = pd.DataFrame({

        "PassengerId": test_df.index,

        "Survived": rf.predict(test_df)

    })



submission.to_csv('submissionFLYHI.csv', index=False)