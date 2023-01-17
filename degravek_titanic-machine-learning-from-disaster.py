# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn import preprocessing
from collections import Counter
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

# Load the train and test datasets to create two DataFrames
train = pd.read_csv('../input/train.csv',header=0)
test = pd.read_csv('../input/test.csv',header=0)
#Combine training and test data
comb = train.append(test,ignore_index=True)
comb.head()
# Look at the data
ptab = pd.pivot_table(comb,index='Sex',columns='Survived',aggfunc='size')
ptab = ptab.div(ptab.sum(axis=1), axis=0)
ptab.plot.barh(stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.xlabel('Fraction of People')
plt.ylabel('Sex')
# Make Sex numerical
comb['nSex'] = comb['Sex']
comb['nSex'] = comb['nSex'].map({'male': 1, 'female': 0}).astype(int)
# Make a new column keeping the "n" prefix
comb['nPclass'] = comb['Pclass']

# Look at the data
ptab = pd.pivot_table(comb,index=['Sex','Pclass'],columns='Survived',aggfunc='size')
ptab = ptab.div(ptab.sum(axis=1), axis=0)
ptab.plot.barh(stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.xlabel('Fraction of People')
plt.ylabel('Class')
# Embarked has two NaNs. Make a new column and fill them in with most common class.
count = Counter(comb['Embarked'])
tmp = count.most_common()[0][0]
comb['nEmbarked'] = comb['Embarked']
comb['nEmbarked'] = comb['nEmbarked'].fillna(tmp)

# Look at the data
ptab = pd.pivot_table(comb,index=['Sex','nEmbarked'],columns='Survived',aggfunc='size')
ptab = ptab.div(ptab.sum(axis=1), axis=0)
ptab.plot.barh(stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.xlabel('Fraction of People')
plt.ylabel('Embarked')

# Make Embarked numerical
comb.ix[comb.nEmbarked == 'Q', 'nEmbarked']  = 0
comb.ix[comb.nEmbarked == 'C', 'nEmbarked']  = 1
comb.ix[comb.nEmbarked == 'S', 'nEmbarked']  = 2
# Look at the data
ptab = pd.pivot_table(comb,index=['nEmbarked'],columns='nPclass',aggfunc='size')
ptab = ptab.div(ptab.sum(axis=1), axis=0)
ptab.plot.barh(stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.xlabel('Fraction of People')
plt.ylabel('Embarked')
# Look at distribution of fares
fig = comb.Fare.plot.hist(alpha=0.8,bins=20)
fig.set(xlabel="Fare [dollars]", ylabel="Frequency")
fig.legend(['Fare'])
# Look at the data
comb.boxplot(column='Fare',by='Pclass')
plt.xlabel('Class')
plt.ylabel('Fare [dollars]')
plt.suptitle("")
plt.title('Fare by Passenger Class')
# Create new column and fill in NaN with median third class fare.
comb['nFare'] = comb['Fare']
tmp = comb[comb['Pclass']==3]
comb.ix[comb.nFare.isnull(), 'nFare'] = tmp['nFare'].median()
# Create a new feature called Title and give it numerical values
comb['nTitle'] = comb['Name']
comb['nTitle'] = comb['nTitle'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# Make the titles numerical
tmp = preprocessing.LabelEncoder()
tmp.fit(comb['nTitle'])
comb['nTitle'] = tmp.transform(comb['nTitle'])
# Make new feature giving family size.
# For consistency make new Parch and SibSp columns
comb['nFamSize'] = comb['Parch']+comb['SibSp']
comb['nParch'] = comb['Parch']
comb['nSibSp'] = comb['SibSp']
# Create new nAgeGroup feature
comb['nAgeGroup'] = np.nan
comb.ix[(comb['Age'] >  0) & (comb['Age'] <= 10), 'nAgeGroup'] = '0 - 10yr'
comb.ix[(comb['Age'] > 10) & (comb['Age'] <= 20), 'nAgeGroup'] = '10yr - 20yr'
comb.ix[(comb['Age'] > 20) & (comb['Age'] <= 40), 'nAgeGroup'] = '20yr - 40yr'
comb.ix[(comb['Age'] > 40) & (comb['Age'] <= 60), 'nAgeGroup'] = '40yr - 60yr'
comb.ix[(comb['Age'] > 60) & (comb['Age'] <= 80), 'nAgeGroup'] = '60yr - 80yr'

# Look at the data
ptab = pd.pivot_table(comb,index=['Sex','nAgeGroup'],columns='Survived',aggfunc='size')
ptab = ptab.div(ptab.sum(axis=1), axis=0)
ptab.plot.barh(stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.xlabel('Fraction of People')
plt.ylabel('Ages')
# Make new age column and fit for missing values
comb['nAge'] = comb['Age']

feat = comb.ix[comb.Age.notnull()][['nSex','nPclass','nEmbarked','nFare','nTitle','nFamSize','nParch','nSibSp']].values
targ = comb.ix[comb.Age.notnull()][['nAge']].values.astype(int)

forest = RandomForestRegressor(n_estimators = 600)
forest = forest.fit(feat,np.ravel(targ))

feat = comb.ix[comb.Age.isnull()][['nSex','nPclass','nEmbarked','nFare','nTitle','nFamSize','nParch','nSibSp']].values
comb.ix[comb.Age.isnull(), 'nAge'] = forest.predict(feat)
# Create final training and test sets
train = comb.ix[0:train.shape[0]-1]

# Random forest classifier
feat = train[['nAge','nSex','nPclass','nEmbarked','nFare','nTitle','nFamSize','nParch','nSibSp']].values
targ = train[['Survived']].values

forest = RandomForestClassifier(n_estimators = 600)
forest = forest.fit(feat,np.ravel(targ))
scores = cross_validation.cross_val_score(forest,feat,np.ravel(targ),cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Look at feature importance
names = ['Age','Fare','Sex','Title','Class','Family Size','SibSp','Embarked','Parch']

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.bar(range(feat.shape[1]), importances[indices],color="r", align="center")
plt.xticks(range(feat.shape[1]), indices)
plt.xlim([-1, feat.shape[1]])
y_pos = np.arange(len(names))
plt.xticks(y_pos, names, rotation='vertical')
plt.ylabel('Relative Importance')
plt.title("Feature importances")