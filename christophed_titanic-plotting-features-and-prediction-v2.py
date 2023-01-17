# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Explo
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.info())
print(test.info())
print(len(test))
bleu_c = '#9EB9DC'
bleu_f = '#0755BA'
saum = '#E8D0A8'
vert = '#7EC66F'
rouge = '#CB4141'
orange = '#FEAF10'
jaune = '#E8E865'
print(train.groupby('SibSp')['SibSp'].count())
train.groupby('Parch')['Parch'].count()

plt.figure(figsize = (20,6))
plt.rc('axes', axisbelow=True)

plt.subplot(131)
bins = [x + 0.5 for x in range(0,4)]
y1=train.loc[train['Survived'] == 1,'Pclass']
y2=train.loc[train['Survived'] == 0,'Pclass']
plt.hist([y1,y2], bins = bins, color = [vert,rouge], label = ['Survécu','Non-Survécu'])
plt.title('Classes de billet')
#plt.xticks(locs = range(1,4), labels = ['1ere','2è','3è'])
plt.ylabel('Compte')
plt.grid(True, axis ='y')
plt.legend()

plt.subplot(132)
bins = [x + 0.5 for x in range(-1,7)]
y1 = train.loc[train['Survived'] == 1, 'Parch']
y2 = train.loc[train['Survived'] == 0, 'Parch']
plt.hist([y1,y2], bins = bins, color = [vert,rouge], label = ['Survécu','Non-Survécu'])
plt.title('Enfants / parents')
#plt.xticks(range(0,7),['0','1','2','3','4','5','6'])
plt.ylabel('Compte de passagers')
plt.xlabel('Nombre')
plt.grid(True, axis ='y')
plt.legend()

plt.subplot(133)
bins = [x + 0.5 for x in range(-1,9)]
y1 = train.loc[train['Survived'] == 1, 'SibSp']
y2 = train.loc[train['Survived'] == 0, 'SibSp']
plt.hist([y1,y2], bins = bins, color = [vert,rouge], label = ['Survécu','Non-Survécu'])
plt.title('Frères/soeurs/époux/epouses')
#plt.xticks(range(0,9),['0','1','2','3','4','5','6','7','8'])
plt.ylabel('Compte de passagers')
plt.xlabel('Nombre')
plt.grid(True, axis ='y')
plt.legend()

y1 = train.loc[train['Survived'] == 1, 'Age'].dropna()
y2 = train.loc[train['Survived'] == 0, 'Age'].dropna()

plt.figure(figsize = (15,6))
plt.hist([y1,y2], bins = range(0,80,4), width = 4, edgecolor = 'black', linewidth = 0.5, color = [vert,rouge], histtype = 'barstacked', label = ['Survécu','Non-Survécu'])
plt.title('Distribution d\'age des passagers')
plt.xlabel('Age')
plt.ylabel('Compte')
plt.legend()

h1 = np.histogram(y1.dropna().values, bins = [0,10,18,50,80])
h2 = np.histogram(y2.dropna().values, bins = [0,10,18,50,80])

plt.figure(figsize = (15,6))
plt.subplot(121)
plt.pie(h1[0], colors = [bleu_c, vert, jaune, rouge], shadow = True, autopct = lambda x: str(round(x,1)) + '%', labels = ['Jeunes enfants ', 'Adolescents ', 'Adultes', 'Anciens'])
plt.title('Partage des survivants par âge')
plt.legend()
plt.subplot(122)
plt.pie(h2[0], colors = [bleu_c, vert, jaune, rouge], shadow = True, autopct = lambda x: str(round(x,1)) + '%', labels = ['Jeunes enfants', 'Adolescents', 'Adultes', 'Anciens (50-80)'])
plt.title('Partage des victimes par âge')
plt.legend()
#  Nettoyage NAs + fusion de train et set pour la préparation de features
train.dropna(subset = ['Embarked', 'Fare'], inplace = True)
train.reset_index(drop = True, inplace = True)
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].value_counts().index[0])
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
print(len(test))

full = pd.concat([train.drop('Survived', axis =1), test], ignore_index = True)
full.drop(['PassengerId', 'Ticket'], axis = 1, inplace = True)

full['Age'] = full['Age'].interpolate()
full.shape
# Création de la variable catégorielle deck
def cat_deck(string):
    return 'Missing' if pd.isnull(string) else str(string)[0]
full['Deck'] = full['Cabin'].apply(lambda x: cat_deck(x))
full = full.join(pd.get_dummies(full['Deck'], prefix = 'Deck'))
full = full.drop(['Cabin','Deck','Deck_T'], axis = 1)

# Variable Age
full['age_cat'] = pd.cut(full['Age'], bins = [0,10,18,50,80.1], labels = [1,2,3,4])
# Variable SibSp
full['Sib_cat'] = pd.cut(full['SibSp'], bins = [-1,0,2,8], labels = [1,2,3])
# Variable Parch
full['Parch_cat'] = pd.cut(full['Parch'], bins = [-1,0,2,9], labels = [1, 2, 3])
# Dummy pour Embarked
full = full.join(pd.get_dummies(full['Embarked'], prefix = 'Embarked'))
full.drop(['Embarked', 'Name'], axis = 1, inplace = True)
# Turn sex into binary m=1 and f=0
full['Sex'] = full['Sex'].apply(lambda x: 1 if x == 'male' else 0)

full.drop(['Age', 'SibSp', 'Parch', 'Deck_Missing'], axis = 1, inplace = True)
full.head()
(full.isna().sum())

# Split back in train and test
tr = full.iloc[range(len(train)),:]
te = full.iloc[len(train):,:]
target = train['Survived']
print(len(te))

# Load libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
clf = DecisionTreeClassifier(max_depth = 9)
clf.fit(tr, target)
pred = clf.predict(te)
dir(clf)

bag_tree = BaggingClassifier(clf,n_estimators = 200)
bag_tree.fit(tr,target)
pred_bag = bag_tree.predict(te)

submission = pd.DataFrame.from_dict({'PassengerId' : test['PassengerId'], 'Survived' : pred})
submission.to_csv('Submit.csv', index = False)

submission = pd.DataFrame.from_dict({'PassengerId' : test['PassengerId'], 'Survived' : pred_bag})
submission.to_csv('Submit_bag.csv', index = False)

imp = pd.DataFrame({'Importance': clf.feature_importances_})
imp['Features'] = tr.columns
imp.sort_values(by = 'Importance', ascending = False, inplace =True)
imp.reset_index(drop = True, inplace = True)
plt.figure()
plt.barh(range(15,0,-1),imp['Importance'].iloc[0:15], tick_label = imp['Features'].iloc[0:15])


#imp = pd.DataFrame({'Importance': bag_tree.feature_importances_})
#imp['Features'] = tr.columns
#imp.sort_values(by = 'Importance', ascending = False, inplace =True)
#imp.reset_index(drop = True, inplace = True)
#plt.figure()
#plt.barh(range(15,0,-1),imp['Importance'].iloc[0:15], tick_label = imp['Features'].iloc[0:15])