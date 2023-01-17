from os import path

import pandas as pd

import seaborn as sns

%matplotlib inline
#import files from local 

battles_df = pd.read_csv('../input/battles.csv')
battles_df.head()
battles_df.info()
battles_df.attacker_outcome.head()
sns.countplot(x='attacker_king',data=battles_df,hue='attacker_outcome')
#convert wins and loss to int

battles_df['attacker_outcome'].replace('win',1,inplace=True)

battles_df['attacker_outcome'].replace('loss',0,inplace=True)
sns.barplot(x='year',y='attacker_outcome', data=battles_df)
#variations on attacker_outcom based on year are very less hence dropping the column

battles_df.drop('year',axis=1,inplace=True)
#name of the battle is insignificant to the outcome

battles_df.drop(['name','battle_number','note'],axis=1,inplace=True)
battles_df.info()
battles_df.attacker_2.head()
pattern = r'[a-z][0-9]'

test = battles_df[~battles_df.attacker_2.isnull()].attacker_2.replace(pattern,1,regex=True)
battles_df.attacker_2.fillna(0,inplace=True)

battles_df.attacker_3.fillna(0,inplace=True)

battles_df.attacker_4.fillna(0,inplace=True)
def find_attacker_allies(my_data):

    col2 = my_data['attacker_2']

    col3 = my_data['attacker_3']

    col4 = my_data['attacker_4']

    number = 0

    if col2 != 0:

        number = number+1

    if col3 != 0:

        number = number+1

    if col4 != 0:

        number = number+1

    

    return number



battles_df['attacker_allies'] = battles_df.apply(find_attacker_allies,axis=1)
#drop attacker_2 , attacker_3, attacker_4

battles_df.drop(['attacker_2','attacker_3','attacker_4'],axis=1,inplace=True)
battles_df.defender_2.fillna(0,inplace=True)

battles_df.defender_3.fillna(0,inplace=True)

battles_df.defender_4.fillna(0,inplace=True)
def find_defender_allies(my_data):

    col2 = my_data['defender_2']

    col3 = my_data['defender_3']

    col4 = my_data['defender_4']

    number = 0

    if col2 != 0:

        number = number+1

    if col3 != 0:

        number = number+1

    if col4 != 0:

        number = number+1

    

    return number



battles_df['defender_allies'] = battles_df.apply(find_defender_allies,axis=1)



battles_df.drop(['defender_2','defender_3','defender_4'],axis=1,inplace=True)
battles_df.head()
battles_df.attacker_king.fillna('None',inplace=True)

battles_df.defender_king.fillna('None',inplace=True)

battles_df.battle_type.fillna('None',inplace=True)

battles_df.attacker_commander.fillna('None',inplace=True)

battles_df.defender_commander.fillna('None',inplace=True)

battles_df.location.fillna('None',inplace=True)

battles_df.region.fillna('None',inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

battles_df.attacker_king = le.fit_transform(battles_df.attacker_king)

battles_df.defender_king = le.fit_transform(battles_df.defender_king)

battles_df.battle_type = le.fit_transform(battles_df.battle_type)

battles_df.attacker_commander = le.fit_transform(battles_df.battle_type)

battles_df.defender_commander = le.fit_transform(battles_df.battle_type)

battles_df.location = le.fit_transform(battles_df.battle_type)

battles_df.region = le.fit_transform(battles_df.battle_type)
battles_df.drop(['attacker_1','defender_1'],axis=1,inplace=True)

battles_df.fillna(0,inplace=True)
X = battles_df.drop('attacker_outcome',axis=1)

y = battles_df.attacker_outcome
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
from sklearn.cross_validation import cross_val_score

scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
scores.mean()
k_range = range(1,15)

scores_list = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')

    scores_list.append(scores.mean())
import matplotlib.pyplot as plt
plt.plot(k_range,scores_list)
max(scores_list)
knn = KNeighborsClassifier(n_neighbors=7)

scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')

scores.mean()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
scores = cross_val_score(logreg,X,y,cv=10,scoring='accuracy')

scores.mean()