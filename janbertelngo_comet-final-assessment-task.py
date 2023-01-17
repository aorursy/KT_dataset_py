import numpy as np

import pandas as pd

import re

from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
import numpy as np

import pandas as pd

import re

from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt

import seaborn as sns

pd.options.mode.chained_assignment = None

pd.set_option('display.max_columns', 1000)

sns.set_style('whitegrid')

sns.color_palette('pastel')

%matplotlib inline
train_data = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_data = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
print(train_data.shape); 

print(test_data.shape);

print(train_data.columns);

train_data.describe
fig = plt.figure(figsize=(15,12), dpi=1600) 

c = ['#24678d','#4caac9','#626262','#d5d5d5','#248d6c']
ax1 = plt.subplot2grid((5,3),(0,0))

train_data.Survived.value_counts().plot(kind='bar', alpha=0.85,color=c[3])

plt.title("Distribution of Survival, (1 = Survived)")



ax2 = plt.subplot2grid((5,3),(0,1),colspan=2)

train_data.Pclass.value_counts().plot(kind='bar', alpha=0.85,color=c[0])

plt.title("Distribution of Classes, (Pclass)")
ax3 = plt.subplot2grid((5,3),(1,0),colspan=2)              

sns.kdeplot(train_data.Age, shade=True, color=c[0], label='Age')

plt.axvline(train_data.Age.median(), color=c[4], label='Median', ls='dashed')

plt.axvline(train_data.Age.mean(), color=c[2], label='Mean', ls='dashed')

plt.legend()

plt.title("Distribution of Age, (Age)")



ax4 = plt.subplot2grid((5,3),(1,2),colspan=2)

train_data.Sex.value_counts().plot(kind='bar', alpha=0.85,color=c[3])

plt.title("Distribution of Gender, (Sex)")
ax5 = plt.subplot2grid((5,3),(2,0))              

train_data.Embarked.value_counts().plot(kind='bar', alpha=0.85,color=c[3])

plt.title("Distribution of Embarked, (Embarked)")



ax6 = plt.subplot2grid((5,3),(2,1),colspan=2)

sns.kdeplot(train_data.Fare[train_data.Pclass==1].apply(lambda x: 80 if x>80 else x), shade=True, color=c[2], label='1st Class')

sns.kdeplot(train_data.Fare[train_data.Pclass==2].apply(lambda x: 80 if x>80 else x), shade=True, color=c[0], label='2nd Class')

sns.kdeplot(train_data.Fare[train_data.Pclass==3].apply(lambda x: 80 if x>80 else x), shade=True, color=c[1], label='3rd Class')

plt.axvline(train_data.Fare.median(), color=c[4], label='Median', ls='dashed')

plt.axvline(train_data.Fare.mean(), color=c[3], label='Mean', ls='dashed')

ax6.set_xlim(-10, 100)

plt.legend()

plt.title("Fare Distribution by Class, (Fare)")
ax7 = plt.subplot2grid((5,3),(3,0),colspan=2)

sns.kdeplot(train_data.Age[train_data.Sex=='male'], shade=True, color=c[0], label='Male')

sns.kdeplot(train_data.Age[train_data.Sex=='female'], shade=True, color=c[4], label='Female')

plt.legend()

plt.title("Age Distribution by Sex")



ax8 = plt.subplot2grid((5,3),(3,2))

train_data.SibSp.value_counts().plot(kind='bar', alpha=0.55,color=c[2])

train_data.Parch.value_counts().plot(kind='bar', alpha=0.55,color=c[4])

plt.legend()

plt.title("Number of Siblings, Spouses, Parents and Children")
fig = plt.figure(figsize=(16,9), dpi=1600)

c = ['#24678d','#4caac9','#626262','#d5d5d5','#248d6c']
ax1 = plt.subplot2grid((16,15),(0,0),rowspan=16, colspan=6)

train_data.Survived.value_counts().plot(kind='bar', alpha=0.85,color=c[3],ax=ax1)

plt.title("Number of Survivors (1=Survived)")
ax2 = plt.subplot2grid((16,15),(0,6),rowspan=4,colspan=3)

train_data.Survived[train_data.Pclass==1].value_counts().sort_index().plot(kind='bar', alpha=0.85,color=c[0], label='1st Class')

plt.legend()

plt.title("Survival in 1st Class")



ax3 = plt.subplot2grid((16,15),(0,9),rowspan=4,colspan=3)

train_data.Survived[train_data.Pclass==2].value_counts().sort_index().plot(kind='bar', alpha=0.85,color=c[1], label='2nd Class')

plt.legend()

plt.title("Survival in 2nd Class")



ax4 = plt.subplot2grid((16,15),(0,12),rowspan=4,colspan=3)              

train_data.Survived[train_data.Pclass==3].value_counts().sort_index().plot(kind='bar', alpha=0.85,color=c[4], label='3rd Class')

plt.legend()

plt.title("Survival in 3rd Class")
ax5 = plt.subplot2grid((16,15),(4,6),rowspan=4,colspan=9)

plt.bar(np.array([0,1])-0.25, train_data.Survived[train_data.Sex=='male'].value_counts().sort_index(), width=0.25,color=c[0], label='Male',alpha=0.85)

plt.bar(np.array([0,1]), train_data.Survived[train_data.Sex=='female'].value_counts().sort_index(), width=0.25,color=c[1], label='Female',alpha=0.85)

plt.xticks(np.arange(0, 2, 1))

plt.legend()

plt.title("Survival By Gender (Sex)")
ax6 = plt.subplot2grid((16,15),(8,6),rowspan=4,colspan=9)

sns.kdeplot(train_data.Age[train_data.Survived==0],shade=True, color=c[2], label='Died')

sns.kdeplot(train_data.Age[train_data.Survived==1], shade=True, color = c[0], label='Survived')

plt.legend()

plt.title("Survival By Age")
ax7 = plt.subplot2grid((16,15),(12,6),rowspan=4,colspan=3)

plt.hist(train_data.Survived[pd.Categorical(train_data.Embarked).codes==0], color=c[2], label='Embarked at C')

plt.legend()

plt.title("Survival By Embarked (C)")



ax8 = plt.subplot2grid((16,15),(12,9),rowspan=4,colspan=3)

plt.hist(train_data.Survived[pd.Categorical(train_data.Embarked).codes==1], color=c[3], label='Embarked at Q')

plt.legend()

plt.title("Survival By Embarked (Q)")



ax9 = plt.subplot2grid((16,15),(12,12),rowspan=4,colspan=3)

plt.hist(train_data.Survived[pd.Categorical(train_data.Embarked).codes==2], color=c[0], label='Embarked at S')

plt.legend()

plt.title("Survival By Embarked (S)")
train_data['FamilySize'] = train_data.SibSp + train_data.Parch;

fig = plt.figure(figsize=(16,9), dpi=1600);

sns.kdeplot(train_data.FamilySize[train_data.Survived==0],shade=True,color=c[2],label='Dead');

sns.kdeplot(train_data.FamilySize[train_data.Survived==1],shade=True,color=c[1],label='Survived');

plt.title('Survival By Family Size');

plt.legend();

plt.show();
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



train_data['Title'] = train_data["Name"].apply(get_title);

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 7, "Dona":10, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 7, "Capt": 7, "Ms": 2}

train_data["TitleCat"] = train_data.loc[:,'Title'].map(title_mapping);

fig = plt.figure(figsize=(8,4.5), dpi=1600);

plt.hist(train_data.TitleCat[(train_data.Survived==0)], alpha=0.5,color=c[2],label='Dead');

plt.hist(train_data.TitleCat[(train_data.Survived==1)], alpha=0.5,color=c[1],label='Survived');

plt.xticks(range(1,11));

plt.legend(); 

plt.show();

train_data.pivot_table(index=['Title'], values=['Age']);
train_data['CabinBlock'] = train_data.Cabin.fillna('0').apply(lambda x: x[0])

train_data['CabinCat'] = pd.Categorical.from_array(train_data.Cabin.fillna('0').apply(lambda x: x[0])).codes

sns.kdeplot(train_data.CabinCat[(train_data.Survived==0) & (train_data.CabinCat!=0)],shade=True,color=c[2],label='Dead')

sns.kdeplot(train_data.CabinCat[(train_data.Survived==1) & (train_data.CabinCat!=0)],shade=True,color=c[1],label='Survived')

fig = plt.figure(figsize=(8,4.5), dpi=1600);

plt.title('Survival By Cabin Passengers only'); 

plt.legend();

plt.show();
child_age = 14

def get_person(passenger):

    age, sex = passenger

    if (age < child_age):

        return 'child'

    elif (sex == 'female'):

        return 'female_adult'

    else:

        return 'male_adult'

    

train_data = pd.concat([train_data, pd.DataFrame(train_data[['Age', 'Sex']].apply(get_person, axis=1), columns=['person'])],axis=1)

dummies = pd.get_dummies(train_data['person'])

train_data = pd.concat([train_data,dummies],axis=1)