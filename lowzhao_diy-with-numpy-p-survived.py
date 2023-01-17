import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import math



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
print(f'# data: {train.shape[0]}')
train.head()
print(f'Features:\n{set(train) - {"Survived"}}')

print('Labels:\n{\'Survived\'}\n')

print(f'Numeric:\n{set(train._get_numeric_data())}')

print(f'Non-numeric:\n{set(train) - set(train._get_numeric_data())}')
display(train.Cabin.isna().value_counts().rename({True:'NaN', False:'not NaN'}))

display(train.Cabin.describe())

display(train.Cabin.value_counts().head())
train.Cabin = ~train.Cabin.isna()

test.Cabin = ~test.Cabin.isna()
total = train.Survived.shape[0]

surv = train[train.Survived == 1]

decs = train[train.Survived == 0]

total_surv = np.sum(train.Survived)

total_decs = np.sum(train.Survived == 0)



print(f'total    : {total}')

print(f'survived : {total_surv}')

print(f'deceased : {total_decs}')

print(f'prior probabilty P(Survived): {total_surv / total}')

print(f'prior probabilty P(Deceased): {total_decs / total}')

train.Survived.value_counts().rename({1:'Survived', 0:'Deceased'}).plot(kind='pie',autopct='%.2f')

plt.show()
sex = train.Sex.value_counts()

sex.sort_index().plot(kind='pie', title='Sex', autopct='%.2f')

plt.show()

sex_surv = surv.Sex.value_counts()

sex_surv = sex_surv / sex

sex_surv = sex_surv / np.sum(sex_surv)

sex_surv.sort_index().plot(kind='pie',title='Survived', autopct='%.2f')

plt.show()

sex_decs = decs.Sex.value_counts()

sex_decs = sex_decs / sex

sex_decs = sex_decs / np.sum(sex_decs)

sex_decs.sort_index().plot(kind='pie',title='Deceased', autopct='%.2f')

plt.show()
print(f'P(Male ∩ Surv) = {np.sum(surv.Sex == "male") / total}')

print(f'P(Female ∩ Surv) = {np.sum(surv.Sex == "female") / total}')

print(f'P(Male ∩ Decs) = {np.sum(decs.Sex == "male") / total}')

print(f'P(Female ∩ Decs) = {np.sum(decs.Sex == "female") / total}')
_,ax = plt.subplots(1, 1)



ax.hist([surv.Pclass,decs.Pclass ],bins=5)



ax.legend(labels=['survived','deceased'])



plt.show()



def histogram(seq) -> None:

    """A horizontal frequency-table/histogram plot."""

    def count_elements(seq) -> dict:

        """Tally elements from `seq`."""

        hist = {}

        for i in seq:

            hist[i] = hist.get(i, 0) + 1

        return hist

    counted = count_elements(seq)

    for k in sorted(counted):

        print('{0:5d} {1}'.format(k,counted[k]))

print('Survived by class:')

histogram(surv.Pclass)

print('Deceased by class:')

histogram(decs.Pclass)



print('\n1: Upper, 2: Middle, 3: Lower')
print(f'P(Upper ∩ Surv) = {np.sum(surv.Pclass == 1) / total}')

print(f'P(Middle ∩ Surv) = {np.sum(surv.Pclass == 2) / total}')

print(f'P(Lower ∩ Surv) = {np.sum(surv.Pclass == 3) / total}\n')

print(f'P(Upper ∩ Decs) = {np.sum(decs.Pclass == 1) / total}')

print(f'P(Middle ∩ Decs) = {np.sum(decs.Pclass == 2) / total}')

print(f'P(Lower ∩ Decs) = {np.sum(decs.Pclass == 3) / total}')
female_low = train[(train.Sex == 'female') & (train.Pclass == 3)].Survived.value_counts().sort_index()

print(female_low.rename({1:'survived', 0:'deceased'}))

female_low_surv_prob = female_low[1]/np.sum(female_low)

female_low_decs_prob = female_low[0]/np.sum(female_low)

print(f'\nEntropy for female and lower class is {-female_low_surv_prob * math.log(female_low_surv_prob,2) - female_low_decs_prob * math.log(female_low_decs_prob,2)}\n\n')



female_mid = train[(train.Sex == 'female') & (train.Pclass == 2)].Survived.value_counts().sort_index()

print(female_mid.rename({1:'survived', 0:'deceased'}))

female_mid_surv_prob = female_mid[1]/np.sum(female_mid)

female_mid_decs_prob = female_mid[0]/np.sum(female_mid)

print(f'\nEntropy for female and middle class is {-female_mid_surv_prob * math.log(female_mid_surv_prob,2) - female_mid_decs_prob * math.log(female_mid_decs_prob,2)}\n\n')



female_up = train[(train.Sex == 'female') & (train.Pclass == 1)].Survived.value_counts().sort_index()

print(female_up.rename({1:'survived', 0:'deceased'}))

female_up_surv_prob = female_up[1]/np.sum(female_up)

female_up_decs_prob = female_up[0]/np.sum(female_up)

print(f'\nEntropy for female and middle class is {-female_up_surv_prob * math.log(female_up_surv_prob,2) - female_up_decs_prob * math.log(female_up_decs_prob,2)}')
male_low = train[(train.Sex == 'male') & (train.Pclass == 3)].Survived.value_counts().sort_index()

print(male_low.rename({1:'survived', 0:'deceased'}))

male_low_surv_prob = male_low[1]/np.sum(male_low)

male_low_decs_prob = male_low[0]/np.sum(male_low)

print(f'\nEntropy for male and lower class is {-male_low_surv_prob * math.log(male_low_surv_prob,2) - male_low_decs_prob * math.log(male_low_decs_prob,2)}\n\n')



male_mid = train[(train.Sex == 'male') & (train.Pclass == 2)].Survived.value_counts().sort_index()

print(male_mid.rename({1:'survived', 0:'deceased'}))

male_mid_surv_prob = male_mid[1]/np.sum(male_mid)

male_mid_decs_prob = male_mid[0]/np.sum(male_mid)

print(f'\nEntropy for male and middle class is {-male_mid_surv_prob * math.log(male_mid_surv_prob,2) - male_mid_decs_prob * math.log(male_mid_decs_prob,2)}\n\n')



male_up = train[(train.Sex == 'male') & (train.Pclass == 1)].Survived.value_counts().sort_index()

print(male_up.rename({1:'survived', 0:'deceased'}))

male_up_surv_prob = male_up[1]/np.sum(male_up)

male_up_decs_prob = male_up[0]/np.sum(male_up)

print(f'\nEntropy for male and middle class is {-male_up_surv_prob * math.log(male_up_surv_prob,2) - male_up_decs_prob * math.log(male_up_decs_prob,2)}')
def graph(formula, s,e,step):  

    x = np.arange(start=s,stop=e,step=step)

    y = eval(formula)

    plt.plot(x, y)  

    plt.show()

p = 0.366

graph('x*p+(1-x)*(1-p)',0,1,0.01)

p = 0.777

graph('x*p+(1-x)*(1-p)',0,1,0.01)
test_try_1 = test[['Sex', 'Pclass', 'PassengerId']]

def choose(row):

    if row.Sex == 'male':

        return 0  # because since all type of male have higher deceased probability.

    else:

        return 1  # because since all type of women have higher survived probability, altough one is 50-50 chance.

test_try_1['Survived'] = test_try_1.apply(lambda x: choose(x),axis=1)

test_try_1 = test_try_1[['PassengerId','Survived']]

test_try_1.to_csv('./test.csv',index=False)
_,ax = plt.subplots(figsize=(20,5))

sns.distplot(surv.Age,ax=ax, bins=80)

sns.distplot(decs.Age,ax=ax, bins=80)

ax.legend(['survived', 'deceased'])

display(surv.Age.describe())

display(decs.Age.describe())