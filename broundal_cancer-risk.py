# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
## load

df = pd.read_csv('../input/kag_risk_factors_cervical_cancer.csv', na_values=['?'])

df.fillna(0, inplace=True)

df.head()
## train model

samples = df.iloc[:10, :-1]

y_test = df.iloc[:10, -1].values

train = df.iloc[10:, :]

x = train.values[:, :-1].astype(np.float32)

y = train.values[:, -1].astype(np.float32)



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, max_depth=10, random_state=1)

model.fit(x, y)

model.oob_score_
## feature importance

importances = list(sorted(zip(df.columns[:-1], model.feature_importances_), key=lambda x: x[1], reverse=True))

importances
def is_test(k):

    if 'STDs' in k:

        return True

    if 'Dx' in k:

        return True

    if k in ['Citology','Hinselmann','Schiller']:

        return True

    

    return False

test_importnace = [k for k in importances if is_test(k[0])]

test_importnace
list(zip(model.predict(samples.values), y_test))
samples
def P(x):

    return model.predict_proba(x.values)



np.round(P(samples), 2)
def base(samples):

    samples['Age'] += 1

    samples['Smokes (years)'] += samples['Smokes']

    samples['Hormonal Contraceptives (years)'] += samples['Hormonal Contraceptives']

    samples['IUD (years)'] += samples['IUD']

    return samples



def start_smoking(samples):

    samples['Smokes'] = 1 

    samples['Smokes (years)'] += 1

    return samples



def start_using_contraceptives(samples):

    samples['Hormonal Contraceptives'] = 1 

    samples['Hormonal Contraceptives (years)'] += 1

    return samples



def start_using_IUD(samples):

    samples['IUD'] = 1 

    samples['IUD (years)'] += 1

    return samples



def enter_pregnancy(samples):

    samples['Num of pregnancies'] +=1

    return samples



def smoke_less(samples):

    samples['Smokes (packs/year)'] -= 5*samples['Smokes']

    return samples
import copy

for i in range(10):

    sample = samples.iloc[i:i+1,:].copy()

    chance_for_cancer = P(sample)[0,1]    

    print('\n\noriginal chance_for_cancer for patient #%d: %2g' % (i,chance_for_cancer))

    

    s = base(sample.copy())

    new_natural_chance_for_cancer = P(s)[0,1]

    if new_natural_chance_for_cancer >= chance_for_cancer+1e-2:

        print('next year chance_for_cancer will be higher by %2g' % (100*(new_natural_chance_for_cancer-chance_for_cancer)))

    elif new_natural_chance_for_cancer <= chance_for_cancer-1e-2:

        print('next year chance_for_cancer will be lower by %2g' % (100*(chance_for_cancer-new_natural_chance_for_cancer)))

    

    for f in [start_smoking, start_using_contraceptives, enter_pregnancy, smoke_less]:

        s = copy.deepcopy(sample.copy())

        s = base(s)

        s = f(s)

        new_chance = P(s)[0,1]

        if new_chance > new_natural_chance_for_cancer+1e-2:

            print('* Recommended to NOT {} as chance is will be higher by {:.3g}%'.format(f.__name__, (new_chance-new_natural_chance_for_cancer)*100))

        elif new_chance < new_natural_chance_for_cancer-1e-2:

            print('* Recommended to {} as chance is will be lower by {:.3g}%'.format(f.__name__, (new_natural_chance_for_cancer - new_chance)*100))
print("The most important tests for cancer are:\n {}".format(test_importnace[:3]))