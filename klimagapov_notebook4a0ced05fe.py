
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv('../input/adult-census-income/adult.csv',
na_values = ':',
usecols = ['age', 'workclass', 'fnlwgt', 'education',
'education.num', 'marital.status', 'occupation', 'relationship',
'race', 'sex', 'capital.gain', 'capital.loss',
'hours.per.week', 'native.country', 'income'])
df.head()

ml = df[(df.sex == 'Male')]
fm = df[(df.sex == 'Female')]
ml1 = df[(df.sex == 'Male') & (df.income=='>50K') ]
fm1 = df[(df.sex == 'Female') & (df.income=='>50K')]

ml_marital=ml['marital.status']
fm_marital=fm['marital.status']

ml_age = ml['age']
fm_age = fm['age']
df.shape
counts = df.groupby('native.country').size()
print (counts)
df1 = df[(df.income=='>50K')]
print ('The rate of people with high income is: ',
int(len(df1)/float(len(df))*100), '%.')
print ('The rate of men with high income is: ',
int(len(ml1)/float(len(ml))*100), '%.')
print ('The rate of women with high income is: ',
int(len(fm1)/float(len(fm))*100), '%.')
print(ml_marital)
ml_marital.hist(histtype='stepfilled', bins=12, color='b')

print(fm_marital)
fm_marital.hist(histtype='stepfilled', bins=12, color='b')
import seaborn as sns
fm_marital.hist(histtype = 'stepfilled',
alpha = .5, bins = 20)
ml_marital.hist(histtype = 'stepfilled',
alpha = .5, bins = 20, color = sns.desaturate("indianred",.75))
df2 = df.drop(df.index[(df['marital.status'] == 'Widowed') &( (df['age'] > df['age'].median() + 35) | (df['age'] < df['age'].median() - 15))])
ml1_age = ml1['age']
fm1_age = fm1['age']
ml2_age = ml1_age.drop(ml1_age.index[(ml1_age > (df['age'].median() + 35)) | (ml1_age < (df['age'].median() - 15))])
fm2_age = fm1_age.drop(fm1_age.index[(fm1_age > df['age'].median() + 35) | (fm1_age < df['age'].median() - 15)])
ml2_age.min()
import matplotlib.pyplot as plt
plt.figure(figsize = (13.4, 5))
df.age[(df['marital.status'] == 'Widowed')].plot(alpha = .25, color = 'black')
df2.age[(df2['marital.status'] == 'Widowed')].plot(alpha = .45, color = 'red')

import numpy as np

wid = df[(df['marital.status'] == 'Widowed')]
div = df[(df['marital.status'] == 'Divorced')]
wid1 = wid['age']
div1 = div['age']
print(ml2_age)

countx, divisionx = np.histogram(wid1)
county, divisiony = np.histogram(div1)
val = [(divisionx[i] + divisionx[i+1])/2
for i in range(len(divisionx) - 1)]
plt.plot(val, countx - county, 'o-')