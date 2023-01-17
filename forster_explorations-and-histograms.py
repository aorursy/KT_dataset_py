# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

df.head()

cols = list(df.columns)



cols_categ_count = { i: len(set(df[i])) for i in cols}



cols_categ_count
cols_categorical = { i: set(df[i]) for i in cols if cols_categ_count[i]<=10 }



cols_categorical
cols_numerical = [i for i in cols if not i in cols_categorical]



cols_numerical
positive_guys = df[df['SARS-Cov-2 exam result']=='positive']



negative_guys = df[df['SARS-Cov-2 exam result']=='negative']



f"positive: {len(positive_guys)} negative: {len(negative_guys)} total: {len(df)} "
558+5086
list(cols_categorical.keys())[0:10]
for i in list(cols_categorical.keys())[0:]:

    names=cols_categorical[i]

    countvec_pos = [len(positive_guys[positive_guys[i]==k])/len(positive_guys) for k in names]

    countvec_neg = [len(negative_guys[negative_guys[i]==k])/len(negative_guys) for k in names]

    

    plt.figure()

    plt.title(i)

    plt.plot(range(len(names)),countvec_pos,'g*', range(len(names)),countvec_neg,'ro')

    
cols_to_check=cols_numerical[1:]

cols_to_check.remove('Mycoplasma pneumoniae')

cols_to_check.remove('Urine - pH')

cols_to_check.remove('Urine - Sugar')

cols_to_check.remove('Urine - Leukocytes')

cols_to_check.remove('Partial thromboplastin time\xa0(PTT)\xa0')

cols_to_check.remove('Prothrombin time (PT), Activity')

cols_to_check.remove('D-Dimer')





for i in cols_to_check:

    print(i)

    plt.figure()

    plt.title(i)

    plt.hist([positive_guys[i], negative_guys[i]], 

             bins=20, density=True, 

             histtype='step',color=['green', 'red'],

             label=['positive','negative']

            )

    plt.legend()