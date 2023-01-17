# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 
import os

os.listdir('../input')
df = pd.read_csv('../input/vodafone-subset-3.csv');

df
import seaborn as sns
df['total_duration'] = df['calls_duration_in_weekdays'] + df ['calls_duration_out_weekdays']+df['calls_count_in_weekends']+ df['calls_count_out_weekends']

a = df[['gender','total_duration']]

b = a['gender'].value_counts()

x = a.groupby('gender')['total_duration'].sum()

df.groupby('gender')['total_duration'].sum().plot(kind='bar')#на картинке можно увидеть, что женщины действительно больше разговаривают по телефону
from scipy.stats import pointbiserialr #Взаимосвязь категориального и числового признаков

pointbiserialr(df['gender'], df['total_duration'])
pd.crosstab(df['phone_value'], df['target'])#построив кросс-таблицу можно увидеть,что чем старше человек, тем дороже его смартфон
sns.pairplot(df[['target', 'phone_value']])
from scipy.stats import pearsonr, spearmanr, kendalltau

r = pearsonr(df['target'], df['phone_value'])

print('Pearson correlation:', r[0], 'p-value:', r[1]) #так как p-value > 0.05, то взаимосвязь статистически не значима 
pd.crosstab(df['car'], df['target'])#построим кросс-таблицу
df.groupby('target')['car'].sum().plot(kind='bar')#на графике можно заметить, что наша гипотеза не подтвердилась

                                                #но видим,что преимуществено авто есть у людей среднего возраста.
from scipy.stats import chi2_contingency, fisher_exact

chi2_contingency(pd.crosstab(df['car'], df['target']))