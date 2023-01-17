#import necessary libraries

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
cereal = pd.read_csv('../input/80-cereals/cereal.csv')

cereal.head()
cereal['type'].unique()
#graph distribution of quantitative data

plt.figure(figsize=[8,6])

plt.hist(x = [cereal[cereal['type']=='C']['sodium'], cereal[cereal['type']=='H']['sodium']], 

         stacked=True, color = ['b','y'],label = ['Cold','Hot'])

plt.title('Sodium Histogram by type of cereal cold/hot')

plt.xlabel('Cold / Hot')

plt.ylabel('sodium')

plt.legend()
hot = cereal[cereal['type']=='H']['sodium']

cold = cereal[cereal['type']=='C']['sodium']

#perform t-test

ttest_ind(hot, cold, equal_var = False)
x = cereal['sodium']

y = cereal['rating']

#perform t-test

ttest_ind(x, y, equal_var = False)
#graph distribution of quantitative data

plt.figure(figsize=[8,6])

plt.hist(x = [cereal[cereal['type']=='H']['rating'], cereal[cereal['type']=='C']['rating']], 

         stacked=True, color = ['r','b'],label = ['Hot','Cold'])

plt.title('Rating Histogram by type of cereal cold/hot')

plt.xlabel('Cold / Hot')

plt.ylabel('Rating')

plt.legend()
hot = cereal[cereal['type']=='H']['rating']

cold = cereal[cereal['type']=='C']['rating']

#perform t-test

ttest_ind(hot, cold, equal_var = False)
#graph distribution of quantitative data

plt.figure(figsize=[8,6])

plt.hist(x = [cereal[cereal['type']=='H']['carbo'], cereal[cereal['type']=='C']['carbo']], 

         stacked=True, color = ['r','b'],label = ['Hot','Cold'])

plt.title('Carbo Histogram by type of cereal cold/hot')

plt.xlabel('Cold / Hot')

plt.ylabel('Carbo')

plt.legend()
hot = cereal[cereal['type']=='H']['carbo']

cold = cereal[cereal['type']=='C']['carbo']

#perform t-test

ttest_ind(hot, cold, equal_var = False)
x = cereal['carbo']

y = cereal['rating']

#perform t-test

ttest_ind(x, y, equal_var = False)
import seaborn as sns 

import matplotlib.pyplot as plt

plt.subplots(figsize=(10,8))

sns.countplot(x="protein",data=cereal,hue = "type").set_title("Protein plot by type of cereal cold/hot")
cereal['mfr'].unique()
import seaborn as sns 

import matplotlib.pyplot as plt

plt.subplots(figsize=(8,6))

sns.countplot(x="mfr",data=cereal).set_title("MFR plot")
#graph distribution of quantitative data

plt.figure(figsize=[8,6])

plt.hist(x = [cereal[cereal['mfr']=='N']['rating'], cereal[cereal['mfr']=='Q']['rating'],cereal[cereal['mfr']=='K']['rating'],

             cereal[cereal['mfr']=='R']['rating'],cereal[cereal['mfr']=='G']['rating'],cereal[cereal['mfr']=='P']['rating'],

             cereal[cereal['mfr']=='A']['rating']], stacked=True,label = ['N','Q','K','R','G','P','A'])

plt.title('Rating Histogram by type of MFR')

plt.xlabel('type of MFR')

plt.ylabel('Rating')

plt.legend()