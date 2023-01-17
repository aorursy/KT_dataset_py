# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import numpy as np

%matplotlib notebook

import matplotlib.pyplot as plt

import collections as coll

import pandas as pd



#alcohol = pd.read_csv("Documents\JADS\Introduction to Data science JBP010\student-mat.csv")

alcohol = pd.read_csv('../input/student-mat.csv')



FQ = alcohol['famrel']

FS = alcohol['famsup']

DALC = alcohol['Dalc']

WALC = alcohol['Walc']

weekconsumption = WALC+DALC



## count cohabitation

COHAB = alcohol['Pstatus']

COHAB_count = (coll.Counter(COHAB))

print (COHAB_count)



## Daily alcohol consumption count

DALC_count = coll.Counter(DALC)

print (DALC_count)



## Weekend alcohol consumption count

WALC_count = coll.Counter(WALC)

print (WALC_count)



## Family Relation counter

FQ_count = coll.Counter (FQ)

print (FQ_count)



FS_count = coll.Counter (FS)

print (FS_count)



## correlations



FS_num = pd.Series(np.where(FS.values == 'yes', 1, 0),

          FS.index)

COHABNUM = pd.Series(np.where(COHAB.values == 'T', 1, 0),

          COHAB.index)



alcohol1 = pd.DataFrame(alcohol)

alcohol1_num_famsup = pd.DataFrame({'FS_num' :FS_num})

alcohol2  = alcohol1.join(alcohol1_num_famsup)

alcohol2_cohabitation = pd.DataFrame({'COHAB_num':COHABNUM})

alcohol3 = alcohol2.join(alcohol2_cohabitation)

alcohol3_weekconsumption = pd.DataFrame({'wkc':weekconsumption })

alcohol4 = alcohol3.join(alcohol3_weekconsumption)



corr_FS_FQ_DALC_WALC_weekconsumtion = pd.DataFrame(alcohol4, columns = ['famrel', 'FS_num','COHAB_num', 'Dalc', 'Walc', 'wkc'])

print(corr_FS_FQ_DALC_WALC_weekconsumtion.corr())

print(corr_FS_FQ_DALC_WALC_weekconsumtion.cov())



##describe

print(pd.DataFrame(alcohol4, columns = ['famrel', 'FS_num','COHAB_num', 'Dalc', 'Walc', 'wkc']).describe())



## Family Relation Histogram

tick_val = [1,2,3,4,5]

plt.xticks(tick_val)

bins = np.arange(10) - 0.5

plt.hist(FQ, bins)

plt.xticks(range(6))

plt.xlim([0, 6])

plt.show()



## Family support Histogram

plot_FS = pd.DataFrame.from_dict (FS_count, orient = 'index')

plot_FS.plot(kind = 'bar')



## Cohabitation Histrogram

plot_cohab = pd.DataFrame.from_dict (COHAB_count, orient = 'index')

plot_cohab.plot(kind = 'bar')
