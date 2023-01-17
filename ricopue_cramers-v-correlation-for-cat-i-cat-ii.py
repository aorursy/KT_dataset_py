# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as ss

import matplotlib.pyplot as plt

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



#path_data = "/kaggle/input/cat-in-the-dat-ii/"

#path_train = path_data + "train.csv"

#df = pd.read_csv(path_train)



train_I = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col = "id")

train_II = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv", index_col = "id")
#From https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

def cramers_v(confusion_matrix):

    """ calculate Cramers V statistic for categorial-categorial association.

        uses correction from Bergsma and Wicher,

        Journal of the Korean Statistical Society 42 (2013): 323-328

    """

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum()

    phi2 = chi2 / n

    r, k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))

    rcorr = r - ((r-1)**2)/(n-1)

    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))



def cramers_v_matrix(df,cols):

    result=pd.DataFrame()

    for col1 in cols:

        corr_list=[]

        for col2 in cols:

            confusion_matrix = pd.crosstab(df[col1], df[col2]).values

            cramers=cramers_v(confusion_matrix)

            if col1==col2:

                cramers=0

            corr_list.append(cramers)

        df_temp = pd.DataFrame(corr_list, index =cols,columns =[col1]) 

        result = pd.concat([result, df_temp], axis=1, sort=False)

    return result



def summary(df):

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name', 'dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    return summary

train_I.dropna(inplace=True)

train_II.dropna(inplace=True)

summary_data_Cat_I=summary(train_I)

summary_data_Cat_II=summary(train_II)

cols_Cat_I=summary_data_Cat_I[(summary_data_Cat_I.Uniques <27) & (summary_data_Cat_I.Name!="target")]['Name'].to_list()

cols_Cat_II=summary_data_Cat_II[(summary_data_Cat_II.Uniques <27) & (summary_data_Cat_II.Name!="target")]['Name'].to_list()

cramers_matrix_I=cramers_v_matrix(train_I,cols_Cat_I)

cramers_matrix_II=cramers_v_matrix(train_II,cols_Cat_II)
#Poor correlation coefficient. Values updated just for Plotting.

result=cramers_matrix_I.round(4)

cols=cols_Cat_I



fig = plt.figure(figsize=(30, 15))



ax = fig.add_subplot(111)

ax.set_title('Cat_I')

plt.imshow(result)

ax.set_aspect('equal')



# We want to show all ticks...

ax.set_xticks(np.arange(len(cols)))

ax.set_yticks(np.arange(len(cols)))

# ... and label them with the respective list entries

ax.set_xticklabels(cols)

ax.set_yticklabels(cols)



# Rotate the tick labels and set their alignment.

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

         rotation_mode="anchor")



# Loop over data dimensions and create text annotations.

for i in range(len(cols)):

    for j in range(len(cols)):

        text = ax.text(j, i, result.iloc[i, j],

                       ha="center", va="center", color="w")





plt.show()
#Poor correlation coefficient. Values updated just for Plotting.

result=cramers_matrix_II.round(4)

cols=cols_Cat_II



fig = plt.figure(figsize=(30, 15))



ax = fig.add_subplot(111)

ax.set_title('Cat_II')

plt.imshow(result)

ax.set_aspect('equal')



# We want to show all ticks...

ax.set_xticks(np.arange(len(cols)))

ax.set_yticks(np.arange(len(cols)))

# ... and label them with the respective list entries

ax.set_xticklabels(cols)

ax.set_yticklabels(cols)



# Rotate the tick labels and set their alignment.

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

         rotation_mode="anchor")



# Loop over data dimensions and create text annotations.

for i in range(len(cols)):

    for j in range(len(cols)):

        text = ax.text(j, i, result.iloc[i, j],

                       ha="center", va="center", color="w")





plt.show()