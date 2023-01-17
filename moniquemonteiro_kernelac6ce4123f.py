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
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
df.columns = [x.lower().strip().replace(' ','_') for x in df.columns]

df.columns
%matplotlib inline
plt.figure(figsize=(15,8))

sns.heatmap(df.isna(),cbar=False)
# Proportion of the dataset that has null values



def find_null_values(threshold_of_null=0):

    null_columns = np.where(df.isna().sum() > threshold_of_null)[0]

    prop = (len(null_columns) / df.shape[1])*100    



    print(f'{round(prop,2)}% ({len(null_columns)}) of columns in the dataset have at least {threshold_of_null} null values.')
find_null_values(df.shape[0]*0.8)
def find_null_values_rows(threshold_of_null=0):

    

    null_rows = np.where(df.isna().sum(axis=1) > threshold_of_null)[0]

    prop = (len(null_rows) / df.shape[0])*100

    

    print(f'{round(prop,2)}% ({len(null_rows)}) of rows in the dataset have at least {threshold_of_null} null values.')
find_null_values_rows(df.shape[1]*0.8)
df.groupby('sars-cov-2_exam_result').count().iloc[:,list(set(np.where(df.groupby('sars-cov-2_exam_result').count() > 500)[1]))]
## Let's find where the we have at least 500 not null values.



df.groupby('sars-cov-2_exam_result').count().iloc[:,list(set(np.where(df.groupby('sars-cov-2_exam_result').count() > 500)[1]))].columns
## Let's use this columns



df.groupby('sars-cov-2_exam_result').count().loc[:,['patient_age_quantile','hematocrit','hemoglobin','platelets',

                                                    'mean_platelet_volume','bordetella_pertussis','influenza_b,_rapid_test',

                                                    'influenza_a,_rapid_test','red_blood_cells','basophils','lymphocytes',

                                                    'leukocytes','mean_corpuscular_hemoglobin_concentration_(mchc)',

                                                    'eosinophils','mean_corpuscular_volume_(mcv)','monocytes',

                                                    'red_blood_cell_distribution_width_(rdw)','respiratory_syncytial_virus',

                                                    'parainfluenza_1','coronavirusnl63','rhinovirus/enterovirus',

                                                    'coronavirus_hku1','parainfluenza_3','chlamydophila_pneumoniae','adenovirus',

                                                    'parainfluenza_4','coronavirus229e','coronavirusoc43','inf_a_h1n1_2009',

                                                    'metapneumovirus','influenza_b,_rapid_test','influenza_a,_rapid_test'

                                                   ]]
df_model = df[['sars-cov-2_exam_result','patient_age_quantile','hematocrit','hemoglobin','platelets','mean_platelet_volume',

               'bordetella_pertussis','influenza_b,_rapid_test','influenza_a,_rapid_test','red_blood_cells','basophils',

               'lymphocytes','leukocytes','eosinophils',

               'mean_corpuscular_volume_(mcv)','monocytes','red_blood_cell_distribution_width_(rdw)',

               'respiratory_syncytial_virus','parainfluenza_1','coronavirusnl63','rhinovirus/enterovirus','coronavirus_hku1',

               'parainfluenza_3','chlamydophila_pneumoniae','adenovirus','parainfluenza_4','coronavirus229e','coronavirusoc43',

               'inf_a_h1n1_2009','metapneumovirus','influenza_b,_rapid_test','influenza_a,_rapid_test']]

df_model.notnull().sum()
df = df_model.dropna(how='any')
## Checking correlations



df_corr = df.assign(target = lambda x: x['sars-cov-2_exam_result'].map({'negative':0,'positive':1}))



plt.figure(figsize=(15,8))

sns.heatmap(df_corr.corr(method='spearman'),vmin= -1, vmax= 1,annot=True)
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import make_column_transformer

from sklearn.model_selection import cross_val_score



df.head()
cat_features = ['bordetella_pertussis','influenza_b,_rapid_test','influenza_a,_rapid_test','chlamydophila_pneumoniae',

                'adenovirus','parainfluenza_4','coronavirus229e','coronavirusoc43','inf_a_h1n1_2009','metapneumovirus']

num_features = ['patient_age_quantile','hematocrit','hemoglobin','platelets','mean_platelet_volume','red_blood_cells','basophils',

                'lymphocytes','leukocytes','eosinophils','mean_corpuscular_volume_(mcv)','monocytes',

                'red_blood_cell_distribution_width_(rdw)']

mkcol = make_column_transformer((StandardScaler(),num_features),(OneHotEncoder(),cat_features),remainder='drop')

x = mkcol.fit_transform(df)

y = df.loc[:,'sars-cov-2_exam_result']
logreg = LogisticRegression(class_weight='balanced')

logreg.fit(x,y)
cross_val_score(logreg,x,y,cv=9).mean()