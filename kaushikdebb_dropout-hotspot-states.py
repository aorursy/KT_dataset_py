# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dropOut_df = pd.read_csv('/kaggle/input/indian-school-education-statistics/dropout-ratio-2012-2015.csv')
dropOut_df.groupby(['State_UT'])['year'].count()
#Arunachal  Pradesh --> Arunachal Pradesh

#Madhya  Pradesh ---> Madhya Pradesh

#Tamil  Nadu ----> Tamil Nadu



dropOut_df['State_UT'] = dropOut_df['State_UT'].apply(lambda x: "Arunachal Pradesh" if x == 'Arunachal  Pradesh' else x)

dropOut_df['State_UT'] = dropOut_df['State_UT'].apply(lambda x: "Madhya Pradesh" if x == 'Madhya  Pradesh' else x)

dropOut_df['State_UT'] = dropOut_df['State_UT'].apply(lambda x: "Tamil Nadu" if x == 'Tamil  Nadu' else x)



#Lets check the state wise row count..

dropOut_df.groupby(['State_UT'])['year'].count()
dropOut_df.info()
dropOut_df.head(50)
school_category = ['Primary_Boys', 'Primary_Girls', 'Primary_Total',

       'Upper Primary_Boys', 'Upper Primary_Girls', 'Upper Primary_Total',

       'Secondary _Boys', 'Secondary _Girls', 'Secondary _Total',

       'HrSecondary_Boys', 'HrSecondary_Girls', 'HrSecondary_Total']



print("shape before = ", dropOut_df.shape)





for cat in school_category:

    

    if "NR" in list(dropOut_df[cat].unique()):

        dropOut_df = dropOut_df[dropOut_df[cat] != 'NR']

    if "Uppe_r_Primary" in list(dropOut_df[cat].unique()):

        dropOut_df = dropOut_df[dropOut_df[cat] != 'Uppe_r_Primary']

    dropOut_df[cat] = dropOut_df[cat].astype(float)

    

    

    

print("shape before = ", dropOut_df.shape)



#dropOut_df.groupby(['State_UT'])['Primary_Boys'].count()

temp_df_list = []

for cat in school_category:

    temp_df_list.append(pd.DataFrame({cat+'_mean': dropOut_df.groupby(['State_UT'])[cat].mean()}))

    

mean_df = temp_df_list[0]

mean_df.reset_index(inplace = True)



for i in range(1, len(temp_df_list)):

    temp_df_list[i].reset_index(inplace = True)

    mean_df = pd.merge(mean_df, temp_df_list[i], on = 'State_UT')

    

mean_df.set_index('State_UT', inplace = True)
plt.figure(figsize=(10, 20))

sns.heatmap(mean_df, annot = True)