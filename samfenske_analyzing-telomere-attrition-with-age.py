# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_columns', None)

telomere=pd.read_excel('/kaggle/input/Age_TL_Ecalamita.xlsx')

telomere
telomere.plot(x='age_at_first_capture',y='TSratio_at_first_capture',kind='scatter')
import seaborn as sns

import matplotlib.pyplot as plt

sns.lmplot(x='age_at_first_capture', y='TSratio_at_first_capture', data=telomere, fit_reg=True)

plt.gca().set_title("Telomere length vs. age")
sns.lmplot(x='age_at_first_capture', y='TSratio_at_first_capture', data=telomere,hue='Sex', fit_reg=True)
male=telomere[telomere['Sex'].isin(['male'])]

male.corr()['age_at_first_capture']['TSratio_at_first_capture']
female=telomere[telomere['Sex'].isin(['female'])]

female.corr()['age_at_first_capture']['TSratio_at_first_capture']
#log scale on y axis

telomere['log TSratio_at_first_capture']=np.log(telomere['TSratio_at_first_capture'])

sns.lmplot(x='age_at_first_capture', y='log TSratio_at_first_capture', data=telomere,hue='Sex', fit_reg=True)
sns.lmplot(x='age_at_first_capture', y='svl_capture', data=telomere,hue='Sex', fit_reg=True)
#mass over length

sns.lmplot(x='body_condition', y='log TSratio_at_first_capture', data=telomere,hue='Sex', fit_reg=True)