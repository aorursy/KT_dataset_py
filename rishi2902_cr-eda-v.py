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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



filename = '/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'

df = pd.read_csv(filename)



# Descriptive Analysis

pd.set_option('display.width', 1000)

pd.set_option('display.max_columns', 20)

df.set_index('sl_no', inplace=True)

print('Peek of data : \n', df.head(20))

print('Dimensions of the data: ', df.shape)

print('Descriptive Statistics :')

print(df.describe(include='all'))

print('Null instances in the Features :')

print(pd.isnull(df).sum())

print('Attributes of the data : ', df.dtypes)

print('Name of the features : ', df.columns)

print('Correlations between the attributes : ')

print(df.corr(method='pearson'))



sta_map = {'Not Placed': 0, 'Placed': 1}

df['status'] = df['status'].map(sta_map)







# Data visualization

# converting status feature in numbers





# A. Gender



sns.barplot(x='gender', y='status', data=df)

plt.show()

print('Percentage of male placed : ', end='')

print(df.status[df['gender'] == 'M'].value_counts(normalize=True)[1]*100)

print('Percentage of female placed : ', end='')

print(df.status[df['gender'] == 'F'].value_counts(normalize=True)[1]*100)

# Males are more likely to be placed than females



# B. ssc percentage



bins = [0, 40, 50, 60, 70, 80, 90, 100]

labels = ['F', 'E', 'D', 'C', 'B', 'A', 'Ex']

df['ssc_p_gp'] = pd.cut(df['ssc_p'], bins, labels=labels)



sns.barplot(x='ssc_p_gp', y='status', data=df)

plt.show()



# students are more percentage in ssc are more likely to be placed



# C. hsc percentage

bins = [0, 40, 50, 60, 70, 80, 90, 100]

labels = ['F', 'E', 'D', 'C', 'B', 'A', 'Ex']

df['hsc_p_gp'] = pd.cut(df['hsc_p'], bins, labels=labels)



sns.barplot(x='hsc_p_gp', y='status', data=df)

plt.show()

# students are more percentage in ssc are more likely to be placed



# D. hsc_s

sns.barplot(x='hsc_s', y='status', data=df)

plt.show()

print('Percentage of hsc_s commerce:', end='')

print(df.status[df['hsc_s'] == 'Commerce'].value_counts(normalize=True)[1]*100)

print('Percentage of hsc_s science:', end='')

print(df.status[df['hsc_s'] == 'Science'].value_counts(normalize=True)[1]*100)

print('Percentage of hsc_s arts:', end='')

print(df.status[df['hsc_s'] == 'Arts'].value_counts(normalize=True)[1]*100)



# E. degree  percentage

bins = [0, 40, 50, 60, 70, 80, 90, 100]

labels = ['F', 'E', 'D', 'C', 'B', 'A', 'Ex']

df['degree_p_gp'] = pd.cut(df['degree_p'], bins, labels=labels)



sns.barplot(x='degree_p_gp', y='status', data=df)

plt.show()



# F. degree type

sns.barplot(x='degree_t', y='status', data=df)

plt.show()

print('Percentage of Science students got placed :', end='')

print(df.status[df['degree_t'] == 'Sci&Tech'].value_counts(normalize=True)[1]*100)

print('Percentage of Comm. and Mgmt. students got placed :', end='')

print(df.status[df['degree_t'] == 'Comm&Mgmt'].value_counts(normalize=True)[1]*100)

print('Percentage of others got placed :', end='')

print(df.status[df['degree_t'] == 'Others'].value_counts(normalize=True)[1]*100)







# G. Work Experience

sns.barplot(x='workex', y='status', data=df)

plt.show()

# Students with work experience are more likely to be placed



# H. Employability test percentage ( conducted by college)

bins = [0, 40, 50, 60, 70, 80, 90, 100]

labels = ['F', 'E', 'D', 'C', 'B', 'A', 'Ex']

df['etest_p_gp'] = pd.cut(df['etest_p'], bins, labels=labels)



sns.barplot(x='etest_p_gp', y='status', data=df)

plt.show()



# Students with more etest percentage are more likely to be placed



# I. specialisation

sns.barplot(x='specialisation', y='status', data=df)

plt.show()



# J. mba_p

bins = [0, 40, 50, 60, 70, 80, 90, 100]

labels = ['F', 'E', 'D', 'C', 'B', 'A', 'Ex']

df['mba_p_gp'] = pd.cut(df['mba_p'], bins, labels=labels)



sns.barplot(x='mba_p_gp', y='status', data=df)

plt.show()



print('Overall observation from data visualization : ')

print('1. Gender: Males are more likely to be placed.')

print('2. Students with higher ssc percentage are more likely to be placed')

print('3. Students with higher hsc percentage are more likely to be placed')

print('4. Students with higher degree percentage are more likely to be placed')

print('5. Sci&tech and com&mgmt have more chances of getting placed than others')

print('6. Students with work experience are more likely to be placed')

print('7. Students with more etest percentage are more likely to be placed')

print('8. MKt&Fin students are more likely to be placed')

print('9. Students with higher mba percentage are more likely to be placed')