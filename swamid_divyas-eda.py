# Code by Divya Swaminathan

# April 2020



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

df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
print('Shape of the data frame is', df.shape)
print('Description of the data ')

df.describe().transpose()
print('Are there any null values in the data frame ?')

df.isna().sum()
df.isin([0]).sum()
df.Outcome.value_counts()
df_impute = df.copy()


fill_columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']





df_impute.iloc[:,1:6] = df_impute.iloc[:,1:6].mask(df_impute.iloc[:,1:6] == 0)

df_impute.head()
df_impute0 = df_impute[df_impute.Outcome == 0]

df_impute1 = df_impute[df_impute.Outcome == 1]
df_impute0.loc[:,fill_columns]= df_impute0.loc[:,fill_columns].fillna(df_impute0.loc[:,fill_columns].mean())
df_impute1.loc[:,fill_columns]= df_impute1.loc[:,fill_columns].fillna(df_impute1.loc[:,fill_columns].mean())
df_impute0.head()
df_impute1.head()
df_impute = pd.concat([df_impute0,df_impute1],axis = 0)

df_impute.head()

df_impute.tail()
from sklearn.utils import shuffle
df_impute = shuffle(df_impute)
df_impute.groupby('Outcome').mean().transpose(), df.groupby('Outcome').mean().transpose()
import matplotlib.pyplot as plt

plt.style.use('ggplot')



table1 = df.groupby('Outcome').mean().transpose()

table1_impute = df_impute.groupby('Outcome').mean().transpose()



table1.plot(kind = 'barh', xlim = [0,210], ylim = [0,10])

plt.xlabel('Mean Values', size = 10)

plt.ylabel('Features', size = 10)

plt.title('Mean values separated by class: df', size = 20)

plt.show()

table1_impute.plot(kind = 'barh', ylim = [0, 10])

plt.xlabel('Mean Values', size = 10)

plt.ylabel('Features', size= 10)

plt.title('Mean values separated by class for imputed data', size = 10)

plt.show()

# HISTOGRAMS for oringinal df



for cols in df.columns:

    x = df.loc[:,cols]

    plt.hist(x)

    plt.title('Histogram of feature ' +str(cols))

    plt.show()
# Histograms for imputed data.



for cols in df_impute.columns:

    x = df_impute.loc[:,cols]

    plt.hist(x)

    plt.title('Histogram for Feature' +str(cols))

    plt.show()
from scipy.stats import skew
#table_skew = []

for cols in df.columns:

    x = df.loc[:,cols]

    xim = df_impute.loc[:,cols]

    #table_skew.append((''+str(cols), skew(x)))

    print(''+str(cols), skew(x),skew(xim))
x = np.log(df_impute.Insulin)

plt.hist(x, color = 'g')

print(skew(x),max(x)/min(x))
corr_matrix = df.corr()

corr_matrix.style.background_gradient(cmap='coolwarm')



# List the top 15 correlations in descending order.

sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

                 .stack()

                 .sort_values(ascending=False))

print(sol[0:15])
corr_matrix_impute = df_impute.corr()

corr_matrix_impute.style.background_gradient(cmap = 'coolwarm')
# List the top 15 correlations in descending order.

sol = (corr_matrix_impute.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

                 .stack()

                 .sort_values(ascending=False))

print(sol[0:15])