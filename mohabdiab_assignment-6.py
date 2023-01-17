# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from scipy.stats import pearsonr

from scipy.stats import f_oneway

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot  as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
wine = pd.read_csv('../input/wineclass/wineclass.csv')
wine.head()
wine.isna().any()
def is_there_null(dataset):

    null_val = dict(dataset.isna().sum())

    missing_counter = int(0)



    for key, value in null_val.items():

        if value == 0:

            #print('There is no Missing Value Here')

            missing_counter = missing_counter + 1

    #print(missing_counter)

        else:

            print('we have at least one missing value here')

    if missing_counter == len(dataset.columns):

        print('There is no Missing Values in any Column in The wine Data')

    return 

is_there_null(wine)
wine.alcohol_class.unique()
pd.crosstab(wine['Class'], wine['alcohol_class'], margins = True) 
pd.crosstab(wine['alcohol_class'], wine['Class'], margins = False) .plot(kind = 'area')
wine.groupby('Class').size().plot.bar()
pd.DataFrame(wine.alcohol_class.value_counts())

plt.figure(figsize=(15,10))

plt.pie(wine.alcohol_class.value_counts(), labels = ['class 3', 'Class 2', 'Class 4', 'Class 1'], autopct = '%1.1f%%', shadow=True) 

plt.show()
wine.describe()
numeric_features = ['Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium',

'TotalPhenols', 'Flavanoids', 'NonFlavanoidPhenols',

'Proanthocyanins', 'ColorIntensity', 'Hue', 'DilutedWines', 'Proline']
plt.figure(figsize=(20,20))

wine.boxplot(column=['Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium',

'TotalPhenols', 'Flavanoids', 'NonFlavanoidPhenols',

'Proanthocyanins', 'ColorIntensity', 'Hue', 'DilutedWines', 'Proline'])
display(wine[numeric_features].describe(include=[np.number]).T)
wine['Class'].value_counts()
wine['alcohol_class'].value_counts()
sns.set(rc={'figure.figsize':(11.7,8.27)})



f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

 

sns.boxplot(wine.Magnesium, ax=ax_box)

sns.distplot(wine.Magnesium, ax=ax_hist)

plt.axvline(wine.Magnesium.mean(), color='r', linestyle='--')

plt.axvline(wine.Magnesium.median(), color='g', linestyle='-')

plt.axvline(wine.Magnesium.max(), color='black', linestyle='-')

plt.axvline(wine.Magnesium.min(), color='black', linestyle='-')

ax_box.set(xlabel='')

pl=sns.scatterplot(data=wine, x="ColorIntensity", y="Hue")
corr, _ = pearsonr(wine.ColorIntensity, wine.Hue)

print('Pearsons correlation: %.3f' % corr)
pl=sns.scatterplot(data=wine, x="ColorIntensity", y="Hue", hue = 'Class', palette=['green','orange','brown'])
def ANOVA_test(wine_feature):

    anova_output = f_oneway(wine_feature[wine['Class'] == 1],

               wine_feature[wine['Class'] == 2],

               wine_feature[wine['Class'] == 3])

    F_stat = anova_output[0]

    P_val = anova_output[1]

    if P_val > 0.05:

        print('we fail to reject the null hypothesis because the P_value = {} bigger than the cut off value 0.05'.format(P_val))

    else:

        print('we reject the null hypothesis because the P_value = {} smaller than the cut off value 0.05'.format(P_val))

ANOVA_test(wine['ColorIntensity'])

ANOVA_test(wine['Alcohol'])