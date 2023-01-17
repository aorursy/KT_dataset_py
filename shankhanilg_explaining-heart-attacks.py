# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings



warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Importing the data



data = pd.read_csv('../input/heart.csv')



data.head(10)
# getting the columns in the dataset



data.columns
#exploring the data statistically



data.describe()
# Understanding the influence of Age on heart diseases, coupled with gender



import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(1, figsize = (15,6))

n = 0

for gender in [0,1]:

    n = n + 1

    plt.subplot(1,2, n)

    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

    for c in range(0,4):

        sns.distplot(data['age'][data['cp'] == c][data['sex'] == gender], 

                     hist = False, rug = True, label = 'Type {} chest pain'.format(c))

    plt.title('Age distribution plot for different types of heart disease wrt {}'.

              format('Male' if gender == 0 else 'Female'))

plt.legend()

plt.show()                    
# Visualizing the distribution of parameter values wrt presence of heart disease



param = ['trestbps', 'chol', 'thalach','oldpeak']



plt.figure (1, figsize = (15,10))

n = 0

for p in param:

    n = n + 1

    plt.subplot(2,2, n)

    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

    for ctype in range(0,4):

            sns.distplot(data[p][data['cp'] == ctype], 

                     hist = False, rug = True, label = "{} type pain".format(ctype))

    plt.title("Distribution plot for {} wrt to type of Heart disease".format(p))

plt.legend()

plt.show() 
# Determining the influence of gender on heart diseases using the same parameters



param = ['trestbps', 'chol', 'thalach','oldpeak'] 



plt.figure (1, figsize = (25,15))

n = 0

for p in param:

    for gender in [0, 1]:

        n = n + 1

        plt.subplot(4,2, n)

        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

        for ctype in range(0,4):

            sns.distplot(data[p][data['cp'] == ctype][data['sex'] == gender], 

                     hist = False, rug = True, label = "{} type pain".format(ctype))

        plt.title("Distribution plot for {} wrt to type of Heart disease in {}".

                  format(p, 'Male' if gender == 0 else 'Female'))

plt.legend()

plt.show()
# Visualizing the count plots of the different categorical variables wrt of gender and chest pain type

param = ['fbs', 'restecg','exang', 'slope', 'ca', 'thal']



plt.figure(1, figsize = (15,6))

n = 0

#for p in param:

#for cp in range(0,4):

    

for gender in [0, 1]:

    #n = n + 1 

    #plt.subplot(1,2, n)

    #plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

    

    pd.crosstab(data['cp'][data['sex'] == gender],

                data[param[0]][data['sex'] == gender]).plot(kind="bar", figsize = (5,5))

    plt.xlabel('Chest Pain')

    plt.ylabel('Count')

    plt.title('Bar plot of {} for {} wrt chest pain type'.format(param[0], 'Male' if gender == 0 else 'Female'))
# Visualizing the count plots of the different categorical variables wrt of gender and chest pain type

param = ['fbs', 'restecg','exang', 'slope', 'ca', 'thal']



plt.figure(1, figsize = (15,6))

n = 0

#for p in param:

#for cp in range(0,4):

    

for gender in [0, 1]:

    n = n + 1

    pd.crosstab(data['cp'][data['sex'] == gender],

                data[param[1]][data['sex'] == gender]).plot(kind="bar", figsize = (5,5))

    plt.xlabel('Chest Pain')

    plt.ylabel('Count')

    plt.title('Bar plot of {} for {} wrt chest pain type'.format(param[1], 'Male' if gender == 0 else 'Female'))
# Visualizing the count plots of the different categorical variables wrt of gender and chest pain type

param = ['fbs', 'restecg','exang', 'slope', 'ca', 'thal']



plt.figure(1, figsize = (15,6))

n = 0

#for p in param:

#for cp in range(0,4):

    

for gender in [0, 1]:

    n = n + 1

    pd.crosstab(data['cp'][data['sex'] == gender],

                data[param[2]][data['sex'] == gender]).plot(kind="bar", figsize = (5,5))

    plt.xlabel('Chest Pain')

    plt.ylabel('Count')

    plt.title('Bar plot of {} for {} wrt chest pain type'.format(param[2], 'Male' if gender == 0 else 'Female'))
# Visualizing the count plots of the different categorical variables wrt of gender and chest pain type

param = ['fbs', 'restecg','exang', 'slope', 'ca', 'thal']



plt.figure(1, figsize = (15,6))

n = 0

#for p in param:

#for cp in range(0,4):

    

for gender in [0, 1]:

    n = n + 1

    pd.crosstab(data['cp'][data['sex'] == gender],

                data[param[3]][data['sex'] == gender]).plot(kind="bar", figsize = (5,5))

    plt.xlabel('Chest Pain')

    plt.ylabel('Count')

    plt.title('Bar plot of {} for {} wrt chest pain type'.format(param[3], 'Male' if gender == 0 else 'Female'))
# Visualizing the count plots of the different categorical variables wrt of gender and chest pain type

param = ['fbs', 'restecg','exang', 'slope', 'ca', 'thal']



plt.figure(1, figsize = (15,6))

n = 0

#for p in param:

#for cp in range(0,4):

    

for gender in [0, 1]:

    n = n + 1

    pd.crosstab(data['cp'][data['sex'] == gender],

                data[param[4]][data['sex'] == gender]).plot(kind="bar", figsize = (5,5))

    plt.xlabel('Chest Pain')

    plt.ylabel('Count')

    plt.title('Bar plot of {} for {} wrt chest pain type'.format(param[4], 'Male' if gender == 0 else 'Female'))
# Visualizing the count plots of the different categorical variables wrt of gender and chest pain type

param = ['fbs', 'restecg','exang', 'slope', 'ca', 'thal']



plt.figure(1, figsize = (15,6))

n = 0

#for p in param:

#for cp in range(0,4):

    

for gender in [0, 1]:

    n = n + 1

    pd.crosstab(data['cp'][data['sex'] == gender],

                data[param[5]][data['sex'] == gender]).plot(kind="bar", figsize = (5,5))

    plt.xlabel('Chest Pain')

    plt.ylabel('Count')

    plt.title('Bar plot of {} for {} wrt chest pain type'.format(param[5], 'Male' if gender == 0 else 'Female'))
_middleAge = data [data['age'] <=55][data['age'] >35]

print(

    "There are {} no of entries that belong to the asset population, out of {} total entries".

    format(_middleAge.shape, data.shape))

_middleAge.head()
# count plot of the types of heart disease they selected popuation have, wrt gender



plt.figure(1, figsize = (15,6))

n = 0

for gender in [0,1]:

    n = n + 1

    plt.subplot(1,2, n)

    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

    sns.countplot(x = 'cp', data = _middleAge[_middleAge['sex'] == gender])

    plt.title('Count plot of Chest pain type in {}'.format('Male' if gender == 0 else 'Female'))

plt.show()
# Visualizing the effect of having oldpeak < 2 and thalach > 150

param = ['oldpeak', 'thalach']



_middleAge_men = _middleAge[_middleAge['sex'] == 0]



plt.figure (1, figsize = (15,6))



plt.subplot( 2, 2, 1)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



for cp in range(0,4):

    sns.distplot(_middleAge_men['oldpeak'][_middleAge_men['oldpeak'] < 2]

                 [_middleAge_men['cp'] == cp], hist = False, rug = True,

                label = "Type {} heart disase".format(cp))

plt.title("Probability distribution of oldpeak < 2 for males wrt chest pain type")

plt.subplot( 2, 2, 2)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



for cp in range(0,4):

    sns.distplot(_middleAge_men['thalach'][_middleAge_men['thalach'] > 150]

                 [_middleAge_men['cp'] == cp], hist = False, rug = True,

                label = "Type {} heart disase".format(cp))

plt.title("Probability distribution of thalach > 150 for males wrt chest pain type")



plt.subplot( 2, 2, 3)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_men[_middleAge_men['oldpeak'] < 2]) 

plt.title("Count plot of oldpeak < 2 for males wrt chest pain type")



plt.subplot( 2, 2, 4)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_men[_middleAge_men['thalach'] > 150])

plt.title("Count plot of thalach > 150 for males wrt chest pain type")



#plt.subplot( 2, 2, 4)

#plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



plt.show()
# visualizing [fbs = 0, resteccg = {0,1} , exang = 0, slope = 1, ca = 0, thal = 2] in male

plt.figure (1, figsize = (12,10))



plt.subplot( 3, 2, 1)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_men[_middleAge_men['fbs'] == 0])

plt.title("Count plot of fbs = 0 for males wrt chest pain type")





plt.subplot( 3, 2, 2)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_men[_middleAge_men['restecg'] <= 1])

plt.title("Count plot of restecg = {0,1} for males wrt chest pain type")





plt.subplot( 3, 2, 3)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_men[_middleAge_men['exang'] == 0])

plt.title("Count plot of exang = 0 for males wrt chest pain type")





plt.subplot( 3, 2, 4)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_men[_middleAge_men['slope'] == 1])

plt.title("Count plot of slope = 1 for males wrt chest pain type")





plt.subplot( 3, 2, 5)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_men[_middleAge_men['ca'] == 0])

plt.title("Count plot of ca = 0 for males wrt chest pain type")





plt.subplot( 3, 2, 6)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_men[_middleAge_men['thal'] == 2])

plt.title("Count plot of thal = 2 for males wrt chest pain type")
# similar visualizations for females

# Visualizing the effect of having oldpeak < 2 and thalach > 150

param = ['oldpeak', 'thalach']



_middleAge_women = _middleAge[_middleAge['sex'] == 1]



plt.figure (1, figsize = (15,6))



plt.subplot( 2, 2, 1)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



for cp in range(0,4):

    sns.distplot(_middleAge_women['oldpeak'][_middleAge_women['oldpeak'] < 2]

                 [_middleAge_women['cp'] == cp], hist = False, rug = True,

                label = "Type {} heart disase".format(cp))

plt.title("Probability distribution of oldpeak < 2 for females wrt chest pain type")

plt.subplot( 2, 2, 2)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



for cp in range(0,4):

    sns.distplot(_middleAge_women['thalach'][_middleAge_women['thalach'] > 150]

                 [_middleAge_women['cp'] == cp], hist = False, rug = True,

                label = "Type {} heart disase".format(cp))

plt.title("Probability distribution of thalach > 150 for females wrt chest pain type")



plt.subplot( 2, 2, 3)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_women[_middleAge_women['oldpeak'] < 2]) 

plt.title("Count plot of oldpeak < 2 for females wrt chest pain type")



plt.subplot( 2, 2, 4)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_women[_middleAge_women['thalach'] > 150])

plt.title("Count plot of thalach > 150 for females wrt chest pain type")



#plt.subplot( 2, 2, 4)

#plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



plt.show()
# visualizing [fbs = 0, resteccg = {0,1} , exang = 0, slope = 1, ca = 0, thal = 2] in female

plt.figure (1, figsize = (12,10))



plt.subplot( 3, 2, 1)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_women[_middleAge_women['fbs'] == 0])

plt.title("Count plot of fbs = 0 for females wrt chest pain type")





plt.subplot( 3, 2, 2)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_women[_middleAge_women['restecg'] <= 1])

plt.title("Count plot of restecg = {0,1} for females wrt chest pain type")





plt.subplot( 3, 2, 3)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_women[_middleAge_women['exang'] == 0])

plt.title("Count plot of exang = 0 for females wrt chest pain type")





plt.subplot( 3, 2, 4)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_women[_middleAge_women['slope'] == 1])

plt.title("Count plot of slope = 1 for females wrt chest pain type")





plt.subplot( 3, 2, 5)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_women[_middleAge_women['ca'] == 0])

plt.title("Count plot of ca = 0 for females wrt chest pain type")





plt.subplot( 3, 2, 6)

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



sns.countplot(x = 'cp', data = _middleAge_women[_middleAge_women['thal'] == 2])

plt.title("Count plot of thal = 2 for females wrt chest pain type")
#Generating heat map between the given set of attributes

plt.figure (1, figsize = (15,10))

sns.heatmap(_middleAge_men.corr(), annot = True, cmap = 'Reds')
# creating dummy variables for categorical variables

# we have the following categorical variables : fbs, restecg, exang, ca, thal, slope

_middleAge_men