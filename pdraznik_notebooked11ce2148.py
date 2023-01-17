# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# loading required modules

import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns

import math

from sklearn import svm

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# setting pandas env variables to display max rows and columns

pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows',1000)





# load train and test dataset

print("Loading.....")

train = pd.read_csv("../input/train.csv")

train_y = train['loss']

train.drop(['loss'], axis=1, inplace=True)



test = pd.read_csv("../input/test.csv")

print("Loaded.")
#Recomended: log transform the label variable

train['loss'] = np.log1p(train['loss'])

print(train['loss'])
# sepearte the categorical and continous features

cont_columns = []

cat_columns = []



for i in train.columns:

    if train[i].dtype == 'float':

        cont_columns.append(i)

    elif train[i].dtype == 'object':

        cat_columns.append(i)



cont_columns.remove('loss')



print("Continuous Valued Columns: \n", cont_columns)

print("\n\n")

print("Categorical Valued Columns: \n", cat_columns)
# Optional: Display info



# printing train dataset information

train.info()



print("\n\n")



# printing test dataset information

test.info()



print("\n\n")



train.describe(include = ['object'])
#Plot the loss function

ax = sns.distplot(train['loss'])
#Calculate the correlation between coninuous variables, loss and one another

corr = train[cont_columns+['loss']].corr()



#Display the correlation between the continuous variables, loss, and one another

sns.set(style="white")



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

#Use light/Dark palettes when looking at abs of corr

cmap = sns.light_palette((260, 75, 50), input="husl", as_cmap=True)

#sns.dark_palette((260, 75, 60), input="husl", as_cmap=True)

#sns.light_palette("#2ecc71", as_cmap=True)



#Use diverging palette when looking at corr

#sns.diverging_palette(240, 5, as_cmap=True)



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(

    np.absolute(corr), 

    mask=mask, 

    cmap=cmap, 

    vmax=.3,

    square=True, 

    linewidths=.5, 

    cbar_kws={"shrink": .5}, 

    ax=ax

)
#Plot the two most correlated continuous valued functions against each other

(sns.jointplot(

    x="cont11", 

    y="cont12", 

    data=train[cont_columns],

    #kind="kde", space=0, color="g"

)

.plot_joint(sns.kdeplot, zorder=1, n_levels=6))
#Plot continuous valued functions and loss against each other

sns.pairplot(

    train[cont_columns+['loss']], 

    vars=(cont_columns[8:14]+['loss']), 

    kind = 'scatter',

    diag_kind='kde'

)
#Count the number of options for each categorical variable

options_count = [(x, len(np.unique(train[x], return_counts=True)[1])) for x in train[cat_columns]]



print( options_count )
#Convert the categorical variables to binary

cols = cat_columns[95:100]

train_test = train.copy()



train_cat_columns_new = pd.get_dummies(train_test[cols])

cat_columns_new = list(train_cat_columns_new.columns.values)





train_test.drop(cat_columns, axis=1, inplace=True)

#train_test.drop(cont_columns, axis=1, inplace=True)

#print(train_test)



result = pd.concat([train_test, train_cat_columns_new], axis=1, join_axes=[train_test.index])

print(result)
#Calculate the correlation between categorical variables, loss and one another

corr_cat = result[cat_columns_new+['loss']].corr()



#Display the correlation between categorical variables, loss and one another

sns.set(style="white")



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

#Use light/Dark palettes when looking at abs of corr

cmap = sns.light_palette((260, 75, 50), input="husl", as_cmap=True)

#sns.dark_palette((260, 75, 60), input="husl", as_cmap=True)

#sns.light_palette("#2ecc71", as_cmap=True)



#Use diverging palette when looking at corr

#sns.diverging_palette(240, 5, as_cmap=True)



# Generate a mask for the upper triangle

mask = np.zeros_like(corr_cat, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(

    np.absolute(corr_cat), 

    mask=mask, 

    cmap=cmap, 

    vmax=.3,

    square=True, 

    linewidths=.5, 

    cbar_kws={"shrink": .5}, 

    ax=ax

)
#Count the number of observations for each option of a categorical variable

sns.countplot(

    data=train[cat_columns+['loss']],

    x=cat_columns[1],

)
#Display the median loss (with error) for each category option

sns.barplot(

    data=train[cat_columns+['loss']],

    x=cat_columns[1], 

    y="loss"

);
#Display the violin plot of loss vs. category option (with error) for numerous categories

g = sns.PairGrid(

    train[cat_columns+['loss']],

    x_vars=cat_columns[:6],

    y_vars=["loss"],

)

g.map(

    sns.violinplot, 

    palette="pastel", 

    split=False, 

    inner="stick", 

    bw=.2

);
#
#Perform SVM





#Probability plots of continuous variables

import matplotlib.gridspec as gridspec

from scipy import stats



plt.figure(figsize=(15,25))

gs = gridspec.GridSpec(7, 2)

for i, cn in enumerate(train[cont_columns].columns):

    ax = plt.subplot(gs[i])

    stats.probplot(train[cn], dist = stats.lognorm, plot = ax)

    ax.set_xlabel('')

    ax.set_title('Probplot of feature: cont' + str(i+1))

plt.show()
#Skewness of continuous variables 

skewness_list = []

for cn in train[cont_columns].columns:

    skewness_list.append(stats.skew(train[cn]))



plt.figure(figsize=(10,7))

plt.plot(np.absolute(skewness_list), 'bo-')

plt.xlabel("continous features")

plt.ylabel("skewness")

plt.title("plotting skewness of the continous features")

plt.xticks(range(15), range(1,15,1))

plt.plot([(0.25) for i in range(0,14)], 'r--')

plt.text(6, .1, 'threshold = 0.25')

plt.show()
#Consider only the highly skewed columns

skewed_cont_columns = []

skew_threshold = 0.25

for i, cn in enumerate(cont_columns):

    if np.abs(skewness_list[i]) >= skew_threshold:

        skewed_cont_columns.append(cn)
#Display the highly skewed columns 

plt.figure(figsize=(15,25))

gs = gridspec.GridSpec(6, 2)

for i, cn in enumerate(skewed_cont_columns):

    ax = plt.subplot(gs[i])

    sns.distplot(train[cn], bins=50)

    ax.set_xlabel('')

    ax.set_title('hist plot of feature: ' + str(cn))

plt.show()
'''

Below function comes in handy in plotting the distribution and probability plot side by side and we look at

original feature

custom transformed feature

boxcox transformed feature

in some cases custom transformation might be better than boxcox transformation, let's analyze

'''

def examine_transform(original, transformed):

    plt.figure(figsize=(15,10))

    gs = gridspec.GridSpec(3,2, width_ratios=(1,2))

    

    ax = plt.subplot(gs[0])

    sns.distplot(original, bins=50)

    ax.set_xlabel('')

    ax.set_title('histogram of orignal feature')

    

    ax = plt.subplot(gs[1])

    prob = stats.probplot(original, dist = stats.norm, plot = ax)

    ax.set_xlabel('')

    ax.set_title('Probplot of original feature')

    

    ax = plt.subplot(gs[2])

    sns.distplot(transformed, bins=50)

    ax.set_xlabel('')

    ax.set_title('histogram of transformed feature')

    

    ax = plt.subplot(gs[3])

    prob = stats.probplot(transformed, dist = stats.norm, plot = ax)

    ax.set_xlabel('')

    ax.set_title('Probplot of transformed feature')

    

    # apply boxcox transformation

    xt, _ = stats.boxcox(original)

    ax = plt.subplot(gs[4])

    sns.distplot(xt, bins=50)

    ax.set_xlabel('')

    ax.set_title('histogram of boxcox transformed feature')

    

    ax = plt.subplot(gs[5])

    prob = stats.probplot(xt, dist = stats.norm, plot = ax)

    ax.set_xlabel('')

    ax.set_title('Probplot of boxcox transformed feature')

    

    

    plt.show()
examine_transform(train.cont1, np.power(train.cont1,0.5))