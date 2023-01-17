# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading the csv file with the train data

data = pd.read_csv(dirname+'/train.csv', sep=',', header=0, index_col=0)

#train_label = pd.read_csv('train_labels.csv', sep=',', header=0,index_col=0)

data.info()
# Define some global variables we will use in our quick EDA

numerical_vars=['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',

                'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm',

                'Horizontal_Distance_To_Fire_Points']

#Columns for the categorical variable Wilderness

n_cat_for_wild=4

cat_var_wild=['Wilderness_Area'+str(i) for i in range(1,n_cat_for_wild+1)]

#Columns for the categorical variable Soil

n_cat_for_soil=40

cat_var_soil=['Soil_Type'+str(i) for i in range(1,n_cat_for_soil+1)]

# Descriptive statistics for numerical variables

# First insight of data

data[numerical_vars].describe().T
# Lets explore how many 0 values are in the features

print('Aspect=0 in rows:'+str(len(data[data['Aspect']==0].index)))

print('Slope=0 in rows:'+str(len(data[data['Slope']==0].index)))

print('Horizontal_Distance_To_Hydrology=0 in rows:'+str(len(data[data['Horizontal_Distance_To_Hydrology']==0].index)))

print('Vertical_Distance_To_Hydrology=0 in rows:'+str(len(data[data['Vertical_Distance_To_Hydrology']<=0].index)))

print('Horizontal_Distance_To_Roadways=0 in rows:'+str(len(data[data['Horizontal_Distance_To_Roadways']<=0].index)))
#We are defining an 3 x 4 matrix to help us plotting features

plot_vars=np.array([['Elevation','Aspect','Slope',None],['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',

                'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points'],['Hillshade_9am','Hillshade_Noon','Hillshade_3pm',None]])

# Boxplots for numrical features

f, axes = plt.subplots(3, 4, sharey=False, figsize=(15,15))

#fig1, ax1 = plt.subplots()

for i in range(plot_vars.shape[0]):

    for j in range(plot_vars.shape[1]):

        if plot_vars[i,j]!=None:

            axes[i,j].set_title(plot_vars[i,j])

            axes[i,j].grid(True)

            axes[i,j].tick_params(

                axis='x',          # changes apply to the x-axis

                which='both',      # both major and minor ticks are affected

                bottom=False,      # ticks along the bottom edge are off

                top=False,         # ticks along the top edge are off

                labelbottom=False)

            axes[i,j].boxplot(data[plot_vars[i,j]])

        else:

            axes[i,j].set_visible(False)



plt.show()
# Histograms for appicant featrures

f, axes = plt.subplots(3, 4, sharey=False, figsize=(15,15))

#fig1, ax1 = plt.subplots()

for i in range(plot_vars.shape[0]):

    for j in range(plot_vars.shape[1]):

        if plot_vars[i,j]!=None:

            axes[i,j].set_title(plot_vars[i,j])

            axes[i,j].grid(False)

            axes[i,j].tick_params(

                axis='x',          # changes apply to the x-axis

                which='both',      # both major and minor ticks are affected

                bottom=False,      # ticks along the bottom edge are off

                top=False,         # ticks along the top edge are off

                labelbottom=False)

            axes[i,j].hist(data[plot_vars[i,j]],bins=20)

        else:

            axes[i,j].set_visible(False)



plt.show()
import seaborn as sns



sns.pairplot(data[numerical_vars+['Cover_Type']], vars = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',

                'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points'],hue='Cover_Type',plot_kws = {'alpha': 0.6})

plt.show()
g=sns.pairplot(data[numerical_vars+['Cover_Type']], vars = ['Hillshade_9am','Hillshade_Noon','Hillshade_3pm'],hue='Cover_Type',plot_kws = {'alpha': 0.6})



g = g.add_legend()

plt.show()

plt.subplots(figsize=(20, 16))

k = 50 #number of variables for heatmap

corrmat = data.corr()

cols = corrmat.nlargest(k, 'Cover_Type')['Cover_Type'].index



cm = np.corrcoef(data[cols].values.T)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True,

                 fmt='.2f', annot_kws={'size': 8}, cmap='Blues',

                 yticklabels=[x[0:10] for x in cols.values], xticklabels=cols.values)

plt.show()
root_label='Soil_Type'

labels=[i[len(root_label):] for i in cat_var_soil]

f, ax = plt.subplots(figsize=(20,5))

data[cat_var_soil].sum().plot(kind='bar',label='Soil Type',color='g')

ax.set_xticklabels(labels)

ax.set_xlabel('Soil Type',labelpad =15,fontsize=20)



root_label='Wilderness_Area'

labels=[i[len(root_label):] for i in cat_var_wild]



f, ax = plt.subplots(figsize=(20,5))

data[cat_var_wild].sum().plot(kind='bar',label='Wilderness Area',color='brown')

ax.set_xticklabels(labels)

ax.set_xlabel('Wilderness Area',labelpad =15,fontsize=20)



plt.show()
f, ax = plt.subplots(figsize=(20,5))

data['Cover_Type'].value_counts().plot(kind='bar',label='Cover type',color='orange')

ax.set_xlabel('Cover Type',labelpad =15,fontsize=20)

plt.show()
# Histograms features vs target label

values=data['Cover_Type'].unique()

values.sort()



f, axes = plt.subplots(3, 4, sharey=False, figsize=(15,15))

#fig1, ax1 = plt.subplots()

for i in range(plot_vars.shape[0]):

    for j in range(plot_vars.shape[1]):

        if plot_vars[i,j]!=None:

            for k in values:

                axes[i,j].hist(data[data['Cover_Type']==k][plot_vars[i,j]],alpha=0.7,histtype='bar',bins=10,label=str(k))

                

            axes[i,j].set_xlabel(plot_vars[i,j],labelpad=10,fontsize=10)

            if j==0:

                axes[i,j].set_ylabel('Count')

                axes[i,j].legend()

        else:

            axes[i,j].set_visible(False)



plt.show()
Area1 = data[data['Wilderness_Area1']==1]

Area2 = data[data['Wilderness_Area2']==1]

Area3 = data[data['Wilderness_Area3']==1]

Area4 = data[data['Wilderness_Area4']==1]



f, ax = plt.subplots(figsize=(15,8))

plt.hist(Area1['Cover_Type'],label='Area1',histtype='step',alpha=0.7,linewidth=5)

plt.hist(Area2['Cover_Type'],label='Area2',histtype='step',alpha=0.7,linewidth=5)

plt.hist(Area3['Cover_Type'],label='Area3',histtype='step',alpha=0.7,linewidth=5)

plt.hist(Area4['Cover_Type'],label='Area4',histtype='step',alpha=0.7,linewidth=5)

plt.title('Wilderness Area vs Cover Type',fontsize=16)

plt.xlabel('Cover Type', fontsize=12)

plt.legend()

plt.show()
plt.figure(figsize =(20,10)) 



for i in cat_var_soil:

    plt.hist(data[data[i]==1]['Cover_Type'],label=i[9:],histtype='bar',alpha=0.7,linewidth=5)

    

plt.title('Soil Type vs Cover Type',fontsize=16)

plt.xlabel('Cover Type', fontsize=12)

plt.legend()

plt.show()