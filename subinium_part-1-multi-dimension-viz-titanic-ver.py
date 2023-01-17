import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib as mpl

import matplotlib.pyplot as plt 

from matplotlib import gridspec # flexible multi figure size

 

import seaborn as sns



print('matplotlib : ', mpl.__version__)

print('seaborn : ', sns.__version__)

print('pandas : ', pd.__version__)



plt.rcParams['figure.dpi'] = 200
# load dataset by pandas's read_csv function

data = pd.read_csv('/kaggle/input/titanic/train.csv')

print(data.shape)
data.head()
import missingno as msno

msno.matrix(data)
import missingno as msno

msno.matrix(data, sort='descending')
data.info()
# count the value first!

survived_count = data['Survived'].value_counts()

print(survived_count)
# No Custom Version

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(survived_count.index, survived_count) # bar(x_label, y_label)

axes[1].pie(survived_count) # pie chart



plt.show()
# Custom Version

fig, axes = plt.subplots(1, 2, figsize=(12, 5))



# Custom Color Pallete



color = ['gray', 'lightgreen']  # To express the meaning of survival

new_xlabel = list(map(str, survived_count.index))



# Axes[0] : Bar Plot Custom

axes[0].bar(new_xlabel, # redefine for categorical x labels 

            survived_count, 

            color=color,# color

            width=0.65, # bar width 

            edgecolor='black', # bar color

            # linewidth=1.5 # edge width

        ) 



axes[0].margins(0.2, 0.2) # margin control (leftright, topbottom)

axes[0].set_xlabel('Survived') # label info



# Axes[0] : Pie Chart Custom

explode = [0, 0.05]



axes[1].pie(survived_count,

            labels=new_xlabel,

            colors=color, # color

            explode=explode, # explode

            textprops={'fontsize': 12, 'fontweight': 'bold'}, # text setting

            autopct='%1.1f%%', # notation

            shadow=True # shadow

           )



fig.suptitle('[Titanic] Bar Plot vs Pie Chart', fontsize=15, fontweight='bold') # figure scale title



plt.show()
categorical_features = ['Sex', 'Embarked','Pclass']



for feature in categorical_features:

    print(f'[{feature}]')

    print(data[feature].value_counts(), '\n')
# Custom Version

fig, axes = plt.subplots(1, 3, figsize=(12, 4))



# Sex

sns.countplot(data['Sex'], ax=axes[0])



# Embarked

sns.countplot(data['Embarked'], ax=axes[1])



# Pclass

sns.countplot(data['Pclass'], ax=axes[2])



    

plt.show()
sns.set_style("whitegrid")



# Custom Version

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)



# Sex

# New Palette

sns.countplot(x='Sex', data=data, ax=axes[0], palette="Set2", edgecolor='black') 



# Embarked

# Fixed Color

sns.countplot(data['Embarked'], ax=axes[1], color='gray', edgecolor='black') 



# Pclass

# Gradient Palette

sns.countplot(data['Pclass'], ax=axes[2], palette="Blues", edgecolor='black') 



# Margin & Label Custom

for ax in axes : 

    ax.margins(0.12, 0.15)

    # you can set axis setting like this

    ax.xaxis.label.set_size(12)

    ax.xaxis.label.set_weight('bold')

    

# figure title    

fig.suptitle('[Titanic] Categorical Distribution', 

             fontsize=16, 

             fontweight='bold',

             x=0.05, y=1.08,

             ha='left' # horizontal alignment

            ) 



plt.tight_layout()

plt.show()
mpl.style.use ('default')
# No custom

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.distplot(data['Age'], ax=axes[0], kde=False)

sns.distplot(data['Fare'],  ax=axes[1], kde=False)

plt.tight_layout()

plt.show()
fig, axes = plt.subplots(1, 2, figsize=(9, 4))



sns.scatterplot(x='Sex', y='Age', data=data, ax=axes[0]) # ax.scatter(data['Sex'], data['Fare'])

sns.scatterplot(x='Sex', y='Age', data=data, ax=axes[1], alpha=0.05)



for ax in axes : ax.margins(0.3, 0.1)

plt.show()
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(9, 10), sharey=True)



sns.stripplot(x='Sex', y='Age', data=data, ax=axes[0][0])

sns.swarmplot(x='Sex', y='Age', data=data, ax=axes[0][1])

sns.violinplot(x='Sex', y='Age', data=data, ax=axes[1][0])

sns.boxplot(x='Sex', y='Age', data=data, ax=axes[1][1])



# Tips for turning multiple plots into loops

# use reshape!

for ax, title in zip(axes.reshape(-1), ['Strip Plot', 'Swarm Plot', 'Violin Plot', 'Box Plot'] ): 

    ax.set_title(title, fontweight='bold')

    

plt.show()