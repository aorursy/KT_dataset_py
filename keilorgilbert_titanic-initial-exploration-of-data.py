import pandas as pd

from pandas import Series, DataFrame

import numpy as np

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

from scipy import stats
titanic = pd.read_csv('../input/titanic/train.csv')

titanic.head()
titanic.info()
titanic.describe()
#Adding a Count column to make the analysis easier.

titanic['Count']=1



sns.set(style='darkgrid')

fig, axes = plt.subplots(3,3,figsize=(14,14))



titles = ['Survival Count','Count by Passenger Class','Count by Sex',

         'Count by Embarked Location', 'Count by SibSp', 'Count by Parch',

         'Fare Distribution ($)', 'Age Distribution (Yrs)', '']



var = ['Survived','Pclass','Sex','Embarked','SibSp','Parch','Fare','Age']



x_labels = [['Perished','Survived'], ['First','Second','Third'], ['Male','Female'], ['Southampton', 'Cherbourg', 'Queenstown']]



for i, ax in enumerate(axes.flatten()):

    if i in [0,1,2,3,4,5]:

        sns.countplot(titanic[var[i]],ax=ax,palette=sns.color_palette('BrBG_r', 7),

                      edgecolor='darkslategray')

        k=0

        j=[0,1,2,3,4,5,6]

        if i == 1:

            j=[1,2,0]

        elif i == 4:

            j=[0,1,2,4,3,6,5]

        elif i == 5:

            j=[0,1,2,3,5,4,6]

        for p in ax.patches:

            ax.text(p.get_x()+p.get_width()/2, p.get_height()+5,

                    titanic[var[i]].value_counts().iloc[j[k]], ha='center')

            k += 1

    if i in [6,7]:

        sns.distplot(titanic[var[i]],ax=ax, bins=30,

             kde_kws={'color':'darkolivegreen','label':'Kde','gridsize':1000,'linewidth':3},

             hist_kws={'color':'goldenrod','label':"Histogram",'edgecolor':'darkslategray'})

        ax.set_xlim([0, max(titanic[var[i]])])

        ax.set_yticklabels([])

    if i in [8]:

        ax.axis('off')

    if i in [0,1,2,3]:

        ax.set_xticklabels(x_labels[i])

    ax.set_title(label=titles[i], fontsize=16, fontweight='bold', pad=10)

    ax.set_xlabel(None)

    ax.set_ylabel(None)



fig.suptitle('Single Variable Overview', position=(.5,1.04), fontsize=30, fontweight='bold')

fig.tight_layout(h_pad=2)
titanic['Fare'].sort_values(ascending=False).head()
sns.boxplot(y=titanic['Fare'],data=titanic,orient='h')
print ('$512 fare has Z-score = '+str(stats.zscore(titanic['Fare'])[258]))



val = 512.3292

third_qrt = titanic['Fare'].quantile(.5)

iqr = stats.iqr(titanic['Fare'])



print('$512 fare is '+str((val - third_qrt)/iqr)+' IQRs above the third quartile')
titanic = titanic[titanic['Fare']<500]
fig = plt.figure(figsize=(14,8))

g = fig.add_gridspec(2,6)

ax1 = fig.add_subplot(g[0, :2])

ax2 = fig.add_subplot(g[0, 2:4])

ax3 = fig.add_subplot(g[0, 4:])

ax4 = fig.add_subplot(g[1, :2])

ax5 = fig.add_subplot(g[1, 2:4])

ax6 = fig.add_subplot(g[1, 4:])



axes = [ax1,ax2,ax3,ax4,ax5,ax6]



titles = ['Overall', 'Passenger Class','Sex','Embarked Location', 'Number of Siblings/Spouse',

         'Number of Parents/Children', 'Fare ($)', 'Age (Yrs)']



var = ['Count','Pclass','Sex','Embarked','SibSp','Parch','Fare','Age']



x_labels = [['Total'],['First','Second','Third'], ['Male','Female'], ['Southampton', 'Cherbourg', 'Queenstown']]



def to_percent(y,position):

    return str(str(int(round(y*100,0)))+"%")



for i, ax in enumerate(axes):

    sns.barplot(x=var[i], y='Survived', data=titanic,palette='Blues',

                ax=ax,edgecolor='darkslategray')

    ax.set_ylabel('Survival Probability')

    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))

    ax.set_xlabel(None)

    if i in [1,2,4,5]:

        ax.set_ylabel(None)

    ax.set_title(label=titles[i], fontsize=16, fontweight='bold', pad=10)

    j=0

    for p in ax.patches:

        ax.text(p.get_x()+p.get_width()*.25, p.get_height()+.01,

                '{0:.0%}'.format(p.get_height()), ha='center')

        j += 1

    if i in [0,1,2]:

        ax.set_xticklabels(x_labels[i])



for i in [6,7]:

    sns.lmplot(var[i], 'Survived', titanic, height=4, aspect=4, y_jitter=.02)

    h = plt.gca()

    h.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))

    h.set(xlabel=None,ylabel='Survival Probability',ylim=(-0.1,1.19))

    h.set_title(label=titles[i], fontsize=20, fontweight='bold', pad=10)



fig.suptitle('Single Variable Survival Rates', position=(.5,1.06), fontsize=30, fontweight='bold')

fig.tight_layout(h_pad=2)
#Add Person column, identifying man, woman, or child

def person(passenger):

    age, sex = passenger

    if np.math.isnan(age):

        return 'unknown_age'

    elif age<17:

        return 'child'

    else:        

        if sex == 'male':

            return 'adult_male'

        else:

            return 'adult_female'



titanic['Person'] = titanic[['Age','Sex']].apply(person, axis=1)



#Plot Person/Sex counts and survival rates

fig, axes = plt.subplots(2,2,figsize=[13,8])

titles = ['Person Count','Sex of Child Count','Survival by Person','Survival by Sex of Child']

var = ['Person','Sex','Person','Sex']



for i, ax in enumerate(axes.flatten()):

    if i in [1,3]: 

        df = titanic[titanic['Person']=='child']

    else: 

        df = titanic

    if i in [0,1]:

        sns.countplot(x=var[i], data=df, palette='Purples_r', edgecolor='darkslategray', ax = ax)

        j=0

        for p in ax.patches:

            ax.text(p.get_x()+p.get_width()/2, p.get_height()*1.02,

                    df[var[i]].value_counts().iloc[j], ha='center')

            j += 1

        ax.set_title(label=titles[i], fontsize=16, fontweight='bold', pad=10)

        ax.set_ylabel(None)

    if i in [2,3]:

        sns.barplot(x=var[i], y='Survived', data=df,palette='Purples_r',

                        ax=ax,edgecolor='darkslategray')

        ax.set_ylabel('Survival Probability')

        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))

        ax.set_title(label=titles[i], fontsize=16, fontweight='bold', pad=10)

        j=0

        for p in ax.patches:

            ax.text(p.get_x()+p.get_width()*.25, p.get_height()+.01,

                    '{0:.0%}'.format(p.get_height()), ha='center')

            j += 1



fig.tight_layout()
fig = plt.figure(figsize=(13,14))

g = fig.add_gridspec(4,1)

ax1 = fig.add_subplot(g[0, :])

ax2 = fig.add_subplot(g[1, :])

ax3 = fig.add_subplot(g[2, :])

ax4 = fig.add_subplot(g[3, :])



axes = [ax1,ax2,ax3,ax4]



titles = ['Person by Passenger Class','Person by Embarked Location', 'Person by Sibling & Spouse Count',

         'Person by Parent & Child Count', 'Person by Fare ($)', 'Person by Age (Yrs)']



var = ['Pclass','Embarked','SibSp','Parch','Fare','Age']



for i, ax in enumerate(axes):

    sns.barplot(x='Person', y='Survived', data=titanic, palette='YlOrRd_d', ax=ax, edgecolor='darkslategray', hue=var[i])

    ax.set_ylabel('Survival Probability')

    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))

    ax.set_title(label=titles[i], fontsize=16, fontweight='bold', pad=10)

    ax.set_xlabel(None)

    if i in [1,3]:

        ax.set_ylabel(None)

    j=0

    for p in ax.patches:

        ax.text(p.get_x()+p.get_width()*.25, p.get_height()+.01,

                '{0:.0%}'.format(p.get_height()), ha='center')

        j += 1



for i in [4,5]:

    sns.lmplot(var[i], 'Survived', titanic, height=4, aspect=4, y_jitter=.02,hue='Person', palette='YlOrRd_d')

    h = plt.gca()

    h.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))

    h.set(xlabel=None,ylabel='Survival Probability',ylim=(-0.1,1.19))

    h.set_title(label=titles[i],fontsize=20, fontweight='bold', pad=10)



fig.suptitle('Survival by Person', position=(.5,1.05), fontsize=30, fontweight='bold')

fig.tight_layout(h_pad=2)
sns.catplot(x='Pclass', y='Fare', data=titanic, kind='point')
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,5))





#Plot Pclass by Embarked Location Count

sns.countplot(x='Pclass',data=titanic,palette='winter', hue='Embarked',edgecolor='darkslategray', ax=ax1)

embrk = ['S','C','Q','S','C','Q','S','C','Q']

pclass = [1,2,3,1,2,3,1,2,3]

i=0

j=[0,0,0,1,1,2,2,2,1]

for p in ax1.patches:

    ax1.text(p.get_x()+p.get_width()/2, p.get_height()+5,

            titanic[titanic['Pclass']==pclass[i]]['Embarked'].value_counts().iloc[j[i]], ha='center')

    i += 1

ax1.set(ylabel=None)

ax1.set_title(label='Pclass by Embarked Location Count', fontsize=16, fontweight='bold', pad=10)



#Plot Person by Pclass Count

sns.countplot(x='Person',data=titanic,palette='winter', hue='Pclass',edgecolor='darkslategray',ax=ax2)

pclass = [1,2,3,1,2,3,1,2,3,1,2,3]

person = ['adult_male','adult_female','unknown_age','child','adult_male','adult_female','unknown_age','child',

         'adult_male','adult_female','unknown_age','child','adult_male','adult_female','unknown_age','child']

i=0

j=[1,0,1,2,2,2,2,1,0,1,0,0]

for p in ax2.patches:

    ax2.text(p.get_x()+p.get_width()/2, p.get_height()+5,

            titanic[titanic['Person']==person[i]]['Pclass'].value_counts().iloc[j[i]], ha='center')

    i += 1

ax2.set(ylabel=None)

ax2.set_title(label='Person by Pclass Count', fontsize=16, fontweight='bold', pad=10)



#Plot Count of Person by Embarked Location

sns.countplot(x='Person',data=titanic,palette='winter', hue='Embarked',edgecolor='darkslategray',ax=ax3)

embrk = ['S','C','Q','S','C','Q','S','C','Q','S','C','Q']

person = ['adult_male','adult_female','unknown_age','child','adult_male','adult_female','unknown_age','child',

         'adult_male','adult_female','unknown_age','child','adult_male','adult_female','unknown_age','child']

i=0

j=[0,0,0,0,1,1,2,1,2,2,1,2]

for p in ax3.patches:

    ax3.text(p.get_x()+p.get_width()/2, p.get_height()+5,

            titanic[titanic['Person']==person[i]]['Embarked'].value_counts().iloc[j[i]], ha='center')

    i += 1

ax3.set(ylabel=None)

ax3.set_title(label='Person by Embarked Location Count', fontsize=16, fontweight='bold', pad=10)



plt.tight_layout()
titanic['Accompanied'] = titanic['SibSp']+titanic['Parch']

titanic['Accompanied'] = np.where(titanic['Accompanied']>0, 'With Family', 'Alone')
fig = plt.figure(figsize=(16, 8))

g = fig.add_gridspec(2,8)

ax1 = fig.add_subplot(g[0, :4])

ax2 = fig.add_subplot(g[0, 4:])

ax3 = fig.add_subplot(g[1, :4])

ax4 = fig.add_subplot(g[1, 4:])



axes = [ax1,ax2,ax3,ax4]



titles = ['Accompanied','Accompanied by Person','Accompanied by Passenger Class','Accompanied by Embarked Location',

          'Accompanied by Fare ($)', 'Accompanied by Age (Yrs)']



var = [None,'Person','Pclass','Embarked','Fare','Age']



for i, ax in enumerate(axes):

    sns.barplot(x='Accompanied', y='Survived', data=titanic, palette='RdBu_d', ax=ax, edgecolor='darkslategray', hue=var[i])

    ax.set_ylabel('Survival Probability')

    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))

    ax.set_title(label=titles[i], fontsize=16, fontweight='bold', pad=10)

    ax.set_xlabel(None)

    if i in [1,3]:

        ax.set_ylabel(None)

    j=0

    for p in ax.patches:

        ax.text(p.get_x()+p.get_width()*.3, p.get_height()+.01,

                '{0:.0%}'.format(p.get_height()), ha='center')

        j += 1



for i in [4,5]:

    sns.lmplot(var[i], 'Survived', titanic, height=4, aspect=4, y_jitter=.02,hue='Accompanied', palette='RdBu_d')

    h = plt.gca()

    h.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))

    h.set(xlabel=None,ylabel='Survival Probability',ylim=(-0.1,1.19))

    h.set_title(label=titles[i],fontsize=20, fontweight='bold', pad=10)



fig.suptitle('Survival by Accompanied Status', position=(.5,1.08), fontsize=30, fontweight='bold')

fig.tight_layout(h_pad=2)
age_buckets = [10,20,30,40,50,60,70,80]



sns.lmplot('Age','Survived',data=titanic,x_bins=age_buckets)

h = plt.gca()

h.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))

h.set(ylabel='Survival Probability',ylim=(-0.1,1.05))

h.set_title(label='Survival by Age Range',fontsize=20, fontweight='bold', pad=10)



sns.lmplot('Age','Survived',hue='Sex',data=titanic,palette='Oranges',x_bins=age_buckets)

h = plt.gca()

h.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))

h.set(ylabel='Survival Probability',ylim=(-0.1,1.05))

h.set_title(label='Survival by Age Range by Sex',fontsize=20, fontweight='bold', pad=10)