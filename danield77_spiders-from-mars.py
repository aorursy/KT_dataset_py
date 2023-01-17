# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from matplotlib.colors import ColorConverter

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats.stats import pearsonr

df = pd.read_csv('../input/celebrity_deaths_4.csv', encoding='latin1')

ix = np.isnan(df['age'].values)|np.isnan(df['fame_score'].values)

pearsonr(df['age'].values[~ix],df['fame_score'].values[~ix])[0]
most_famous = df.sort_values('fame_score', ascending=False).head(1000)

ix = np.isnan(most_famous['age'].values)|np.isnan(most_famous['fame_score'].values)

pearsonr(most_famous['age'].values[~ix],most_famous['fame_score'].values[~ix])[0]
print(df['age'].mode())
def indexation(column,order):

    if column.name=='age':

        retour = column.unique().tolist()

        retour.sort()

        if order == 'descending':

            retour = retour[::-1]

        del retour[30:]

    elif column.name=='nationality':

        retour = df['nationality'].value_counts().head(30).index.tolist()

    return retour

		

def count_by_field(df, column, operation, projection):

    res = []

    fields = indexation(df[column],'ascending')

    for field in fields:

        if operation == 'count':

            field_count = pd.value_counts(df[column]==field)[1]

        elif operation == 'mean':

            field_count = df[projection][df[column]==field].mean()

        res.append((field, field_count))

    return res

		

def plot_count_by_field(df, field, operation, projection):

    ages_counts = count_by_field(df, field, operation, projection)

    ages = [t[0] for t in ages_counts] 

    counts = [t[1] for t in ages_counts]

    if field=='age' and operation=='count':

        title = 'Celebrities deaths number by ages'

    if field=='nationality' and operation=='mean':

        if projection=='age':

            title = 'Average age at death of celebrities by nationality'

        elif projection=='fame_score':

            title = 'Average fame at death of celebrities by nationality'

    plot_ages_values(ages, counts, title)

		

def plot_ages_values(ages, values, title):

    plt.clf()

    y_pos = np.arange(len(values))

    cc = ColorConverter()

    min_mean = min(values)

    max_mean = max(values)

    colors = []

    compteur = 0

    for m in values:

        if ages[compteur] == 27:

            colors.append(cc.to_rgb((1., 0., 0.)))

        else:

            colors.append(cc.to_rgb((0.2, (m-min_mean)/(max_mean-min_mean), 0.5)))

        compteur=compteur+1

    plt.barh(y_pos, values, color=colors); 

    plt.yticks(y_pos, ages, fontsize=10) 

    plt.title(title, fontsize=22)

    plt.show()

		

plot_count_by_field(df,'age','count','')
plot_count_by_field(most_famous,'age','count','')
print(df['cause_of_death'].value_counts().head(10))
months_distribution = pd.Categorical(df['death_month'], ["January","February","March","April","June","July","August","September","October","November","December"]).value_counts()

months_distribution.plot.bar()

import matplotlib.pyplot as plt

plt.show()
plot_count_by_field(df,'nationality','mean','age')
plot_count_by_field(df,'nationality','mean','fame_score')
print(df[(df['nationality']=='South')])
print(df[df['age']==df['age'].max()])