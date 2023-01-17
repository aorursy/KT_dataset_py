import pandas as pd

import matplotlib

import numpy as np

from matplotlib import pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns

import math



%matplotlib inline

matplotlib.rcParams.update({'font.size': 15})

colors = ["#003f5c", "#bc5090", "#ffa600", "#127681", "#ea5455"]

customPalette = sns.set_palette(sns.color_palette(colors))

sns.set_style("whitegrid")

filepath = '../input/osic-pulmonary-fibrosis-progression/';

test = pd.read_csv(filepath + 'test.csv')

train = pd.read_csv(filepath + 'train.csv')
def show_dataset_common_info(dataset, label, fignumber):

    df = dataset[['Patient', 'Sex', 'SmokingStatus']].drop_duplicates()



    _, axes = plt.subplots(1, 2, figsize=(12,5))

    axes[0].set_title('Fig.%s. Gender and patient habits of %s dataset' % (fignumber, label))



    df.groupby(['SmokingStatus'])['Patient'].count().plot.pie(

        label='', autopct='%.2f%%', labeldistance=None, ax=axes[1], textprops={'color':"w"})

    axes[1].legend(loc='upper right')



    df.groupby(['Sex'])['Patient'].count().plot.pie(

        label='', autopct='%.2f%%', labeldistance=None, ax=axes[0], textprops={'color':"w"})

    axes[0].legend(loc='upper right')



    plt.show()

    print('Total patient count in %s dataset: %s' % (label, df['Patient'].count()))



show_dataset_common_info(train, 'train', '1')
df = train.groupby(['Sex', 'SmokingStatus'])['Patient'].unique().reset_index()

df['Patient'] = df['Patient'].apply(lambda x: len(x))

ax = sns.catplot(x="SmokingStatus", y="Patient",

                 kind="bar", data=df,

                 hue="Sex", palette=customPalette,

                 height=4, aspect=2.5)



start, end = ax.axes[0,0].get_ylim()



plt.title('Fig.2. The number of patients in different smoking status', fontsize=20)

plt.xlabel('', fontsize=19)

plt.ylabel('Count', fontsize=19)

plt.yticks(np.arange(start, end, max(int(math.fabs(end-start)/5),1)))

ax.axes[0,0].yaxis.set_major_formatter(ticker.ScalarFormatter())
df = train.groupby(['Sex', 'Age'])['Patient'].unique().reset_index()

df['Patient'] = df['Patient'].apply(lambda x: len(x))

df = df.rename(columns={'Patient': 'Count'})



fig, axes = plt.subplots(2, 1, figsize=(12,5))

fig.subplots_adjust(wspace=0.2, hspace=0.2)

ax = sns.barplot(x="Age", y="Count",

                 hue='Sex', data=df,

                 palette=customPalette, ax=axes[0])

ax.set_title('Fig.3. Age distribution of patients')

ax.legend(loc='upper right')

ax.set_xlabel('')

step = max(math.floor(math.fabs(df['Age'].max()-df['Age'].min())/11),1)

ax.xaxis.set_major_locator(ticker.MultipleLocator(step))



color = {"Male": "C1", "Female": "C0"}

for label in df['Sex'].unique():

    data = df[df['Sex'] == label].set_index('Count')

    sns.distplot(data['Age'], label=label, hist=True, kde=True, ax=axes[1], rug=True, bins=10)



    mean = data['Age'].mean()

    median = data['Age'].median()

    std = data['Age'].std()



    axes[1].axvline(mean, color=color[label], label="Mean", ls='-')

    axes[1].axvline(mean+std, color=color[label], label="Mean+std", ls='--')

    axes[1].axvline(mean-std, color=color[label], label="Mean-std", ls='--')

    axes[1].axvline(median, color=color[label], label="Median", ls='-.')



plt.show()
def _range(df):

    mini = df.min()

    maxi = df.max()

    rang = maxi - mini

    return rang



df = train[['Patient','Weeks','FVC','Percent']].groupby(['Patient']).agg([_range]).reset_index()

df.columns = ["".join(x) for x in df.columns.ravel()]

df['Patient'] = df['Patient'].keys()



fig, axes = plt.subplots(1, 2, figsize=(12,5))

fig.subplots_adjust(wspace=0.2, hspace=0.2)

ss_loc = df.groupby(['Weeks_range'])['Patient'].count().reset_index()

ax = sns.barplot(x='Weeks_range', y='Patient', color="C2", palette=customPalette,

                 data=ss_loc, ax=axes[0])

ax.set_xlabel('Weeks')

ax.set_ylabel('Count')

step = max(math.ceil(math.fabs(ss_loc['Weeks_range'].max()-ss_loc['Weeks_range'].min())/5),1)

ax.xaxis.set_major_locator(ticker.MultipleLocator(step))



sns.distplot(ss_loc['Weeks_range'], hist=True, kde=True, ax=axes[1], bins=5)

axes[1].set_xlabel('Weeks')



ax.set_title('Fig.4. Distribution of FVC observation interval')



plt.show()
fig, axes = plt.subplots(2, 2, figsize=(18,8))

fig.subplots_adjust(wspace=0.2, hspace=0.3)

labels = [('FVC_range', 'ml') , ('Percent_range', '%')]

axes[0,0].set_title('Fig.5. Distribution of change of lung capacity (LC) per obs. interval')



for i, label in enumerate(labels):

    ax = sns.barplot(x='Patient', y=label[0], color="C2", palette=customPalette,

                     data=df, ax=axes[i, 0])



    ax.set_xlabel('Patient #')

    ax.set_ylabel('Change of LC, %s' % label[1])

    step = max(math.ceil(math.fabs(df['Patient'].max()-df['Patient'].min())/5),1)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))



    sns.distplot(df[label[0]], hist=True, kde=True, ax=axes[i, 1])

    axes[i, 1].set_xlabel('Change of LC, %s' % label[1])



    mean = df[label[0]].mean()

    median = df[label[0]].median()

    std = df[label[0]].std()



    axes[i, 1].axvline(mean, label="Mean", ls='-')

    axes[i, 1].axvline(mean+std, label="Mean+std", ls='--')

    axes[i, 1].axvline(mean-std, label="Mean-std", ls='--')

    axes[i, 1].axvline(median, label="Median", ls='-.')



plt.show()
def _range(df):

    mini = df.min()

    maxi = df.max()

    rang = maxi - mini

    return rang



df = train[['Patient', 'Age', 'FVC', 'Percent']].groupby(['Age', 'Patient']).agg([_range]).reset_index()

df.columns = ["".join(x) for x in df.columns.ravel()]



fig, axes = plt.subplots(2, 2, figsize=(18,8))

fig.subplots_adjust(wspace=0.2, hspace=0.2)

labels = [('FVC_range', 'ml') , ('Percent_range', '%')]

axes[0, 0].set_title('Fig.6. Distribution of median change of lung capacity (LC) per patient age')



for i, label in enumerate(labels):

    ss_loc = df.groupby(['Age'])[label[0]].median().reset_index()

    ax = sns.barplot(x='Age', y=label[0], color="C2", palette=customPalette,

                     data=ss_loc, ax=axes[i, 0])

    ax.set_ylabel('Change of LC, %s' % label[1])

    ax.set_xlabel('Age')

    step = max(math.ceil(math.fabs(ss_loc['Age'].max()-ss_loc['Age'].min())/5),1)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))



    sns.distplot(ss_loc[label[0]], hist=True, kde=True, ax=axes[i, 1], bins=10)

    axes[i, 1].set_xlabel('Change of LC, %s' % label[1])



plt.show()
df = train[['Patient', 'Age', 'FVC', 'Percent', 'Sex']].groupby(['Age', 'Patient', 'Sex']).agg([_range]).reset_index()

df.columns = ["".join(x) for x in df.columns.ravel()]



fig, axes = plt.subplots(2, 2, figsize=(18,8))

fig.subplots_adjust(wspace=0.2, hspace=0.2)

labels = [('FVC_range', 'ml') , ('Percent_range', '%')]

axes[0, 0].set_title('Fig.7. Distribution of median change of lung capacity (LC) per patient age')



for i, label in enumerate(labels):

    ss_loc = df.groupby(['Age', 'Sex'])[label[0]].median()

    ss_rloc = ss_loc.reset_index()

    ax = sns.barplot(x="Age", y=label[0], data=ss_rloc,

                     hue="Sex", palette=customPalette, ax=axes[i, 0])

    ax.set_ylabel('Change of LC, %s' % label[1])

    ax.set_xlabel('Age')

    ax.legend(loc='upper right')

    step = max(math.ceil(math.fabs(ss_rloc['Age'].max()-ss_rloc['Age'].min())/5),1)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))

    sns.distplot(ss_loc[:,'Female',:], hist=True, kde=True, ax=axes[i, 1], label='Female')

    sns.distplot(ss_loc[:,'Male',:], hist=True, kde=True, ax=axes[i, 1], label='Male')

    axes[i, 1].set_xlabel('Change of LC, %s' % label[1])



    mean = ss_loc[:,'Female',:].mean()

    median = ss_loc[:,'Female',:].median()

    std = ss_loc[:,'Female',:].std()



    axes[i, 1].axvline(mean, label="Mean", ls='-')

    axes[i, 1].axvline(mean+std, label="Mean+-std", ls='--')

    axes[i, 1].axvline(mean-std, ls='--')

    axes[i, 1].axvline(median, label="Median", ls='-.')



    mean = ss_loc[:,'Male',:].mean()

    median = ss_loc[:,'Male',:].median()

    std = ss_loc[:,'Male',:].std()



    axes[i, 1].axvline(mean, ls='-', color="C1")

    axes[i, 1].axvline(mean+std, ls='--', color="C1")

    axes[i, 1].axvline(mean-std, ls='--', color="C1")

    axes[i, 1].axvline(median, ls='-.', color="C1")

    axes[i, 1].legend(loc='upper right')



plt.show()
show_dataset_common_info(test, 'test', '8')
fig, axes = plt.subplots(1,1, figsize=(12,5))

df = train[['Patient','Weeks','Percent']]

for patient in test['Patient'].to_numpy():

    data = df.groupby(['Patient']).get_group(patient)[['Weeks','Percent']]

    sns.lineplot(x='Weeks', y='Percent', data=data, palette=customPalette,

                 label=patient[0:6]+"***"+patient[-2:])

axes.set_ylabel('Change of LC, %')

axes.set_title('Fig.9. Observations of lung function of patients from test dataset')

plt.show()