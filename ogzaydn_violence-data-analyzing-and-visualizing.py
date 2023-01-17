import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/violence-against-women-and-girls/makeovermonday-2020w10/violence_data.csv')

df.head()
df.info()
df.describe(include='all')
df.Country.unique()
df['Demographics Question'].unique().tolist()
df.Question.unique().tolist()
Ethiopia = df[df.Country == 'Ethiopia']

Ethiopia.head(3)

graph = Ethiopia[Ethiopia['Demographics Question'] == 'Education']



g = sns.catplot(x='Demographics Response',y='Value',col='Gender',hue='Question',

                order=['No education','Primary','Secondary','Higher'],

                data=graph,kind='bar',ci=None)

g.set_axis_labels('Education Level','Percentage (%)')

g.fig.suptitle('Ethiopians agreeing a husband is justified in hitting his wife',y=1.05)
graph = Ethiopia[Ethiopia['Demographics Question'] == 'Education']



g = sns.catplot(x='Demographics Response',y='Value',hue='Gender',

                order=['No education','Primary','Secondary','Higher'],

                data=graph,kind='bar',ci=None)

g.set_axis_labels('Education Level','Percentage (%)')

g.fig.suptitle('Ethiopians agreeing a husband is justified in hitting his wife',y=1.05)
graph = Ethiopia[(Ethiopia['Demographics Question'] == 'Residence') & (Ethiopia['Gender'] == 'F')]



g = sns.catplot(x='Demographics Response',y='Value',

                order=['Rural','Urban'],

                data=graph,kind='bar',ci=None)

g.set_axis_labels('Residence Type','Percentage (%)')

g.fig.suptitle('Ethiopian women agreeing a husband is justified in hitting his wife',y=1.05)
graph = Ethiopia[(Ethiopia['Demographics Question'] == 'Age') & (Ethiopia['Gender'] == 'F')]



g = sns.catplot(x='Demographics Response',y='Value',

                data=graph,kind='bar',ci=None)

g.set_axis_labels('Age','Percentage (%)')

g.fig.suptitle('Ethiopian women agreeing a husband is justified in hitting his wife',y=1.05)
cts = df[(df.Country == 'Pakistan') | (df.Country == 'India')] #cts means countries

cts
graph = cts[cts['Demographics Question'] == 'Education']



g = sns.catplot(x='Demographics Response',y='Value',col='Country',row='Gender',

                order=['No education','Primary','Secondary','Higher'],

                data=graph,kind='bar',ci=None)

g.set_axis_labels('Education Level','Percentage (%)')

g.fig.suptitle('Those agreeing a husband is justified in hitting his wife',y=1.05)
graph = cts[cts['Demographics Question'] == 'Age']



g = sns.catplot(x='Demographics Response',y='Value',col='Country',row='Gender',

                data=graph,kind='bar',ci=None)

g.set_axis_labels('Age','Percentage (%)')

g.fig.suptitle('Those agreeing a husband is justified in hitting his wife',y=1.05)
graph = cts[cts['Demographics Question'] == 'Residence']



g = sns.catplot(x='Demographics Response',y='Value',col='Country',row='Gender',

                order=['Rural','Urban'],

                data=graph,kind='bar',ci=None)

g.set_axis_labels('Residence Type','Percentage (%)')

g.fig.suptitle('Those agreeing a husband is justified in hitting his wife',y=1.05)