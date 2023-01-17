# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

%matplotlib inline

matplotlib.style.use('ggplot')

plt.rcParams["figure.figsize"] = [12,6]



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read the data from csv file

data = pd.read_csv('../input/appendix.csv')
# Format 'Launch Data' as datatime

data['Launch Date'] = pd.to_datetime(data['Launch Date'])

# Year is most important, let's save it to a column

data['year'] = pd.to_datetime(data['Launch Date']).dt.year
# Show the first 5 rows

data.head(5)
# Let's look at the columns and entries

data.info()
print('% Audited\n')

print(data['% Audited'].describe())

print('\n')

print('Audited (> 50% Course Content Accessed)\n')

print(data['Audited (> 50% Course Content Accessed)'].describe())

print('\n')

print('Participants (Course Content Accessed)\n')

print(data['Participants (Course Content Accessed)'].describe())
data[['year','Participants (Course Content Accessed)']].groupby(['year']).sum().plot(kind='bar');
sns.barplot(x='year', y='% Audited', data=data);
sns.stripplot(x='year', y='% Audited', data=data);
data.corr()['% Audited'].sort_values().plot(kind='bar');
# Which course titles are most popular?

data.sort_values(by='Participants (Course Content Accessed)', ascending=False)[['Course Title','Participants (Course Content Accessed)']]
# Filter only courses about 'Introduction to Computer Science'

data_cs = data[data['Course Title'].str.contains('Introduction to Computer Science')]
data_cs.head()
sns.factorplot('year',data=data,hue='Course Subject',kind='count');
sns.stripplot(x='Course Subject', y='% Audited', data=data);

locs, labels = plt.xticks();

plt.setp(labels, rotation=90);
sns.swarmplot(x='year', y='% Audited', hue='Course Subject', data=data);
sns.boxplot(x='Institution', y='% Male', hue='Course Subject', data=data)
sns.boxplot(x='year', y='% Male', hue="Institution", data=data)
sns.boxplot(x='year', y='Median Age', hue="Institution", data=data)
sns.boxplot(x='Course Subject', y='Median Age', data=data)

locs, labels = plt.xticks();

plt.setp(labels, rotation=90);
data['CS'] = pd.get_dummies(data['Course Subject'])['Computer Science']
data.head()
data.corr()['CS'].sort_values()
from sklearn.tree import DecisionTreeClassifier

X = data[['Participants (Course Content Accessed)','Total Course Hours (Thousands)',"% Male","% Bachelor's Degree or Higher",'% Audited']]

y = data['CS']



model = DecisionTreeClassifier()

model.fit( X , y )



imp = pd.DataFrame( 

    model.feature_importances_  , 

    columns = [ 'Importance' ] , 

    index = X.columns 

)

imp = imp.sort_values( [ 'Importance' ] , ascending = True )

imp[ : 10 ].plot( kind = 'barh' )

print (model.score( X , y ))
def plot_correlation_map( df ):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )
plot_correlation_map(data )
sns.lmplot(x="% Male", y="Median Age", hue="Course Subject", data=data);
data['Year'].value_counts().plot(kind='bar')
data['Course Title'].value_counts().head().plot(kind='bar')
data['Course Subject'].value_counts().plot(kind='bar')
data.info()
data['Course Subject'].value_counts().plot(kind='bar')
data.sort_values(by='Median Age', ascending=False)['Course Subject'].head(100).value_counts().plot(kind='bar')
data.sort_values(by='% Male', ascending=False)['Course Subject'].head(100).value_counts().plot(kind='bar')