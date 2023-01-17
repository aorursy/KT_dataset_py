# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/ba-freelancer-germany/BA_freelancer_out.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'BA_freelancer_out.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
fig = px.bar(df, 

             x='Qualification', y='Personnel description', color_discrete_sequence=['darkgreen'],

             title='Geschäftsanalysen', text='Location')

fig.show()
fig = px.bar(df, 

             x='Titel', y='Offer', color_discrete_sequence=['crimson'],

             title='Geschäftsanalysen', text='Tags')

fig.show()
fig = px.bar(df, 

             x='Personnel description', y='No references', color_discrete_sequence=['magenta'],

             title='Geschäftsanalysen', text='Hourly rate')

fig.show()
fig = px.line(df, x="Qualification", y="Personnel description", color_discrete_sequence=['darkseagreen'], 

              title="Geschäftsanalysen")

fig.show()
fig = px.line(df, x="No references", y="Offer", color_discrete_sequence=['teal'], 

              title="Geschäftsanalysen")

fig.show()
plt.figure(figsize=(10,6))

sns.countplot(x= 'No references', data = df, palette="cool",edgecolor="black")

plt.title('Geschäftsanalysen No References')

plt.show()
#Codes from Gabriel Preda

def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count("No references", "No references", df,4)
fig = px.scatter(df, x="No references", y="Qualification",color_discrete_sequence=['crimson'], title="Business Analyst with No references" )

fig.show()