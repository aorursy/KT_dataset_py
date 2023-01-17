#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTD732VlbvoWoXF2Rnv-MYIx50AMaeiuJLawL3hiIsdw4OPKoKh&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

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
df = pd.read_csv('../input/cusersmarildownloadscarehomecsv/carehome.csv', sep=';')

df
sns.countplot(x="11/04/2020",data=df,palette="GnBu_d",edgecolor="black")



# changing the font size

sns.set(font_scale=1)
sns.countplot(x="15/05/2020",data=df,palette="GnBu_d",edgecolor="black")



# changing the font size

sns.set(font_scale=1)
plt.style.use('fivethirtyeight')

plt.figure(figsize = (6, 4))



# KDE plot of loans that were repaid on time

sns.kdeplot(df.loc[df['11/04/2020'] == 0, '15/05/2020'] / 365, label = 'target == 0')



# KDE plot of loans which were not repaid on time

sns.kdeplot(df.loc[df['11/04/2020'] == 1, '15/05/2020'] / 365, label = 'target == 1')



# Labeling of plot

plt.xlabel('11/04/2020'); plt.ylabel('15/05/2020'); plt.title('Covid-19 Deaths in Care Homes');
df1=df.copy(deep=True)
#Lets see this column variation using a histogram

plt.figure()

A=plt.hist(df1["15/05/2020"], edgecolor="red")
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('location').size()/df['15/05/2020'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
fig = px.bar(df, x= "location", y= "11/04/2020", color_discrete_sequence=['crimson'],)

fig.show()
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
plotPerColumnDistribution(df, 6, 3)
plt.figure(figsize = (8, 8))



# Graph the age bins and the average of the target as a bar plot

plt.bar(df.index.astype(str), 100 * df['11/04/2020'])



# Plot labeling

plt.xticks(rotation = 75); plt.xlabel('11/04/2020'); plt.ylabel('15/05/2020')

plt.title('Covid-19 Deaths in Care Homes');
plt.figure(figsize = (10, 12))



# iterate through the sources

for i, source in enumerate(['06/05/2020', '13/05/2020', '15/05/2020']):

    

    # create a new subplot for each source

    plt.subplot(3, 1, i + 1)

    # plot repaid loans

    sns.kdeplot(df.loc[df['11/04/2020'] == 0, source], label = 'target == 0')

    # plot loans that were not repaid

    sns.kdeplot(df.loc[df['11/04/2020'] == 1, source], label = 'target == 1')

    

    # Label the plots

    plt.title('Distribution of %s by Target Value' % source)

    plt.xlabel('%s' % source); plt.ylabel('13/04/2020');

    

plt.tight_layout(h_pad = 2.5)
# sklearn preprocessing for dealing with categorical variables

from sklearn.preprocessing import LabelEncoder
# Create a label encoder object

le = LabelEncoder()

le_count = 0



# Iterate through the columns

for col in df:

    if df[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(df[col].unique())) <= 2:

            # Train on the training data

            le.fit(df[col])

            # Transform both training and testing data

            df[col] = le.transform(app_train[col])

            #app_test[col] = le.transform(app_test[col])

            

            # Keep track of how many columns were label encoded

            le_count += 1

            

print('%d columns were label encoded.' % le_count)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSZVilbXS_prBw7cLzUeH9sS8ElEvdaLnKGi0DFp7InrwqEXZie&usqp=CAU',width=400,height=400)