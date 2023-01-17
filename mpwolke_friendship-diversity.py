#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

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
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadsfriendshipscsv/friendships.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'friendships.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
# checking dataset



print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
df.describe().T
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
plot_count("I don't have any friends", "I don't have any friends", df,4)
facet = sns.FacetGrid(df, hue = "More than half", aspect = 3)

facet.map(sns.kdeplot, "Unnamed: 9", shade = True, color = "purple")

facet.set(xlim = (0, df["Unnamed: 9"].max()))

facet.add_legend()



plt.show();
facet = sns.FacetGrid(df, hue = "I don't have any friends", aspect = 3)

facet.map(sns.kdeplot, "Unnamed: 17", shade = True, color = "purple")

facet.set(xlim = (0, df["Unnamed: 17"].max()))

facet.add_legend()



plt.show();
Total = np.random.normal(size=100)

sns.distplot(Total, color = "purple");
from scipy import stats



Total = np.random.normal(0, 1, size=30)

bandwidth = 1.06 * Total.std() * Total.size ** (-1 / 5.)

support = np.linspace(-4, 4, 200)



kernels = []

for Total_i in Total:



    kernel = stats.norm(Total_i, bandwidth).pdf(support)

    kernels.append(kernel)

    plt.plot(support, kernel, color= "purple")



sns.rugplot(Total, color=".2", linewidth=3);
import plotly.express as px

fig = px.line(df, x="Unnamed: 9", y="I don't have any friends", color_discrete_sequence=['purple'], 

              title="Proportion of Friends")

fig.show()
fig = px.scatter(df, x="Unnamed: 25", y="All similar",color_discrete_sequence=['crimson'], title="Proportion of Friends" )

fig.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Total)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set2', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()