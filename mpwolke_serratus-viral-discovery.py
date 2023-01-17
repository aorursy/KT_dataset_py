# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/serratus-ultrahigh-throughput-viral-discovery/notebook/200423_ab/testing_SraRunInfo.csv', encoding='ISO-8859-2')

df.head()
df.isnull().sum()
# Numerical features

Numerical_feat = [feature for feature in df.columns if df[feature].dtypes != 'O']

print('Total numerical features: ', len(Numerical_feat))

print('\nNumerical Features: ', Numerical_feat)
index_int_float = ['spots', 'bases', 'spots_with_mates', 'avgLength', 'size_MB', 'TaxID']      



plt.figure(figsize=[20,12])

i = 1

for col in index_int_float :

    plt.subplot(5,10,i)

    sns.violinplot(x=col, data= df, orient='v')

    sns.despine()

    i = i+1

plt.tight_layout()

plt.show()
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
index_str = ['Run', 'ReleaseDate', 'LoadDate', 'AssemblyName', 'download_path', 'Experiment', 'LibraryName', 'LibraryStrategy', 'LibrarySelection', 'LibrarySource', 'LibraryLayout', 'Platform', 'Model', 'SRAStudy', 'BioProject', 'Sample', 'BioSample', 'SampleType', 'ScientificName', 'SampleName', 'Tumor', 'CenterName', 'Submission', 'Consent', 'RunHash', 'ReadHash']



plt.figure(figsize=[30,10])

i = 1

for col in index_str :

    plt.subplot(4,10,i)

    sns.scatterplot(x=col, y = 'size_MB' ,data= df)

    sns.despine()

    i = i+1

plt.tight_layout()

plt.show()
import cv2



from PIL import Image

im = Image.open("../input/serratus-ultrahigh-throughput-viral-discovery/notebook/200411/div_v_align_plot1.png")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
def plot_dist_col(column):

    pos__df = df[df['spots_with_mates'] ==1]

    neg__df = df[df['spots_with_mates'] ==0]



    '''plot dist curves for train and test weather data for the given column name'''

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.distplot(pos__df[column].dropna(), color='green', ax=ax).set_title(column, fontsize=16)

    sns.distplot(neg__df[column].dropna(), color='purple', ax=ax).set_title(column, fontsize=16)

    plt.xlabel(column, fontsize=15)

    plt.legend(['spots_with_mates', 'avgLength'])

    plt.show()

plot_dist_col('avgLength')
f= plt.figure(figsize=(12,5))



ax=f.add_subplot(121)

sns.distplot(df["size_MB"],color='#333ed6',ax=ax)

ax.set_title('Distribution of')



ax=f.add_subplot(122)

sns.distplot(df["avgLength"],color='#fc038c',ax=ax)

ax.set_title('Distribution of')
from PIL import Image

im = Image.open("../input/serratus-ultrahigh-throughput-viral-discovery/notebook/200411/div_v_align_plot2.png")

#tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000

#tlabel[tlabel != 0] = 255

# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))

plt.imshow(im)

display(plt.show())
sns.catplot(x="spots", kind="count",hue = 'spots_with_mates',palette='viridis',data=df)

plt.xticks(rotation=45)
#Code from Gabriel Preda

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
plot_count("spots_with_mates", "Spots with Mates", df,4)
hist = df[['size_MB','avgLength']]

bins = range(hist.size_MB.min(), hist.size_MB.max()+10, 5)

ax = hist.pivot(columns='avgLength').size_MB.plot(kind = 'hist', stacked=True, alpha=0.5, figsize = (10,5), bins=bins, grid=False)

ax.set_xticks(bins)

ax.grid('on', which='major', axis='x')
bboxtoanchor=(1.1, 1.05)

#seaborn.set(rc={'axes.facecolor':'03fc28', 'figure.facecolor':'03fc28'})

df.plot.area(y=['avgLength','spots','size_MB', 'spots_with_mates', 'ProjectID', 'TaxID'],alpha=0.4,figsize=(12, 6));
fig = px.bar(df, 

             x='size_MB', y='avgLength', color_discrete_sequence=['#27F1E7'],

             title='Viral Discovery', text='spots')

fig.show()
fig = px.density_contour(df, x="size_MB", y="avgLength",title='Viral Discovery', color_discrete_sequence=['purple'])

fig.show()
fig = px.line(df, x="ReleaseDate", y="size_MB", color_discrete_sequence=['darkseagreen'], 

              title="Viral Discovery")

fig.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.LibrarySource)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()


#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.ScientificName)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Platform)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Model)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.CenterName)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()