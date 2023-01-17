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
df = pd.read_csv('../input/open-medic-2017/OPEN_MEDIC_2017.CSV',sep=';',encoding='latin-1')

df.head()
# checking dataset



print ("Rows     : " ,df.shape[0])

print ("Columns  : " ,df.shape[1])

print ("\nFeatures : \n" ,df.columns.tolist())

print ("\nMissing values :  ", df.isnull().sum().values.sum())

print ("\nUnique values :  \n",df.nunique())
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'L_ATC4', data = df, palette="GnBu_d",edgecolor="black")

plt.subplot(132)

sns.countplot(x= 'L_ATC5', data = df, palette="flag",edgecolor="black")

plt.subplot(133)

sns.countplot(x= 'L_CIP13', data = df, palette="Greens_r",edgecolor="black")

plt.show()
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'ATC1', data = df, palette="GnBu_d",edgecolor="black")

plt.subplot(132)

sns.countplot(x= 'l_ATC1', data = df, palette="flag",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(133)

sns.countplot(x= 'ATC2', data = df, palette="Greens_r",edgecolor="black")

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(16,10))

sns.countplot(x="L_ATC2",data=df,palette="flag",edgecolor="black")

plt.title('Therapeutic Sub-Group Label', weight='bold')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
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
plot_count("L_ATC2", "Therapeutic Sub-Group Label", df,4)
plt.figure(figsize=(20, 12))

plt.subplot(121)

sns.boxplot(x = 'CIP13', y = 'L_ATC2', data = df)

plt.subplot(122)

sns.boxplot(x = 'CIP13', y = 'l_ATC1', data = df)

plt.show()
# Distribution of different type of amount

fig , ax = plt.subplots(1,3,figsize = (12,5))



CIP13 = df.CIP13.values

GEN_NUM= df.GEN_NUM.values

AGE = df.AGE.values



sns.distplot(CIP13 , ax = ax[0] , color = 'pink').set_title('Pharmaceutical Specialty Identification Code' , fontsize = 14)

sns.distplot(GEN_NUM , ax = ax[1] , color = 'cyan').set_title('Generic Group' , fontsize = 14)

sns.distplot(AGE , ax = ax[2] , color = 'purple').set_title('Age' , fontsize = 14)





plt.show()
import plotly.express as px

fig = px.line(df, x="ATC1", y="l_ATC1", color_discrete_sequence=['darksalmon'], 

              title="Anatomical Main Group ID & Label")

fig.show()
anticaries = df[(df['L_ATC4']=='MEDICAMENTS PROPHYLACTIQUES ANTICARIES')].reset_index(drop=True)

anticaries.head()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in anticaries.L_ATC4)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()