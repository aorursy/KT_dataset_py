#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRH1ZkhpgLMpW8mwLLLXs8IGaYaIRQSlTgyuN1luLQ0KFXqdp43',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/russian-premier-league/repository/ilikeevb--football-prediction-29a122c/data/RPL.csv', delimiter=';', encoding = "cp1251", nrows = nRowsRead)

df.dataframeName = 'RPL.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
df.dtypes
df["Год"].plot.hist()

plt.show()
df["Удары"].plot.hist()

plt.show()
df["Пропущено"].plot.hist()

plt.show()
df["Точные навесы"].plot.box()

plt.show()
df["Минуты"].plot.box()

plt.show()
sns.pairplot(df, x_vars=['Забито'], y_vars='Передачи', markers="+", size=4)

plt.show()
dfcorr=df.corr()

dfcorr
sns.heatmap(dfcorr,annot=True,cmap='winter')

plt.show()
fig, axes = plt.subplots(1, 1, figsize=(14, 6))

sns.boxplot(x='Пропущено', y='Минуты', data=df, showfliers=False);
fig, axes = plt.subplots(1, 1, figsize=(14, 6))

sns.boxplot(x='Удары', y='Удары в створ', data=df, showfliers=False);
fig, axes = plt.subplots(1, 1, figsize=(14, 6))

sns.boxplot(x='Точные навесы', y='Навесы', data=df, showfliers=False);
g = sns.jointplot(x="Часть", y="Минуты", data=df, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Часть$", "$Минуты$");
import matplotlib.style



import matplotlib as mpl



mpl.style.use('classic')
sns.jointplot(df['Точные навесы'],df['Навесы'],data=df,kind='scatter')
sns.jointplot(df['Пропущено'],df['Минуты'],data=df,kind='scatter')
sns.jointplot(df['Забито'],df['Передачи'],data=df,kind='kde',space=0,color='g')
fig=sns.jointplot(x='Год',y='Навесы',kind='hex',data=df)
g = (sns.jointplot("Удары", "Удары в створ",data=df, color="r").plot_joint(sns.kdeplot, zorder=0, n_levels=6))
ax= sns.boxplot(x="Минуты", y="Навесы", data=df)

ax= sns.stripplot(x="Минуты", y="Навесы", data=df, jitter=True, edgecolor="gray")



boxtwo = ax.artists[2]

boxtwo.set_facecolor('yellow')

boxtwo.set_edgecolor('black')

boxthree=ax.artists[1]

boxthree.set_facecolor('red')

boxthree.set_edgecolor('black')

boxthree=ax.artists[0]

boxthree.set_facecolor('green')

boxthree.set_edgecolor('black')



plt.show()
fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.violinplot(x='Часть',y='Пропущено',data=df)
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Удары',y='Удары в створ',data=df)

plt.subplot(2,2,2)

sns.violinplot(x='Удары',y='Удары в створ',data=df)

plt.subplot(2,2,3)

sns.violinplot(x='Удары',y='Удары в створ',data=df)

plt.subplot(2,2,4)

sns.violinplot(x='Удары',y='Удары в створ',data=df)
sns.set(style="darkgrid")

fig=plt.gcf()

fig.set_size_inches(10,7)

fig = sns.swarmplot(x="Передачи", y="Точные передачи", data=df)
sns.set(style="whitegrid")

fig=plt.gcf()

fig.set_size_inches(10,7)

ax = sns.violinplot(x="Навесы", y="Точные навесы", data=df, inner=None)

ax = sns.swarmplot(x="Навесы", y="Точные навесы", data=df,color="white", edgecolor="black")
fig=sns.lmplot(x="Удары", y="Удары в створ",data=df)
# venn2

from matplotlib_venn import venn2

Минуты = df.iloc[:,0]

Забито = df.iloc[:,1]

Передачи = df.iloc[:,2]

Навесы = df.iloc[:,3]

# First way to call the 2 group Venn diagram

venn2(subsets = (len(Минуты)-15, len(Забито)-15, 15), set_labels = ('Минуты', 'Забито'))

plt.show()
# donut plot

feature_names = "Минуты","Забито","Передачи"

feature_size = [len(Минуты),len(Забито),len(Передачи)]

# create a circle for the center of plot

circle = plt.Circle((0,0),0.2,color = "white")

plt.pie(feature_size, labels = feature_names, colors = ["red","green","blue","cyan"] )

p = plt.gcf()

p.gca().add_artist(circle)

plt.title("Number of Each Feature")

plt.show()
df.plot.area(y=['Удары','Удары в створ','Навесы','Точные навесы'],alpha=0.4,figsize=(12, 6));
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Победитель)

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

text = " ".join(str(each) for each in df.Проигравший)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="green").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
nRowsRead = 1000 # specify 'None' if want to read whole file

df1 = pd.read_csv('../input/russian-premier-league/data/RPL.csv', delimiter=';', encoding = "cp1251", nrows = nRowsRead)

df1.dataframeName = 'RPL.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df1.head()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df1.Соперник)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="blue").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()