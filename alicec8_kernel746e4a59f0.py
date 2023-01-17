# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/mental-health-in-tech-survey/survey.csv')
print(df.head())
print(df.shape)
print(df.dtypes)
# select numeric columns
df_numeric = df.select_dtypes(include = [np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)
# select non numeric columns
df_non_numeric = df.select_dtypes(exclude = [np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)
cols = df.columns[:27]
colours = ['#000099', '#ffff00'] # specify the colours, yellow: missing, blue: not missing
sns.heatmap(df[cols].isnull(), cmap = sns.color_palette(colours))
# do this when dataset is largeand visualization take too long
for col in df.columns:
    percent_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,round(percent_missing*100)))
cols_to_drop = ['state', 'work_interfere', 'comments']
df_cleaned = df.drop(cols_to_drop, axis=1)
print(df_cleaned.shape)
print(df_cleaned.head)
# print(df_cleaned['Age'].value_counts())
print(df_cleaned['Age'].unique())
sns.boxplot(x=df_cleaned['Age'])

# print(df['Age'].quantile(0.50))
# print(df['Age'].quantile(0.95))

age= df_cleaned['Age']
size = age.shape[0]
removed_outliers = age.between(1,100)

# print(removed_outliers)
print(str(age[removed_outliers].size) + "/" + str(size) + " data points remain.")

age[removed_outliers].plot().get_figure()
print(removed_outliers.value_counts())
index_names = df_cleaned[~removed_outliers].index
print(index_names)
import plotly.graph_objs as go
df_cleaned.drop(index_names, inplace=True)
print(df_cleaned['Age'].unique())
df_cleaned.head(400)
# clean gender data

# look at unique values
df_cleaned['Gender'].unique()
df_cleaned['Gender'].value_counts()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import random

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

# visualize using wordcloud

counts = df_cleaned['Gender'].value_counts()
counts.index= counts.index.map(str)
wordcloud = WordCloud().generate_from_frequencies(counts)
wordcloud = WordCloud(background_color = "black", relative_scaling = 0.50, width = 800, height = 400).generate_from_frequencies(counts)
plt.figure(figsize = (20,10), facecolor = 'k')
plt.imshow(wordcloud.recolor(color_func = grey_color_func, random_state=1), interpolation = "bilinear")
plt.axis("off")
plt.savefig('wordcloud.png', facecolor='k', bbox_inches='tight')
df_cleaned['Gender'].unique()
    
df_cleaned['Gender'] = df_cleaned['Gender'].replace(['female','Cis Female', 'F', 'Femake', 'woman', 'Female ','cis-female/femme', 'Female (cis)','femail','Woman','f', 'femail']
                                                     , 'Female')
df_cleaned['Gender'] = df_cleaned['Gender'].replace(['M', 'Male', 'male', 'maile', 'm',  'Cis Male', 'Mal',
                       'Male (CIS)','Make','Guy (-ish) ^_^','Male ', 'Man','msle'
                      ,'Mail', 'cis male', 'Malr', 'Cis Man',], 'Male')
df_cleaned['Gender'] = df_cleaned['Gender'].replace(['Male-ish', 'm', 'Trans-female', 'something kinda male?',
                       'queer/she/they','non-binary','Nah', 'Enby','fluid',
                       'Genderqueer', 'Androgyne', 'Agender','male leaning androgynous',
                       'Trans woman','Neuter', 'Female (trans)', 'queer',
                       'A little about you','ostensibly male, unsure what that really means'], 'other')
df_cleaned['Gender'].unique()
df_cleaned['Gender'].value_counts()
sns.set_style("darkgrid")
# x = df_cleaned['Gender'].value_counts().index
# y = df_cleaned['Gender'].value_counts()
# plt.bar(x,y)

sns.barplot(x=df_cleaned['Gender'].value_counts().index, 
            y=df_cleaned['Gender'].value_counts())


# print(df_cleaned['mental_health_consequence'].unique())
# print(df_cleaned.groupby(['mental_health_consequence','Gender']).head())
res = pd.DataFrame(df_cleaned.groupby(['mental_health_consequence','Gender'])['Gender'].count())
res.plot.bar()
res = pd.DataFrame(df_cleaned.groupby(['mental_health_consequence','Gender'])['mental_health_consequence'].count())

res.plot.bar()
sns.set(style="whitegrid", palette = "muted")
sns.set(style="ticks")
df_g_c = df_cleaned[['mental_health_consequence','Gender']]
df_g_c.head()
sns.catplot(x = 'mental_health_consequence',
            y = 'Gender'.count('Gender'),
            hue = df_g_c['Gender'].unique(), data = df_g_c
           )

# sns.stripplot(x = df_cleaned['mental_health_consequence'].unique(),
#             y = df_cleaned.groupby(['mental_health_consequence','Gender']).size(),
#             hue = df_cleaned['Gender'].unique(),
#             dodge = True, data = df_cleaned)