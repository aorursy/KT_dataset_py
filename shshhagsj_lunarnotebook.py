# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import plotly.graph_objects as go

import plotly.graph_objs as go



import plotly.express as px

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



df = pd.read_csv("/kaggle/input/500-years-of-mysterious-lunar-anomalies/Lunar Anomalies.csv")





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df.Location.value_counts()[0:3].plot(kind="bar")
for i in df.columns:

    print("Unique Values For",i," : ",len(df[i].unique()))
df["Mevsim"]=["Kış" if x=="Feb" or x=="Dec" or x=="Jan"  else "Ilkbahar" if x=="March" or x=="Apr" or x=="May" else "Sbahar" if x=="Sep" or  x=="Oct"or x=="Nov"  else "Summer" for x in df.Month]

df.info()

df.head()




df.dropna(inplace=True)

df["text"] = df.Description



df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))



#noktalama işaretleri

df['text'] = df['text'].str.replace('[^\w\s]',' ')



#sayılar

df['text'] = df['text'].str.replace('\d',' ')



#stopwords

import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

sw = stopwords.words('english')

sw+=["bkz","bir"]

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))



#lemmi

from textblob import Word

text = " ".join(i for i in df.text)

wordcloud = WordCloud(background_color = "black").generate(text)

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis("off")

plt.tight_layout(pad = -9)

plt.show()
fig = px.sunburst(df.tail(200), path=['Year',"Mevsim"])

fig.update_layout(title="Years and Seasons",title_x=0.5)

fig.show()

dictt = {}

for i in text.split():

    if i in dictt:

        dictt[i]+=1

    else:

        dictt[i] = 0
pd.Series(dictt).sort_values(ascending=False).drop(["e"])[0:10].plot(kind="bar")
for i in pd.Series(dictt).sort_values(ascending=False).drop(["e"]).index[0:10]: 

    df[i] = np.zeros(498)

    
z = 0

num = 0

while z<536:

    for i in pd.Series(dictt).sort_values(ascending=False).drop(["e"]).index[0:10]:

        df.loc[z,i] = int(i in df.loc[z].text)

    

    num+=1

    if num == 498:

        break

    z = df.index[num]
df_bl
df_bl = df[df['Location'].isin(df.Location.value_counts()[0:10].index)].copy()

fig = px.sunburst(df_bl, path=['bright',"Location"])

fig.update_layout(title="Where did we see bright",title_x=0.5)

fig.show()

for i in df.columns[8:18]:

    df_bl[i] = df_bl[i].astype(int)
df_blc = df_bl.drop(["Month","Credit","Description","text"],axis=1).copy()
sns.set(font_scale=0.8)

fig_dims=(16,10)

fig,ax=plt.subplots(figsize=fig_dims)

corr = pd.get_dummies(df_blc).corr().iloc[2:12]

sns.heatmap(corr,

yticklabels=corr.index,

    xticklabels=corr.columns,

    fmt=".2f",

    annot=True

)