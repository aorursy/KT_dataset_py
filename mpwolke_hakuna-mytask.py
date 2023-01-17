#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTwKBswXahTNjAoOo_9SW8SBbqK4_uDPnJVRHfNdYBUATTblq4A&s',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQv-MSGIiF_uuHl1qDnHSsYEPyK76XNgKrjACWSevVx0VkL67kv&s',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fpath = '/kaggle/input/submission-to-task/cities.csv'

df = pd.read_csv('../input/submission-to-task/cities.csv')

#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT5rOR0XpzRjiwnFUtuPFwRE17DGUciaf_O8uqAXkac7mkuXwT94Q&s',width=400,height=400)
df.head()
df.info()
df.describe()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzZPiNqqvhnCydN9VJzJ6QyVY8cf3kqOwqw-TrziivIrTOG0MtjA&s',width=400,height=400)


categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]
print(categorical_cols)
print(numerical_cols)
df = df.rename(columns={'+03600':'03600', '+03600.1': '036001'})
import matplotlib.pyplot as plt

import seaborn as sns
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4QjqDFq8fqyqhuBvB74m3TI1wfngE9DuAQ48rOtsY-We828kM&s',width=400,height=400)
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('03600').size()/df['036001'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
sns.scatterplot(x='Algiers',y='03600',data=df)
plt.figure(figsize=(8, 5))

pd.value_counts(df['036001']).plot.bar()

plt.show()
sns.countplot(df["036001"])
g = sns.jointplot(x="03600", y="99", data=df, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$03600$", "$99$");
sns.regplot(x=df['03600'], y=df['036001'])
sns.lmplot(x="03600", y="036001", hue="99", data=df)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Algiers)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://razoesparaacreditar.com/wp-content/uploads/2015/12/screen-shot-2015-07-07-21.46.32.png',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://razoesparaacreditar.com/wp-content/uploads/2015/12/resize-1.jpeg',width=400,height=400)