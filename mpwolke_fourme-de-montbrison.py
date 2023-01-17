#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSDhUDlqZMfVrfOSgULtEoOlKUfog4Jlpn08yW-RnfWy4yTqVo6qA&s',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRvcJoKE_uhZno_agAn1Gv7pDGwaeB1sUIXJqamj8aaJxneep0uA&s',width=400,height=400)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/french-cheese-detection/fourmes_test/6.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/french-cheese-detection/fourmes_test/2.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTdNoyBZjN27snbcANS02eIZvPXNAtxMWWmHeZeZXMqxsjtwEDr&s',width=400,height=400)
nRowsRead = 1000 # specify 'None' if want to read whole file

fourme = pd.read_csv('../input/french-cheese-detection/fourmes_train.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

fourme.dataframeName = 'fourmes_train.csv'

nRow, nCol = fourme.shape

print(f'There are {nRow} rows and {nCol} columns')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRH0jrIzZh6CZmtk1U2ymX-Ni2_ITPKfeCtE2rhcUvI10ORpGT1xw&s',width=400,height=400)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/french-cheese-detection/fourmes_test/5.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ0_4rRVxesGfzALCebZVCeVpf0roph-NrUNf7B6exhHO4aRkOh&s',width=400,height=400)
fourme.head()
fourme.dtypes
fourme["x1"].plot.hist()

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRRCl7fjJvH_UYAeZPmCGGvfNQtnPYELG_b9NLnEsrSsifMyrC_&s',width=400,height=400)
fourme["y2"].plot.box()

plt.show()
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/french-cheese-detection/fourmes_train/11.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
sns.pairplot(fourme, markers="+", diag_kind="kde")

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8StgQDeRrTI3-oq72aOrZZoHPiTyMJUZloM7DPgsji0eF0Sye&s',width=400,height=400)
sns.pairplot(fourme, x_vars=['x1', 'x2', 'y1'], y_vars='y2', markers="+", size=4)

plt.show()
fourmecorr=fourme.corr()

fourmecorr
sns.heatmap(fourmecorr,annot=True,cmap='seismic')

plt.show()
sns.heatmap(fourmecorr,annot=True,cmap='Greens')

plt.show()
sns.heatmap(fourmecorr,annot=True,cmap='Pastel1')

plt.show()
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/french-cheese-detection/fourmes_train/16.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
fig, axes = plt.subplots(1, 1, figsize=(14, 6))

sns.boxplot(x='x1', y='y1', data=fourme, showfliers=False);
# Necessary Functions: 

def pie_plot(labels, values, colors, title):

    fig = {

      "data": [

        {

          "values": values,

          "labels": labels,

          "domain": {"x": [0, .48]},

          "name": "Job Type",

          "sort": False,

          "marker": {'colors': colors},

          "textinfo":"percent+label+value",

          "textfont": {'color': '#FFFFFF', 'size': 10},

          "hole": .6,

          "type": "pie"

        } ],

        "layout": {

            "title":title,

            "annotations": [

                {

                    "font": {

                        "size": 25,



                    },

                    "showarrow": False,

                    "text": ""



                }

            ]

        }

    }

    return fig
import plotly.offline as py

value_counts = fourme['x1'].value_counts()

labels = value_counts.index.tolist()

py.iplot(pie_plot(labels, value_counts,['#1B9E77', '#7570B3'], "x1"))
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/french-cheese-detection/fourmes_train/18.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/french-cheese-detection/fourmes_train/1.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
import plotly.express as px



# Grouping it by Genre and track

plot_data = fourme.groupby(['x1', 'y1'], as_index=False).x2.sum()



fig = px.line_polar(plot_data, theta='x1', r='x2', color='y1')

fig.update_layout(

    title_text='Fourme de Montbrison',

    height=500, width=1000)

fig.show()
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(fourme.x1, fourme.y1, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="x1", y="y1", data=fourme, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$x1$", "$y1$");
fourme = fourme.rename(columns={'class':'cl'})
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in fourme.cl)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200, background_color="white").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRQ5WMK_SPjCYynBkl8W1spEgKr_wWiGn_Zh6Guawf0yBpI7GhM&s',width=400,height=400)