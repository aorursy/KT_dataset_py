import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm_notebook as tqdm

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

from plotly import graph_objs as go,figure_factory as FF,offline as py

py.init_notebook_mode(connected=True)
data = pd.read_csv("../input/data.csv",index_col=0)
data.head()
from PIL import Image

import requests

from io import BytesIO





def get_image(url):

    response = requests.get(url)

    img = Image.open(BytesIO(response.content))

    return img.convert("RGBA")
def remove_white_border(img):

    data = np.asarray(img)

    return data[1:-1,1:-1,]



country_flags = []

for url in tqdm(data["Flag"].unique()):

    img = get_image(url)

    country_flags.append(remove_white_border(img))
if len(country_flags) == 164:

    for _ in range(6):

        country_flags.append(np.zeros((15, 21, 4)))

fig,axes = plt.subplots(17,10,figsize=(10,10))

for idx,item in enumerate(country_flags):

    row = idx//10

    col = idx%10

    axes[row][col].imshow(item)

    axes[row][col].get_xaxis().set_visible(False)

    axes[row][col].get_yaxis().set_visible(False)

plt.show()
def plot_bar(x,y,data_args={},layout_args={}):

    trace = [go.Bar(y=y,x=x,marker={"color":"green"},**data_args)]

    fig = go.Figure(data=trace,layout=go.Layout(**layout_args))

    py.iplot(fig)
player_counts_by_country = data.groupby("Nationality",as_index=False)["ID"].count().sort_values("ID",ascending=False)

plot_bar(x=player_counts_by_country["Nationality"][:50],y=player_counts_by_country["ID"][:50],layout_args={"title":"TOP50 Player counts of country"})
table = FF.create_table(player_counts_by_country[player_counts_by_country["ID"]==1])

py.iplot(table)
top50_country = player_counts_by_country["Nationality"][:50]

top50_data = data[data["Nationality"].isin(top50_country)]
plt.figure(figsize=(30,10))

sns.boxplot(data=top50_data,y="Age",x="Nationality",order=top50_country)

_ = plt.xticks(rotation=70)

_ = plt.title("Players age",fontsize=20)

sns.despine(top=True,right=True)


grid = sns.FacetGrid(top50_data[top50_data["Nationality"].isin(top50_country[:20])], col="Nationality", hue="Nationality",col_order=top50_country[:20],

                     col_wrap=5, height=3)

grid.map(sns.distplot, "Age")

grid.fig.tight_layout(w_pad=1)