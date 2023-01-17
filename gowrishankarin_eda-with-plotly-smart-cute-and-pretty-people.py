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
import matplotlib.pyplot as plt

from PIL import Image



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 9999

pd.options.display.float_format = '{:20, .2f}'.format

train_df = pd.read_csv("../input/train_relationships.csv")

train_df.head()
train_df.shape
files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("../input/train")) for f in fn]

train_images_df = pd.DataFrame({

    'files': files,

    'familyId': [file.split('/')[3] for file in files],

    'kinId': [file.split('/')[4] for file in files],

    'uniqueId': [file.split('/')[3] + '/' + file.split('/')[4] for file in files]

})

train_images_df.head()
print("Total number of members in the dataset: {0}".format(train_images_df["uniqueId"].nunique()))

print("Total number of families in the dataset: {0}".format(train_images_df["familyId"].nunique()))
family_with_most_pic = train_images_df["familyId"].value_counts()

kin_with_most_pic = train_images_df["uniqueId"].value_counts()

print("Family with maximum number of images: {0}, Image Count: {1}".format(family_with_most_pic.index[0], family_with_most_pic[0]))

print("Member with maximum number of images: {0}, Image Count: {1}".format(kin_with_most_pic.index[0], kin_with_most_pic[0]))
family_series = family_with_most_pic[:25]

labels = (np.array(family_series.index))

sizes = (np.array((family_series / family_with_most_pic.sum()) * 100))



trace = go.Pie(labels=labels, values=sizes)

layout = go.Layout(title='Pic Count by Families')

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='Families')
most_pic_members = train_images_df[train_images_df["uniqueId"] == kin_with_most_pic.index[0]].files.values

fig, ax = plt.subplots(4, 6, figsize=(50, 40))

row = 0

col = 0

for index in range(len(most_pic_members[:24])):

    with open(most_pic_members[index], 'rb') as f:

        img = Image.open(f)

        ax[row][col].imshow(img)



        if(col < 5):

            col = col + 1

        else: 

            col = 0

            row = row + 1

fig.show()
family_with_most_members = train_images_df.groupby("familyId")["kinId"].nunique().sort_values(ascending=False)

print("Family with maximum number of members: {0}, Member Count: {1}".format(family_with_most_members.index[0], family_with_most_members[0]))

print("Family with least number of members: {0}, Member Count: {1}".format(

    family_with_most_members.index[len(family_with_most_members)-1], 

    family_with_most_members[len(family_with_most_members)-1]))

large_family_df = train_images_df[train_images_df["familyId"]  == family_with_most_members.index[0]]

large_family_df.head()






def render_bar_chart(data_df, column_name, title, filename):

    series = data_df[column_name].value_counts()

    count = series.shape[0]

    

    trace = go.Bar(x = series.index, y=series.values, marker=dict(

        color=series.values,

        showscale=True

    ))

    layout = go.Layout(title=title)

    data = [trace]

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename=filename)

    

    

render_bar_chart(large_family_df, 'uniqueId', 'Pic Count by Members', 'members')
def render_images(large_family_df):

    large_family_pics = [large_family_df.loc[large_family_df.loc[large_family_df["uniqueId"] == aKin].index[0]]["files"] for aKin in large_family_df["uniqueId"].unique()]

    nrows = round(len(large_family_pics) / 6) + 1





    fig, ax = plt.subplots(nrows, 6, figsize=(50, 40))

    row = 0

    col = 0

    for index in range(len(large_family_pics)):

        with open(large_family_pics[index], 'rb') as f:

            img = Image.open(f)

            ax[row][col].imshow(img)



            if(col < 5):

                col = col + 1

            else: 

                col = 0

                row = row + 1

    fig.show()

render_images(large_family_df)
large_family_df = train_images_df[train_images_df["familyId"]  == family_with_most_members.index[1]]

render_images(large_family_df)

render_bar_chart(large_family_df, 'uniqueId', 'Pic Count by Members', 'members')
large_family_df = train_images_df[train_images_df["familyId"]  == family_with_most_members.index[2]]

render_images(large_family_df)

render_bar_chart(large_family_df, 'uniqueId', 'Pic Count by Members', 'members')
train_df = pd.read_csv("../input/train_relationships.csv")

train_df.head()