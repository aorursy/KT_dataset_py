from IPython.display import YouTubeVideo

YouTubeVideo('7cEd2ZrItNg')
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
train_df = pd.read_csv("../input/train.csv")

print("Shape of train data: {0}".format(train_df.shape))

test_df = pd.read_csv("../input/test.csv")

print("Shape of test data: {0}".format(test_df.shape))



diagnosis_df = pd.DataFrame({

    'diagnosis': [0, 1, 2, 3, 4],

    'diagnosis_label': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

})



train_df = train_df.merge(diagnosis_df, how="left", on="diagnosis")



train_image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("../input/train_images")) for f in fn]

train_images_df = pd.DataFrame({

    'files': train_image_files,

    'id_code': [file.split('/')[3].split('.')[0] for file in train_image_files],

})

train_df = train_df.merge(train_images_df, how="left", on="id_code")

del train_images_df

print("Shape of train data: {0}".format(train_df.shape))



test_image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("../input/test_images")) for f in fn]

test_images_df = pd.DataFrame({

    'files': test_image_files,

    'id_code': [file.split('/')[3].split('.')[0] for file in test_image_files],

})





test_df = test_df.merge(test_images_df, how="left", on="id_code")

del test_images_df

print("Shape of test data: {0}".format(test_df.shape))



train_df.head()
test_df.head()
print("Number of unique diagnosis: {0}".format(train_df.diagnosis.nunique()))

diagnosis_count = train_df.diagnosis.value_counts()
import matplotlib.pyplot as plt

from PIL import Image



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 9999

pd.options.display.float_format = '{:20, .2f}'.format
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

    

    

render_bar_chart(train_df, 'diagnosis_label', 'Diabetic Retinopathy: Observation Distribution by Severity ', 'members')
SAMPLES_TO_EXAMINE = 5

import cv2

def render_images(files):

    plt.figure(figsize=(50, 50))

    row = 1

    for an_image in files:

        image = cv2.imread(an_image)[..., [2, 1, 0]]

        plt.subplot(6, 5, row)

        plt.imshow(image)

        row += 1

    plt.show()

    

no_dr_pics = train_df[train_df["diagnosis"] == 0].sample(SAMPLES_TO_EXAMINE)

render_images(no_dr_pics.files.values)
mild_dr_pics = train_df[train_df["diagnosis"] == 1].sample(SAMPLES_TO_EXAMINE)

render_images(mild_dr_pics.files.values)
moderate_dr_pics = train_df[train_df["diagnosis"] == 2].sample(SAMPLES_TO_EXAMINE)

render_images(moderate_dr_pics.files.values)
severe_dr_pics = train_df[train_df["diagnosis"] == 3].sample(SAMPLES_TO_EXAMINE)

render_images(severe_dr_pics.files.values)
preoliferative_dr_pics = train_df[train_df["diagnosis"] == 4].sample(SAMPLES_TO_EXAMINE)

render_images(preoliferative_dr_pics.files.values)