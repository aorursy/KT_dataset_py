# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

data.tail(10)
data.rename(columns={'pelvic_tilt numeric':'pelvic_tilt_numeric'}, inplace=True)

data.tail()
N=data[data['class']=="Normal"]

AN=data[data['class']=="Abnormal"]
data['class']=[1 if each =='Normal' else 0 for each in data['class']]

y=data['class'].values

x_data=data.drop(['class'],axis=1)

dataf=x_data.corr()

plt.figure(figsize=(10,10))

ax=sns.heatmap(dataf,annot=True, linewidths=1, fmt= '.2f', annot_kws={"size": 14})

plt.show()
index_vals = data['class'].astype('category').cat.codes

import plotly.graph_objects as go

fig = go.Figure(data=go.Splom(

                dimensions=[dict(label='pelv i',

                                 values=data['pelvic_incidence']),

                            dict(label='pelv t n.',

                                 values=data['pelvic_tilt_numeric']),

                            dict(label='lumbar',

                                 values=data['lumbar_lordosis_angle']),

                            dict(label='sacral',

                                 values=data['sacral_slope']),

                           dict(label='pelvic rad',

                                 values=data['pelvic_radius']),

                           dict(label='degree',

                                 values=data['degree_spondylolisthesis'])],

                text=data['class'],

                marker=dict(color=index_vals,

                            showscale=True, # colors encode categorical variables

                            line_color='white', line_width=0.75)

                ))



fig.update_layout(

    title='Biomechanical Orthopoedic Data',

    dragmode='select',

    width=1050,

    height=800,

    hovermode='closest',

)



fig.show()
fig = plt.figure(figsize=(16,6))

ax1 = plt.subplot(121)

ax1.scatter(AN.pelvic_tilt_numeric,AN.lumbar_lordosis_angle,color="maroon",label="Anormal")

ax1.scatter(N.pelvic_tilt_numeric,N.lumbar_lordosis_angle,color="lime",label="Normal")

ax1.legend()

ax1.grid(True)

ax2 = plt.subplot(122)

ax2.scatter(AN.sacral_slope,AN.lumbar_lordosis_angle,color="navy",label="Anormal")

ax2.scatter(N.sacral_slope,N.lumbar_lordosis_angle,color="cyan",label="Normal")

ax2.legend()

ax2.grid(True)

fig.show()
#split test data

x_train, x_test,y_train,y_test=train_test_split(x_data,y,test_size=0.3,random_state=42)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)

print("{} nn score: {}".format(3,knn.score(x_test,y_test)))
score_list=[]

for each in range(1,20):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,20),score_list)

plt.grid(True)

plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=12)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)

print("{} nn score: {}".format(3,knn.score(x_test,y_test)))