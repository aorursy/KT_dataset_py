# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.info()
data.isnull().any()



# There is no null values.
data.head()
sex = ["male" if each == 1 else "female" for each in data.sex]

data.sex = sex
# Let take a look at the proportion of the patient's gender.



print(data.sex.value_counts())



sns.countplot(data.sex)

plt.xlabel("Sex")

plt.ylabel("Count")



# There are 207 male and 96 female patient.
# Distribution of the patient's ages.



data1 = [go.Histogram(x = data.age, opacity = 0.8, name = "Ages", marker = dict(color = "rgba(235,123,25,0.7)"))]



layout = go.Layout(title = "Distribution of Age", xaxis = dict(title = "Age"), yaxis = dict(title = "Count"))



fig = go.Figure(data = data1, layout = layout)



iplot(fig)

# Impact of gender and age on the resting electrocardiographic results (restecg)



plt.figure(figsize=(15,10))

sns.swarmplot(x = "sex", y = "age", hue = "restecg", data = data)

plt.show()



# Is there any correlation between maximum heart rate and serum cholesttoral amount.



import scipy.stats as stats



sns.jointplot(x = data.chol, y = data.trestbps, kind = "kde", size = 7).annotate(stats.pearsonr)

plt.show()



# Distribution of oldpeak values



data3 = go.Histogram(x = data.oldpeak, opacity = 0.5, name = "Oldpeak", marker = dict(color = "rgba(123,145,25,0.8)"))



layout = go.Layout(title = "Distribution of Oldpeak Values", xaxis = dict(title = "Oldpeak"), yaxis = dict(title = "Count"))



fig = go.Figure(data = data3, layout = layout)



iplot(fig)
ill = data[data.target == 1]



normal = data[data.target == 0]



ill.age.mean()

normal.age.mean()



print("Average age of ill patients is", ill.age.mean())

print("Average age of normal patients is", normal.age.mean())
trace1 = go.Bar(x = ["trestbps","chol","thalach"], y = [ill.trestbps.mean(),ill.chol.mean(),ill.thalach.mean()], name = "Ill")



trace2 = go.Bar(x = ["trestbps","chol","thalach"], y = [normal.trestbps.mean(),normal.chol.mean(),normal.thalach.mean()], name = "Normal")



data5 = [trace1,trace2]



layout = go.Layout(barmode = "group", title = "Ill vs Normal Patient")



fig =go.Figure(data = data5, layout = layout)



iplot(fig)