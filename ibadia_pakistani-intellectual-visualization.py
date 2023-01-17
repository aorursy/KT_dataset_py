import matplotlib.pyplot as plt

import plotly.plotly as py

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline

plt.rcParams['figure.figsize']=(12,5)

df=pd.read_csv("../input/Pakistan Intellectual Capital - Computer Science - Ver 1.csv", encoding = "ISO-8859-1")

df.head()
df.Department.groupby(df.Department).count().plot(kind="bar")
Specialization="Area of Specialization/Research Interests"

df[Specialization].value_counts().sort_values()[::-1][:20].plot(kind="bar")
province="Province University Located"

df[province].value_counts().sort_values().plot(kind="bar")
df["Country"].value_counts().sort_values()[::-1][1:].plot(kind="bar")