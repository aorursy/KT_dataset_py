
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import plotly.express as px
import plotly.offline as of


import os

paths = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pt = os.path.join(dirname, filename)
        print(pt)
        paths.append(pt)


df1 = pd.read_csv(paths[1])

df1.head()
df1['Age'].hist(bins = 25, figsize=(13,13))
plt.xlabel('Age of players')
plt.ylabel('Count of players')
plt.title("Histogram of players' age")

fig = plt.gcf()
fig.savefig('plot1.png')
df1.plot.scatter('Overall','Potential', figsize=(13,13), marker = 'o', c = 'Age', colormap='viridis')
plt.xlabel('Overall')
plt.title("Scatter plot of overall and potential")

fig = plt.gcf()
fig.savefig('plot2.png')
fig = px.scatter(df1, x="Age", y="Overall", color="Nationality",
                 size='Potential', hover_data = ['Name'],
                title = 'Overall by Age')

fig.show()

of.plot(fig, filename='fig1.html')
df2 = df1.dropna()
df2 = df2[ (df2['Club'] == 'FC Barcelona') | (df2['Club'] == 'Real Madrid CF') | (df2['Club'] == 'Chelsea')]
df2.head()

fig = px.histogram(df2, x="Age", color="Club", nbins =10, opacity = 0.8, barmode="group",

                title = 'Count of players by Age (Histogram)')
fig.show()

of.plot(fig, filename='fig2.html')