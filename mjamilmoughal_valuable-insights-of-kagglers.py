import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import squarify
plt.style.use("fivethirtyeight")
import warnings
warnings.filterwarnings('ignore')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import base64
import io
from scipy.misc import imread
import codecs
from IPython.display import HTML
from matplotlib_venn import venn2
from subprocess import check_output
data = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv", skiprows=[1])
data.head()
print('Total number of respondents:', data.shape[0])
print('Total number of Countries with respondents: ', data["Q3"].nunique())
print("Country with most respondents:", data['Q3'].value_counts().index[0], 'with', data['Q3'].value_counts().values[0], 'respondents')
data['Q3'].value_counts().sort_values(ascending=True).plot(kind='bar', title='Where are Kagglers residing?', figsize=(10,10))
plt.show()
plt.subplots(figsize=(22,12))
sns.countplot(y=data['Q1'],order=data['Q1'].value_counts().index)
plt.show()
sns.barplot(data['Q2'].value_counts().index, data['Q2'].value_counts().values, palette='inferno')
plt.title('Age ranges of Kagglers')
plt.xlabel('')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()
response_count=data['Q3'].value_counts()[:15].to_frame()
sns.barplot(response_count['Q3'],response_count.index,palette='inferno')
plt.title('Top 15 countries by Number of Respondents')
plt.xlabel('')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()
sns.barplot(data['Q9'].value_counts().index, data['Q9'].value_counts().values, palette='inferno')
plt.title('Salaries of Kagglers')
plt.xlabel('')
fig=plt.gcf()
fig.set_size_inches(20,8)
plt.show()
data['Q9'].value_counts()
data['Q5'].value_counts()
data['Q5'].value_counts().plot.bar(figsize=(15,10), fontsize=12)
plt.xlabel('UG Majors', fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.title('UG Majors - Split of respondents', fontsize=20)
plt.show()
temp_series=data['Q4'].value_counts()
labels=(np.array(temp_series.index))
sizes=(np.array((temp_series/temp_series.sum())*100))
trace=go.Pie(labels=labels, values=sizes, hole=0.5)
layout=go.Layout(title='Education distribution')
dataEducation=[trace]
fig=go.Figure(data=dataEducation, layout=layout)
py.iplot(fig, filename='Education')
data['Q6'].value_counts()
data['Q6'].value_counts().plot.bar(figsize=(15,10), fontsize=12)
plt.xlabel('Job Title', fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.title('Job Title - Split of respondents', fontsize=20)
plt.show()
data['Q7'].value_counts().sort_values(ascending=True).plot(kind='bar', title='In which industry Kagglers work?', figsize=(10,10))
plt.show()

