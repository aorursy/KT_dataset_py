import seaborn as sb

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os

print(os.listdir("../input"))

df_mcq=pd.read_csv("../input/multipleChoiceResponses.csv",low_memory=False)

df_form=pd.read_csv('../input/freeFormResponses.csv',low_memory=False)

df_mcq.head()
a=df_mcq.loc[df_mcq['Q1'].isin(['Male','Female'])]
a.head()
plt.figure(figsize=(10,7))

sb.countplot(y='Q4',data=a,hue="Q1")

plt.xticks(rotation=-45)

plt.ylabel('The highest level of formal education ')

plt.legend(title='sex')
b=df_mcq.loc[df_mcq['Q1'].isin(['Male','Female']) & df_mcq['Q6'].isin(['Data Scientist','Data Analyst'])]
country=b.loc[df_mcq['Q3'].isin(['Russia','United States of America','India','China','Brazil'])]

plt.figure(figsize=(15,7))

sb.countplot(x='Q3',data=country,hue="Q6")

plt.xticks(rotation=-30)

plt.xlabel('country ')

plt.legend(title='Role')
plt.figure(figsize=(10,7))

sb.countplot(x='Q1',data=b,hue="Q6")

plt.xticks(rotation=-30)

plt.xlabel('Gender')

plt.legend(title='current role')
plt.figure(figsize=(10,7))

sb.countplot(y='Q4',data=b,hue="Q6")

plt.xticks(rotation=-90)

plt.ylabel('Level of education')

plt.legend(title='current role')
plt.figure(figsize=(10,7))

sb.countplot(x='Q2',data=b,hue="Q6")

plt.xticks(rotation=-30)

plt.xlabel('Age(# years)')

plt.legend(title='current role')
plt.figure(figsize=(10,7))

sb.countplot(x='Q2',data=b,hue="Q1")

plt.xticks(rotation=-30)

plt.xlabel('Gender')

plt.legend(title='sex')
c=b.loc[b['Q9']!='I do not wish to disclose my approximate yearly compensation']

plt.figure(figsize=(20,10))

sb.countplot(x='Q9',data=c,hue="Q1")

plt.xticks(rotation=-90)

plt.xlabel('Yearly compensation ')

plt.legend(title='sex')


plt.figure(figsize=(10,7))

sb.countplot(y='Q12_MULTIPLE_CHOICE',data=b,hue="Q1")

plt.xticks(rotation=-90)

plt.xlabel('Primary tool at work ')

plt.legend(title='sex')
plt.figure(figsize=(10,7))

sb.countplot(y='Q12_MULTIPLE_CHOICE',data=b,hue="Q6")

plt.xticks(rotation=-90)

plt.xlabel('Primary tool at work ')

plt.legend(title='Job Title')
## Using matplotlib pyplot we can increase the size of figure

fig,ax=plt.subplots()

fig.set_size_inches(40,25)

plt.scatter(x="Q2",y='Q3',data=b)
plt.figure(figsize=(10,7))

sb.countplot(y='Q2',data=b,hue="Q6")

plt.ylabel('age group')

plt.xticks(rotation=-45)

plt.legend(title='Job Title')
top_industry_catlog=df_form['Q7_OTHER_TEXT'].value_counts().head(10)
plt.figure(figsize=(10,7))

sb.barplot(top_industry_catlog.index,top_industry_catlog.values)

plt.xticks(rotation=-90)

plt.ylabel('count')

plt.xlabel('Top_industry')
for i in df_form.columns:

    top_industry_catlog=df_form[i].value_counts().head()

    print(i)

    sb.barplot(top_industry_catlog.index,top_industry_catlog.values)

    plt.xlabel(i)

    plt.ylabel("count")

    plt.show()
from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot

import cufflinks as cf

# for notebook

init_notebook_mode(connected=True)

# for offline use

cf.go_offline()
plt.figure(figsize=(20,20))

c[['Time from Start to Finish (seconds)','Q2','Q9']].scatter_matrix()