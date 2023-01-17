

import numpy as np 

import pandas as pd 

from subprocess import check_output

%matplotlib notebook

import matplotlib.pyplot as plt

import seaborn as sns

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv(r'../input/h1b_kaggle.csv',na_values = '\xa0')

data.head()
data.describe()
data.isnull().sum()
data['YEAR'] = data['YEAR'].astype('category')

data.head()

data['CASE_STATUS'].value_counts()
data_certified = data[data['CASE_STATUS']=='CERTIFIED']



top10_companies = data_certified.groupby(['EMPLOYER_NAME'])['EMPLOYER_NAME'].count()

top10_companies = top10_companies.sort_values(ascending = False)

top10_companies = pd.DataFrame({"EMPLOYER_NAME":top10_companies.index,'No_of_petitions':top10_companies.values})

top10_companies = top10_companies[0:15]

top10_companies

plt.figure(figsize =(25,10))

ax = sns.barplot(x= "No_of_petitions",y= "EMPLOYER_NAME",data= top10_companies)
top10_title = data_certified.groupby(['JOB_TITLE'])['JOB_TITLE'].count()

top10_title = top10_title.sort_values(ascending = False)

top10_title = pd.DataFrame({"JOB_TITLE":top10_title.index,"No_of_petitions":top10_title.values})

top10_title = top10_title[0:15]

plt.figure(figsize =(25,10))

ax = sns.barplot(x= "No_of_petitions",y= "JOB_TITLE",data= top10_title)
payScale = data_certified.groupby(['JOB_TITLE'])['PREVAILING_WAGE'].mean()

payScale = payScale.sort_values(ascending = False)

payScale = pd.DataFrame({"JOB_TITLE":payScale.index,"Avg_Wage":payScale.values})

top10_title = data_certified.groupby(['JOB_TITLE'])['JOB_TITLE'].count()

top10_title = top10_title.sort_values(ascending = False)

top10_title = pd.DataFrame({"JOB_TITLE":top10_title.index,"No_of_petitions":top10_title.values})

top10_title.index = top10_title['JOB_TITLE']

payScale.index = payScale['JOB_TITLE']

new = payScale.merge(top10_title,on = 'JOB_TITLE',how = 'left')



# libraries

import numpy as np

import matplotlib.pyplot as plt

 

# set width of bar

barWidth = 0.25

 

# set height of bar

bars1 = new['Avg_Wage']

bars2 = new['No_of_petitions']

 

# Set position of bar on X axis

r1 = np.arange(len(bars1))

r2 = [x + barWidth for x in r1]

 

# Make the plot

plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1')

plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='var2')

 

# Add xticks on the middle of the group bars

plt.xlabel('group', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(bars1))], new['JOB_TITLE'])

 

# Create legend & Show graphic

plt.legend()

plt.show()

data_engg = data[data['JOB_TITLE']=='DATA ENGINEER']
%matplotlib notebook

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns
a = data_engg['YEAR'].value_counts()

df = pd.DataFrame({"Year":a.index,"No_of_petitions":a.values})

plt.figure(figsize = (8,10))

ax = sns.barplot(x="Year",y="No_of_petitions",data = df)



data['JOB_TITLE'].value_counts()
len(data_hardWare)