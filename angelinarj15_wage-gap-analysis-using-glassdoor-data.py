# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

%matplotlib inline

import plotly.graph_objects as go

from plotly import __version__

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

cf.go_offline()
# Reading the data

glassdoor_data=pd.read_csv("/kaggle/input/glassdoor-analyze-gender-pay-gap/Glassdoor Gender Pay Gap.csv")

glassdoor_data.shape
# Defining colour schemes to be used in the notebook for plotly plots

notebook_colours=["plum","slateblue","navy","firebrick",

                                      "darksalmon","slateblue","maroon","lightskyblue","blue","darkmagenta"]
# Checking the top 5 rows of the data

glassdoor_data.head()
# No null values in the data

glassdoor_data.isnull().sum()
# Summary Stats of numerical variables

glassdoor_data.describe()
# Summary stats of categorical variables

glassdoor_data.describe(include=np.object)
# Seniority has only 5 discrete levels and can be converted to a categorical variable

glassdoor_data.Seniority.unique()
# PerfEval is a rating and has only 5 discrete variables and can be converted to a categorical variable

glassdoor_data.PerfEval.unique()
glassdoor_data['Seniority'] = glassdoor_data['Seniority'].astype(object)

glassdoor_data['PerfEval'] = glassdoor_data['PerfEval'].astype(object)

glassdoor_data.info()
glassdoor_data['TotalPay']=glassdoor_data['BasePay']+glassdoor_data['Bonus']

glassdoor_data.head()
# Creating labels for creating age groups

labels = ['18 - 30', '31 - 42', '43 - 54', '55 - 65'] # 

labels



bins=[17,30,42,54,65] # the lower bin value is included in pd.cut 17---> starts at 18

age_binned=pd.cut(glassdoor_data['Age'],bins=bins,labels=labels)

age_binned.tail()



glassdoor_data['AgeBuckets']=age_binned

glassdoor_data['AgeBuckets']=glassdoor_data['AgeBuckets'].astype(object)

glassdoor_data.head()



glassdoor_data.describe(include=np.object)
# For categorical variables

iplot(cf.subplots([glassdoor_data['Gender'].figure(kind='hist',color=notebook_colours[0]),

                   glassdoor_data['AgeBuckets'].figure(kind='hist',color=notebook_colours[1]),

                   glassdoor_data['Seniority'].figure(kind='hist',color=notebook_colours[2]),

                   glassdoor_data['PerfEval'].figure(kind='hist',color=notebook_colours[4]),

                   glassdoor_data['Education'].figure(kind='hist',color=notebook_colours[6]),

                   glassdoor_data['Dept'].figure(kind='hist',color=notebook_colours[7]),

                   glassdoor_data['JobTitle'].figure(kind='hist',color=notebook_colours[8])],shape=(3,3)))
iplot(cf.subplots([glassdoor_data['BasePay'].figure(kind='hist',color=notebook_colours[8]),

                   glassdoor_data['Bonus'].figure(kind='hist',color=notebook_colours[6]),

                   glassdoor_data['TotalPay'].figure(kind='hist',color=notebook_colours[4])],shape=(3,1)))
# Creating list of categorical variables to iterate through

bivar=['JobTitle', 'PerfEval', 'Education', 'Dept', 'Seniority', 'AgeBuckets']



# Plotting count of males and females for each categorical variable

for i in bivar:

    fig=px.histogram(glassdoor_data,x=i,color='Gender',color_discrete_sequence=notebook_colours,

                     barmode='group',title='Gender diverisity across {}'.format(i))

    fig.show()
px.histogram(glassdoor_data,x="JobTitle",color="Education",barmode="stack",

             color_discrete_sequence=notebook_colours)
# Analyzing the average slaries of jobs within each department

px.histogram(glassdoor_data,x="Dept",y="BasePay",color="JobTitle",barmode="group",

             histfunc='avg',title='Base Pay offered within each department',

             color_discrete_sequence=(notebook_colours))
px.histogram(glassdoor_data,x="Dept",y="Bonus",color="JobTitle",barmode="group",histfunc='avg',

             title='Bonus offered within each department',

            color_discrete_sequence=(notebook_colours))
px.histogram(glassdoor_data,x="Dept",y="TotalPay",color="JobTitle",barmode="stack",histfunc='avg',

             title='Total pay offered within each department',

            color_discrete_sequence=(notebook_colours))
px.scatter(glassdoor_data, x="BasePay", y="Bonus",trendline="ols" ,color="Gender", 

           facet_col="Dept",color_discrete_sequence=notebook_colours)
# Reshaping the data to a matrix format for heatmap

seniority_pivot = glassdoor_data.pivot_table(index = 'Seniority',columns='JobTitle',values = 'TotalPay') 

#agg function by default is mean

seniority_pivot
# Heatmap of payscale with seniority

plt.figure(figsize=(12,6)) 

sns.heatmap(seniority_pivot,linewidths=1,linecolor='black',cmap='BuGn')
# At every level of seniority the quartiles for females are lower than males

px.box(glassdoor_data,x="Dept",y="TotalPay",color='Gender',color_discrete_sequence=notebook_colours)
# Calculating average pay across Department and gender

gender_dept_pay=glassdoor_data.groupby(['Dept','Gender'],axis=0,as_index=False).mean() 

gender_dept_pay
# Pivoting to get the data at required level

paygap_dept=gender_dept_pay.pivot(index='Dept',values=['Bonus','BasePay','TotalPay'],columns='Gender')

paygap_dept.head()
# Calculating difference in total pay

paygap_dept['DeptPayGap']=paygap_dept['TotalPay','Male']-paygap_dept['TotalPay','Female']

paygap_dept.head()
### Reshaping data to get average of pay in each for males and females in each job title within department

gender_job_dept_pay=glassdoor_data.groupby(['Dept','JobTitle','Gender'],axis=0,as_index=False).mean() 

gender_job_dept_pay
# Treemap with outermost layer as department, then job title and gender in the inner layer

fig = px.treemap(gender_job_dept_pay, path=['Dept','JobTitle','Gender'], values='TotalPay',

                 color='TotalPay', color_continuous_scale='bugn',

                 title="Earning disparity in Job Titles within each department",

                 labels={"TotalPay":'Average Total Pay'},width=1200, height=600)

fig.show()
### Reshaping data to get average of pay in each for males and females in each seniority level within department

gender_seniority_dept_pay=glassdoor_data.groupby(['Dept','Seniority','Gender'],axis=0,as_index=False).mean() 

gender_seniority_dept_pay.head(6)
# Scatter plot with average total pay and seniority

fig=px.scatter(gender_seniority_dept_pay,'Seniority',

            'TotalPay',color='Gender',size=(gender_seniority_dept_pay['TotalPay']/10000)-6, # factor of total pay calculated 

           color_discrete_sequence=notebook_colours, facet_col='Dept',labels={"TotalPay":'Average Total Pay'})



fig.show()
# Box plot to understand spread of performance rating

px.box(glassdoor_data,x="Dept",y="PerfEval",color='Gender',color_discrete_sequence=notebook_colours,

      title='Performance Evaluation in Departments')
# Aggregating numerical attributes at a Department, Evaluation and gender level

gender_eval_dept_pay=glassdoor_data.groupby(['Dept','PerfEval','Gender'],axis=0,as_index=False).mean() 

gender_eval_dept_pay.head()
# Scatter plot to understand pay disparity in performance ratings 

px.scatter(gender_eval_dept_pay,x="Dept",y="TotalPay",color='Gender',color_discrete_sequence=notebook_colours[6:8],

      title='Average Pay by performance evaluation',facet_col='PerfEval',size=(gender_eval_dept_pay['BasePay']/10000)-8)
### Reshaping data to get average of pay in each for males and females by educational background 

gender_ed_dept_pay=glassdoor_data.groupby(['Dept','Education','Gender'],axis=0,as_index=False).mean() 

gender_ed_dept_pay.head()
# Sunburst chart with innermost layer as education, then department and gender in the outest layer

fig = px.sunburst(gender_ed_dept_pay, path=['Dept','Education','Gender'], values='TotalPay',

                 color='TotalPay', color_continuous_scale='bugn',title="Earning disparity in levels of education",

                  labels={"TotalPay":'Average Total Pay'})

fig.show()