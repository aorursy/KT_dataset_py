import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline
import matplotlib.patches as mpatches


import plotly.plotly as py



import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()

import datetime as dt
import numpy as np
#extract data:

loan=pd.read_csv('../input/loan.csv')
loan.head(10)
#I am going to investigate 7 columns: loan amount, term, int rate, grade, issue date, loan status, address state

loan00 = loan.iloc[:,[2,5,6,8,15,16,23]]
loan00.head(10)
loan00.dtypes
#convert the data type from string to date 



loan00['issue_d2']=pd.to_datetime(loan00['issue_d'])
loan00.head(10)
loan00.dtypes
#select the month of the date, and calculate sum loan during each month 

loan01 = loan00.groupby(['issue_d2']).sum()

loan01=loan01.reset_index()

loan01['Issue_d']=[loan01.to_period('M')for 

                   loan01 in loan01['issue_d2']] #select month of date

loan01=loan01.drop(['issue_d2','int_rate'], axis=1)

loan01=loan01.groupby(['Issue_d']).sum()

loan01['loan_amnt']=loan01['loan_amnt']/1000
loan01.head(10)
#Use a graph to show the loan amount trend from 2007 to 2016

pic1 = loan01.plot(figsize=(20,10), fontsize=20,color='r')

plt.legend(loc='best',fontsize=0)

pic1.set_xlabel('Timeline', fontsize=25)

pic1.set_ylabel('Loan_Amount (000)',fontsize=25)

pic1.set_title('Accumulative Loan Amount from 2007 to 2015',fontsize=25)
#the curve starts to fluctuate since 2014, so let's take a closer look to the moving cycle

loan01_2=loan01.tail(24)
loan01_2
#Use a graph to show the loan amount trend of recent 2 years

pic3 = loan01_2.plot(figsize=(20,10), fontsize=20,color='r')

plt.legend(loc='best',fontsize=0)

pic3.set_xlabel('Timeline', fontsize=25)

pic3.set_ylabel('Loan_Amount (000)',fontsize=25)

pic3.set_title('Accumulative Loan Amount from 2014 to 2015',fontsize=25)
#I also wanted to see the loan amount in different states

loan02=loan00.iloc[:,[0,-2]]
loan02_1=loan02.groupby(['addr_state']).sum()
loan02_2=loan02_1.reset_index()
loan02_2['loan_amnt(mil.)']=loan02_2['loan_amnt']/1000000

loan02_2=loan02_2.drop(['loan_amnt'],axis=1)
loan02_2
#draw a color map of sum of loan in different states 

scale = [[0, 'rgb(229, 239, 245)'],[1, 'rgb(1, 97, 156)']]



[[0.0, 'rgb(223,221,228)'], [0.2, 'rgb(199,199,216)'], [0.4, 'rgb(169,170,201)'],[0.6, 'rgb(139,135,181)'], [0.8, 'rgb(98,88,158)'], [1.0, 'rgb(63,20,122)']]











data  = [dict(type='choropleth', colorscale=scale,

              autocolorscale = False,

              showscale = False, 

              locations=loan02_2['addr_state'],

              z=loan02_2['loan_amnt(mil.)'].astype(float),

              locationmode = 'USA-states',

              #text=loan02_2['text'], hoverinfo='location+z',

              marker= dict(line=dict(color='rgb(255,255,255)', width=2)),

              )]



layout = dict(title='Lending Club Loan Volumn Reginal Outlook <br />(Sum of Personal Loan Amount in Million)',

              geo = dict(scope='USA', projection=dict(type='albers usa'),

                         showlakes=True,

                        lakecolor='rgb(95,145,237)'))



fig = dict(data=data, layout=layout)



iplot(fig)
#I also wanted to investigate the relationship between interest rate of different grades and terms.

loan03=loan00.iloc[:,[1,2,3]]
loan03_1 = loan03.groupby(['grade','term']).mean()

loan03_1 = loan03_1.reset_index()

#Reorganize the data using pivot command

loan03_2 = loan03_1.pivot(index='grade', columns='term',values='int_rate')
loan03_2
#Draw a horizental bar chart to compare rates of different conditions

pic2=loan03_2.plot(kind='barh',figsize=(15,8),color=['peachpuff','darkkhaki'])

pic2.legend(loc='lower right',fontsize=12)

pic2.set_title('Average Interest Rate of Each Grade for 36 Month and 60 Month',fontsize=20)

pic2.set_xlabel('Interest Rate',fontsize=12)

loan00.head()
loan04=loan00.iloc[:,[-1,-3]]

loan04.describe().transpose()
loan04.head()
#I also wanted to see the content of ten different loan status:

pd.crosstab(index=loan04['loan_status'], columns='count')
loan04['issue_y']=loan04['issue_d2'].apply(lambda x: x.year)

loan04=loan04.drop(['issue_d2'],axis=1)

loan04.head()
loan04['count']=1
loan04.head()
loan04_2=loan04.groupby(['issue_y','loan_status']).sum()
loan04_2=loan04_2.unstack(level=1).fillna(0)

loan04_2.head()
loan04_3=loan04_2

loan04_3.describe().transpose().iloc[:,0]


loan04_3['total loan record']=np.sum(loan04_3, axis=1)

    
loan04_3
loan04_3['total default record']=np.sum(loan04_3[[0,2,3,8,9]], axis=1)
loan04_3['default rate']=round(loan04_3

                               ['total default record']

                               /loan04_3['total loan record'],4)
#Slice the records amount part:

loan04_4=loan04_3.iloc[:,[-3,-2]]
loan04_4
#draw a line chart to show the trend of loan records and defult loan records:



line1, = plt.plot(loan04_4['total loan record'], label="total loan record", linestyle='--')

line2, = plt.plot(loan04_4['total default record'], label="total default record", linewidth=2)

legend1 = plt.legend(handles=[line1], loc=1)

ax = plt.gca().add_artist(legend1)

plt.legend(handles=[line2], loc=2)

plt.ticklabel_format(useOffset=False)

plt.title('Total Default Records & Total Loan Records')

plt.show()
#Slice the rate part:

loan04_5=loan04_3.iloc[:,[-1]]
loan04_5
#Draw a grapg to learn the trend of default rate from 2007 to 2015

pic4=loan04_5.plot(figsize=(20,10), fontsize=20, color='orange',lw=3, legend=False)

pic4.set_title('Default Rate from 2007 to 2015', fontsize=20)

pic4.set_xlabel('Year', fontsize=18)

plt.ticklabel_format(useOffset=False)
loan05=loan00.iloc[:,[3,5]]

loan05['count']=1

loan05.head(10)
loan05_1=loan05.groupby(['grade','loan_status']).sum()

loan05_2=loan05_1.unstack(level=1)

loan05_2
loan05_3=loan05_2

loan05_3['total default record']=np.sum(loan05_3[[0,2,3,8,9]], axis=1)

loan05_3['total loan record']=np.sum(loan05_3, axis=1)

loan05_3
loan05_3['default rate']=round(loan05_3

                               ['total default record']

                               /loan05_3['total loan record'],4)

loan05_3
loan05_4=loan05_3.iloc[:,[-1]]

loan05_4
loan05_5=loan05_4.reset_index()

loan05_5
loan_merge=loan03_2.reset_index()

loan05_6=pd.merge(loan_merge,loan05_5, how='inner', on=['grade'])

loan05_6
loan05_7=loan05_6.iloc[:,[0,1,-1]]

loan05_7['default rate']=loan05_7.iloc[:,[-1]]*100

loan05_7=loan05_7.iloc[:,[0,1,-1]]

loan05_7['interest rate']=round((loan05_7.iloc[:,[1]]),2)

loan05_7=loan05_7.iloc[:,[0,-2,-1]]

loan05_7=loan05_7.set_index(['grade'])

loan05_7
fig,ax=plt.subplots()

loan05_7.plot(kind='bar', ax=ax, rot=0, color=['lightslategrey','orange'], 

              title='Comparison of 36 months interest rate and default rate', legend=False)



orange_patch = mpatches.Patch(color='orange', label='interest rate')

legend1=plt.legend(handles=[orange_patch], loc=(0.04,0.73))

ax = plt.gca().add_artist(legend1)

grey_patch = mpatches.Patch(color='grey', label='default rate')

plt.legend(handles=[grey_patch],loc=(0.04,0.86))
loan06=loan00.iloc[:,[-1,3]]

loan06['issue_y']=loan06['issue_d2'].apply(lambda x: x.year)

loan06=loan06.drop(['issue_d2'],axis=1)

loan06['count']=1
loan06=loan06.groupby(['issue_y','grade']).sum()
loan06
def grade(year):

    loan6_02=loan06.ix[year]

    colors = ['forestgreen', 'skyblue', 'lightcyan', 'darkkhaki', 'gold','teal','tomato']

    plt.figure(figsize=(5,5))

    figure=(15,15)

    fig1=plt.pie(loan6_02, colors=colors, shadow=False, startangle=140, autopct='%1.1f%%')

    plt.legend(labels=loan6_02.index, bbox_to_anchor=(0.05,0.8), loc='best', fontsize=12)

    plt.axis='equal'

    plt.title(year,fontsize=15)

    plt.show
grade(2007)
grade(2009)
grade(2012)
grade(2015)