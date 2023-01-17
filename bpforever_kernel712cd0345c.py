import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

##for data visulatization
##lets select a background style , not important just for representation

plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize']=(36,7)
#for interactivity

import ipywidgets as widgets

from ipywidgets import interact

from ipywidgets import interact_manual
data = pd.read_csv('startup_funding.csv')
#changing the names of the  columns inside the data 

print(data)
data.columns =['Sno','Data','StartupName','Industryvertical','SubVertical','City','InvestorSName','InvestmentType','AmountInUSD','Remarks']
#lets clean the strings

def clean_string(x):

    return str(x).replace('\\xc2\\xa0','').replace('\\\\xc2\\\\xa0','')

## if you dont understand what happended here go to this url:https://stackoverflow.com/questions/10993612/how-to-remove-xa0-from-string-in-python
#lets apply the function to clean the data

for col in [

'StartupName','Industryvertical','SubVertical','City','InvestorSName','InvestmentType','AmountInUSD','Remarks']:

    data[col] = data[col].apply(lambda x:clean_string(x))

#lambda function is very uselfull for data cleaning purposes so pandas has an function called .apply() in which pandas performs the task for all x . for more infor refer url:https://medium.com/@chaimgluck1/have-messy-text-data-clean-it-with-simple-lambda-functions-645918fcc2fc    



#lets check the head of the data

data.head()
import warnings 

warnings.filterwarnings('ignore') 

## i dont think i need to expalin this.
#lets calculate the total missing values in the data

total = data.isnull().sum().sort_values(ascending =False)

#percentage of missing data 

percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending =False)
#for representation point of view i will stopre the two values in a dataset called missing_data

missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent %'])
#lets check the head of the data

missing_data
#lets check the values in the remark column

data['Remarks'].value_counts()
##AS YOU CAN OBSERVE REMARK HAS LOT OP NAN AND HIGH CORNDINAL VALUES

data = data.drop(['Remarks'],axis=1)

data.columns
print(data['AmountInUSD'])
### now this is little bit tricky u need to observe how the amount value is stored in the columnd AmountinUsd

def clean_amount(x):

    x = ''.join([c for c in str(x) if c in ['0','1','2','3','4','5','6','7','8','9']])

    #we are replacing each anomaly with empty string ""

    x=str(x).replace(",","").replace("+","")

    x=str(x).lower().replace("ubdisclosed","")

    x=str(x).lower().replace("n/a","")

    if x == '':

        x = '-9'

    return x

##now the -999 is to label that a startup has not got any funding

#lets apply the function on the column

data["AmountInUSD"] = data["AmountInUSD"].apply(lambda x:float(clean_amount(x)))

print(data['AmountInUSD'])







#lets check the head of the column after cleaning it

plt.rcParams['figure.figsize'] = (15,3)

data['AmountInUSD'].plot(kind='line',color ='black')

plt.title('Distribution of Amount',fontsize =15)

plt.show()
print(data['Data'].head())
data['Data'].dtype

## note 

## When you see dtype('O') inside dataframe this means Pandas string.
data['Data'][data['Data'] =='12/05.2015']='12/05/2015'

data['Data'][data['Data'] == '13/04.2015']='13/04/2015'

data['Data'][data['Data'] == '15/01.2015'] ='15/01/2015'

data['Data'][data['Data'] == '22/01//2015'] ='22/01/2015'

data['Data'][data['Data'] == '05/072018']='05/07/2018'

data['Data'][data['Data'] == '01/07/015'] ='01/07/2015'

data['Data'][data['Data'] == '\\\\xc2\\\\xa010/7/2015'] ='10/07/2015'

###how did i know which one of them is not right , refer the next code cell if something is not formated correctly it will give the error below 

#converting them into a Datetime object

data['yearmonth'] = (pd.to_datetime(data['Data'],

                                   format="%d/%m/%Y").dt.year*100)+(pd.to_datetime(data['Data'],format = "%d/%m/%Y").dt.month)

print(data['yearmonth'])

temp=data['yearmonth'].value_counts().sort_values(ascending =False).head(10)
print("No. of funding per month in decreasing order (TOp 10)\n",temp)
year_month = data['yearmonth'].value_counts()
#lets plot the data

plt.rcParams['figure.figsize'] = (15,7)

sns.barplot(year_month.index,year_month.values,palette= 'copper')

plt.xticks(rotation =90)

plt.xlabel('year-month of transaction',fontsize=12)

plt.ylabel('No. of funding made',fontsize=12)

plt.title('year-month Distribution',fontsize=16)

plt.show()
#lets check the maximum funding of a startup

print("maximum funding to a startup is :",data['AmountInUSD'].dropna().sort_values().max())
#lets check the startips with more than 50crore+ funding

@interact

def check(column = 'AmountInUSD', x= 50000):#50 crore funding of startups

    return data[data[column] > x].sort_values(by = 'AmountInUSD',ascending =False)
# lets check out different ventures of PATYM

data[data.StartupName=='Paytm']
#lets check the minimum funding in a startup

print('Minimum funding to a Startup is :',data['AmountInUSD'].dropna().sort_values().min())
#lets check the startups with least funding 

data[['AmountInUSD','StartupName']].sort_values(by= 'AmountInUSD',ascending=True).head()