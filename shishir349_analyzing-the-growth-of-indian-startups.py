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
data=pd.read_csv('/kaggle/input/indian-startup-funding/startup_funding.csv')
import numpy as np 



# for dataframe operations

import pandas as pd 



# for data visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# lets select  a background style

plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (16, 7)



# for interactivity

import ipywidgets as widgets

from ipywidgets import interact

from ipywidgets import interact_manual
# lets import the data



# changing the names of the columns inside the data

data.columns = ["SNo", "Date", "StartupName", "IndustryVertical", "SubVertical",

                        "City", "InvestorsName", "InvestmentType", "AmountInUSD", "Remarks"]



# lets clean the strings

def clean_string(x):

    return str(x).replace("\\xc2\\xa0","").replace("\\\\xc2\\\\xa0", "")



# lets apply the function to clean the data

for col in ["StartupName", "IndustryVertical", "SubVertical", "City", 

            "InvestorsName", "InvestmentType", "AmountInUSD", "Remarks"]:

    data[col] = data[col].apply(lambda x: clean_string(x))

    

# lets check the head of the data

data.head()
data[data['StartupName']=='LimeRoad.com']
# missing data 



# lets import warnings module

import warnings

warnings.filterwarnings('ignore')



# lets calculate the total missing values in the data

total = data.isnull().sum().sort_values(ascending = False)



# lets calculate the percentage of missing values in the data

percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending = False)



# lets store the above two values in a dataset called missing data

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent %'])



# lets check the head of the data

missing_data
data['Remarks'].value_counts()
# lets remove Remarks column, as it contains a lot of nans, and high cardinal column



data = data.drop(['Remarks'], axis = 1)



# lets check the column names after removing the Remarks Column, as it having 

data.columns
# lets convert the amount column into numerical, so that we can analyze the values inside it



# function to clean the AmounInUsd Column

def clean_amount(x):

    x = ''.join([c for c in str(x) if c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']])

    x = str(x).replace(",","").replace("+","")

    x = str(x).lower().replace("undisclosed","")

    x = str(x).lower().replace("n/a","")

    if x == '':

        x = '-999'

    return x



# lets apply the function on the column

data["AmountInUSD"] = data["AmountInUSD"].apply(lambda x: float(clean_amount(x)))



# lets check the head of the column after cleaning it

plt.rcParams['figure.figsize'] = (15, 3)

data['AmountInUSD'].plot(kind = 'line', color = 'black')

plt.title('Distribution of Amount', fontsize = 15)

plt.show()
## Cleaning the dates



data['Date'][data['Date']=='12/05.2015'] = '12/05/2015'

data['Date'][data['Date']=='13/04.2015'] = '13/04/2015'

data['Date'][data['Date']=='15/01.2015'] = '15/01/2015'

data['Date'][data['Date']=='22/01//2015'] = '22/01/2015'

data['Date'][data['Date']=='05/072018'] = '05/07/2018'

data['Date'][data['Date']=='01/07/015'] = '01/07/2015'

data['Date'][data['Date']=='\\\\xc2\\\\xa010/7/2015'] = '10/07/2015'
# some of the date formats are written wrongly so we have to clean them thoroughly so that we can analyze these values



# converting them into a Datetime object

data["yearmonth"] = (pd.to_datetime(data['Date'],

         format='%d/%m/%Y').dt.year*100)+(pd.to_datetime(data['Date'],format='%d/%m/%Y').dt.month)





temp = data['yearmonth'].value_counts().sort_values(ascending = False).head(10)

print("Number of funding per month in decreasing order(Top 10)\n",temp)

year_month = data['yearmonth'].value_counts()



# lets plot the data

plt.rcParams['figure.figsize'] = (15, 7)

sns.barplot(year_month.index, year_month.values, palette = 'copper')

plt.xticks(rotation = 90)

plt.xlabel('Year-Month of transaction', fontsize=12)

plt.ylabel('Number of fundings made', fontsize=12)

plt.title("Year-Month Distribution", fontsize=16)

plt.show()
# lets check the Maximum funding of a startup

print("Maximum funding to a Startups is : ", data["AmountInUSD"].dropna().sort_values().max())
# lets check the startups with more than 50crore+ funding



@interact

def check(column = 'AmountInUSD', x = 500000000): # 50 crore funding of startups

    return data[data[column] > x].sort_values(by = 'AmountInUSD', ascending = False)
data[data['AmountInUSD'] == -999.0].shape
# Average Funding

print("On Average indian startups got funding of : ", data["AmountInUSD"].dropna().sort_values().mean())
# lets check the no. of funding each startsup got



print("Total startups funded : ", len(data["StartupName"].unique()))

print(data["StartupName"].value_counts().head(10))

startupname = data['StartupName'].value_counts().head(20)



# lets plot the data

plt.rcParams['figure.figsize'] = (12, 15)

sns.barplot(y = startupname.index, x = startupname.values, alpha=0.9, palette = 'Dark2')

plt.xticks(rotation='vertical')

plt.xlabel('Startup Name', fontsize=12)

plt.ylabel('Number of fundings made', fontsize=12)

plt.title("Number of fundings Startups Have", fontsize=16)

plt.show()
# lets analyze the Industry of the Startups



# data cleaning

data['IndustryVertical'] = data['IndustryVertical'].replace('nan', 'Consumer Technology')



industry = data['IndustryVertical'].value_counts().head(10)

print(industry)



# lets plot the data

plt.rcParams['figure.figsize'] = (15, 5)

sns.barplot(industry.index, industry.values, palette = 'autumn')

plt.xticks(rotation='vertical')

plt.xlabel('Industry vertical of startups', fontsize=12)

plt.ylabel('Number of fundings made', fontsize=12)

plt.title("Industry vertical of startups with number of funding", fontsize=16)

plt.show()
from wordcloud import WordCloud



names = data["InvestorsName"][~pd.isnull(data["InvestorsName"])]

wordcloud = WordCloud(max_font_size=50, width=600, height=300, background_color = 'cyan').generate(' '.join(names))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.title("Wordcloud for Investor Names", fontsize=35)

plt.axis("off")

plt.show()
# analyzing the effect of Cities on a Startup



# lets clean the data for better analysis

data['City'] = data['City'].replace(('Bengaluru', 'nan'),('Bangalore', 'Bangalore'))



city = data['City'].value_counts().head(10)

print(city)



# lets plot the data

sns.barplot(city.index, city.values, palette = 'Wistia')

plt.xticks(rotation='vertical')

plt.xlabel('city location of startups', fontsize=12)

plt.ylabel('Number of fundings made', fontsize=12)

plt.title("city location of startups with number of funding", fontsize=16)

plt.show()
# lets analyze the investors on startups



# lets clean the dataset

data['InvestorsName'][data['InvestorsName'] == 'Undisclosed investors'] = 'Undisclosed Investors'

data['InvestorsName'][data['InvestorsName'] == 'undisclosed Investors'] = 'Undisclosed Investors'

data['InvestorsName'][data['InvestorsName'] == 'undisclosed investors'] = 'Undisclosed Investors'

data['InvestorsName'][data['InvestorsName'] == 'Undisclosed investor'] = 'Undisclosed Investors'

data['InvestorsName'][data['InvestorsName'] == 'Undisclosed Investor'] = 'Undisclosed Investors'

data['InvestorsName'][data['InvestorsName'] == 'Undisclosed'] = 'Undisclosed Investors'

data['InvestorsName'][data['InvestorsName'] == 'nan'] = 'Undisclosed Investors'



# lets check the value counts

investors = data['InvestorsName'].value_counts().head(10)

print(investors)



# lets plot the data

sns.barplot(investors.index, investors.values, palette = 'cool')

plt.xticks(rotation='vertical')

plt.xlabel('Investors Names', fontsize=12)

plt.ylabel('Number of fundings made', fontsize=12)

plt.title("Investors Names with number of funding", fontsize=16)

plt.show()