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
import seaborn as sn 
import matplotlib.pyplot as plt 
import datetime as dt 
data=pd.read_csv('/kaggle/input/indian-startup-funding/startup_funding.csv')
data=data.replace('\\xc2\\xa0 ','').replace('\\\\xc2\\\\xa0','')
##renaming columns 

data.columns = ['SNo','Date','Startup_Name','Industry_Vertical','SubVertical','City','Investors_Name','InvestmentType','Amount','Remarks']

data.head()
data.info()
data.Amount.isna().sum().sum()
data.shape
3044-960
data.Amount.sort_values()[2050:2085]
#Converting Amount to Numerical so that we can use it in Visualisation 

def change_amt(amount):
    amt=[]
    for x in amount:
        x=str(x).replace(",","")
        x=str(x).lower().replace("\\xc2\\xa0","")
        try:
            x=int(x)
        except:
            x= -999 
        amt.append(x)
        
    return amt
        
data.Amount=change_amt(data.Amount)
data.head(50)
import warnings 
warnings. filterwarnings("ignore")
df = pd.DataFrame({'date_time' : data.Date})
df['correct'] = pd.to_datetime(df['date_time'],errors='coerce',format="%d/%m/%Y")
df[df.correct.isnull()]
#Cleaning  dates

data['Date'][data['Date']=='12/05.2015'] = '12/05/2015'
data['Date'][data['Date']=='13/04.2015'] = '13/04/2015'
data['Date'][data['Date']=='15/01.2015'] = '15/01/2015'
data['Date'][data['Date']=='22/01//2015'] = '22/01/2015'
data['Date'][data['Date']=='05/072018'] = '05/07/2018'
data['Date'][data['Date']=='01/07/015'] = '01/07/2015'
data['Date'][data['Date']=='\\\\xc2\\\\xa010/7/2015'] = '10/07/2015'
data.Date= pd.to_datetime(data['Date'],errors='coerce',format="%d/%m/%Y")
data.head()
data.Date.dt.month.value_counts()
bar_var=data.Date.dt.year.value_counts()

ax = sn.barplot(bar_var.index,bar_var.values)
ax.set(xlabel='Year', ylabel='No. of Fundings Made',title="No. Of Funding V/S Year")
plt.show()
data.head()
data_cleaned=data[data.Amount!=-999]


data_cleaned['year']=data_cleaned.Date.dt.year

data_year=data_cleaned.groupby('year',as_index=False)['Amount'].sum()

ax = sn.barplot(data_year.year,data_year.Amount,data=data_year)
ax.set(xlabel='Year', ylabel='Amount of Funding',title="Amount V/S Year")
plt.show()
invest_count=data_cleaned.InvestmentType.value_counts()
invest_count
import re
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
data['InvestmentType'] = data['InvestmentType'].apply(lambda x: remove_punctuation(str(x)))

funding_map = {
    "SeedAngel Funding": "Seed Angel Funding",
    "SeedFunding": "Seed Funding",
    "PrivateEquity": "Private Equity",
    "Crowd funding": "Crowd Funding",
    "Angel  Seed Funding": "Seed Angel Funding",
    "Seed  Angel Funding": "Seed Angel Funding",
    "Seed Angle Funding": "Seed Angel Funding",
    "Seed  Angle Funding": "Seed Angel Funding",
    "SeednFunding": "Seed Funding",
    "Seed funding": "Seed Funding",
    "Seed Round": "Seed Funding",
    "preSeries A": "PreSeries A",
    "preseries A": "PreSeries A",
    "Pre Series A": "PreSeries A"
}

for i, v in funding_map.items():
    data['InvestmentType'][data['InvestmentType']==i] = v 
    
data.InvestmentType.value_counts()
invest_count=data.InvestmentType.value_counts()
invest_count=invest_count.sort_values(ascending=False)[0:10]

invest=pd.DataFrame()

invest['InvestmentType']=invest_count.index
invest['Count']=invest_count.values


ax = sn.barplot(x='Count',y='InvestmentType',data=invest)
ax.set(xlabel='Count', ylabel='Investment Type',title="Top 10 Investment Type V/S Count")

for p in ax.patches:
    width = p.get_width()
    ax.text(width -1.5  ,
            p.get_y()+p.get_height()/2. + 0.2,
            '{:1.0f}'.format(width),
            ha="left")
plt.show()

data.head()
location_map = {
    "Bengaluru": "Bangalore",
    "Delhi": "NCR",
    "New Delhi": "NCR",
    "Gurugram": "NCR",
    "Gurgaon": "NCR",
    "Noida": "NCR"
}
for i, v in location_map.items():
    data['City'][data['City']==i] = v 

location_count=data.City.value_counts()
location_count=location_count.sort_values(ascending=False)[0:10]

location=pd.DataFrame()

location['City']=location_count.index
location['Count']=location_count.values

ax = sn.barplot(x='Count',y='City',data=location)
ax.set(xlabel='Count', ylabel='City',title="TOP 10 City V/S Startup Count")

for p in ax.patches:
    width = p.get_width()
    ax.text(width -1.5 ,
            p.get_y()+p.get_height()/2. + 0.1,
            '{:1.0f}'.format(width),
            ha="left")

plt.show()

Investors_Name=data[~data.Investors_Name.isna()]

data.Investors_Name.head(50)
from wordcloud import WordCloud, STOPWORDS
import random
random.seed(123)

inv_names = []
for invs in Investors_Name['Investors_Name']:
    for inv in str(invs).split(","):
        if inv != "":
            inv_names.append(inv.strip().lower().replace(" ","_").replace("'",""))

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    #return "hsl(0, 0%%, %d%%)" % (font_size*3)
    return (100, 100, font_size*3)

def plot_wordcloud(text, mask=None, max_words=40, max_font_size=80, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=grey_color_func):
    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    prefer_horizontal = 1.0,
                    max_font_size = max_font_size, 
                    min_font_size = 10,
                    random_state = 42,
                    #color_func = lambda *args, **kwargs: (140,0,0),
                    #color_func = color_map(),
                    colormap="Blues",
                    width=1200, 
                    height=600,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        #image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_color), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size, 'color': 'blue',
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'blue', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
plot_wordcloud(' '.join(inv_names), title="Investors with most number of funding deals")
industry_df=data_cleaned.groupby('Industry_Vertical',as_index=False)['Amount'].sum()

industry_df=industry_df.sort_values(by='Amount',ascending=False)[0:10]
industry_df
vertical_map = {
    "ECommerce": "eCommerce",
    "E-Commerce": "eCommerce",
    
}
for i, v in vertical_map.items():
    data_cleaned['Industry_Vertical'][data_cleaned['Industry_Vertical']==i] = v 

    
industry_df=data_cleaned.groupby('Industry_Vertical',as_index=False)['Amount'].sum()

industry_df=industry_df.sort_values(by='Amount',ascending=False)[0:10]
industry_df


ax = sn.barplot(x='Amount',y='Industry_Vertical',data=industry_df)
ax.set(xlabel='Amount', ylabel='Industry_Vertical',title="TOP 10 Industry Vertical V/S Amount Invested")

for p in ax.patches:
    width = p.get_width()
    ax.text(width -1.5 ,
            p.get_y()+p.get_height()/2. + 0.1,
            '{:1.0f}'.format(width),
            ha="left")

plt.show()

startup_df=data_cleaned.groupby('Startup_Name',as_index=False)['Amount'].sum()
startup_df.head()
startup_df
startup_df=startup_df.sort_values(by='Amount',ascending=False)[0:10]

ax = sn.barplot(x='Amount',y='Startup_Name',data=startup_df)
ax.set(xlabel='Amount', ylabel='Startup',title="TOP 10 Startup V/S Amount Invested")

for p in ax.patches:
    width = p.get_width()
    ax.text(width -1.5 ,
            p.get_y()+p.get_height()/2. + 0.1,
            '{:1.0f}'.format(width),
            ha="left")

plt.show()

