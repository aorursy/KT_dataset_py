# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import dataset

import pandas as pd

all_data = pd.read_csv("../input/pew-research-core-trends-survey-jan-2018/January 3-10 2018 - Core Trends Survey - SPSS.csv")



#store full text column names in shorter variables

state_index = "State based on self-reported zipcode"

opinion_society = "PIAL11. Overall, when you add up all the advantages and disadvantages of the internet, would you say the internet has mostly been [ROTATE: (a GOOD thing) or (a BAD thing)] for society?"

opinion_self="PIAL12. How about you, personally? Overall, when you add up all the advantages and disadvantages of the internet, would you say the internet has mostly been [ROTATE IN SAME ORDER AS PIAL11: (a GOOD thing) or (a BAD thing)] for you?"



#Get a subset of dataframe with only state information 

#and answers to two internet opinion questions

df =all_data[{state_index,opinion_society,opinion_self}]

df=df.reindex(columns=[state_index,opinion_society,opinion_self]) #reorder columns





#Aggregate data by state for Effect on Self Question

df_self=df.pivot_table(index=state_index, columns=opinion_self, aggfunc='size', fill_value=0)

df_self.tail()



#Aggregate data by state for Effect on Society Question

df_society=df.pivot_table(index=state_index, columns=opinion_society, aggfunc='size', fill_value=0)

df_society.tail()

#below is the code I used to access the BLS API

#the service kept going down as I was working on this project, so I abandoned this approach for a static dataset



#get get to BLS API

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

blsKey = user_secrets.get_secret("blsKey")



import urllib.error, urllib.parse, urllib.request, json, datetime



def safeGet(url):

    try:

        return urllib.request.urlopen(url)

    except urllib.error.URLError as e:

        if hasattr(e,"code"):

            print("The server couldn't fulfill the request.")

            print("Error code: ", e.code)

        elif hasattr(e,'reason'):

            print("We failed to reach a server")

            print("Reason: ", e.reason)

        return None



#BLS Reference for constructing a series ID

# Series ID example:   OEUN000000011100011000001

#Positions       Value           Field Name

#1-2             OE              Prefix

#3               U               Seasonal Adjustment Code

#4               N               Area Type Code

#5-11            0000000         Area Code

#12-17           111000          Industry Code

#18-23           110000          Occupation Code

#24-25           01              Data Type Code



seriesID='OE'+'U'+'N'+'0000000'+'541500'+'000000'+'01' #541500 is the industry code for 'Computer Systems Design and Related Services', which is the closest match I found for the tech industry

year='2018'

url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'+seriesID+'/?registrationkey='+blsKey+'&startyear='+year+'&endyear='+year



#turn URL result into JSON

data=json.load(safeGet(url))



#accessing value from JSON 

total_employment_computer_systems = data['Results']['series'][0]['data'][0]['value']

print(total_employment_computer_systems)



#PSEUDOCODE FOR WHAT I WOULD HAVE DONE NEXT

#list all 50 states and their corresponding Area Code to use in the SeriesID (list available from BLS, but could not access because site was down)

#loop through list, creating a seriesID for each state

#get data via API for each state and store it in a dataframe
import pandas as pd

df_empl = pd.read_csv("../input/tech-employment-by-state/Tech employment by state.csv")





#dictionary of state abbreviations

#source: https://gist.github.com/rogerallen/1583593

states = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Hawaii': 'HI',

    'Idaho': 'ID',

    'Illinois': 'IL',

    'Indiana': 'IN',

    'Iowa': 'IA',

    'Kansas': 'KS',

    'Kentucky': 'KY',

    'Louisiana': 'LA',

    'Maine': 'ME',

    'Maryland': 'MD',

    'Massachusetts': 'MA',

    'Michigan': 'MI',

    'Minnesota': 'MN',

    'Mississippi': 'MS',

    'Missouri': 'MO',

    'Montana': 'MT',

    'Nebraska': 'NE',

    'Nevada': 'NV',

    'New Hampshire': 'NH',

    'New Jersey': 'NJ',

    'New Mexico': 'NM',

    'New York': 'NY',

    'North Carolina': 'NC',

    'North Dakota': 'ND',

    'Northern Mariana Islands':'MP',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Palau': 'PW',

    'Pennsylvania': 'PA',

    'Puerto Rico': 'PR',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virgin Islands': 'VI',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY',

}



      

#replace state column with abbreviations

for index, row in df_empl.iterrows(): 

    state_name=row["State"]

    state_abbr=states[state_name]

    df_empl.at[index, 'State'] = state_abbr

    

df_empl.head()
#Join datastets based on State column values

df_joined = pd.merge(df_empl,df_society,  how='left', left_on=['State'], right_on = ['State based on self-reported zipcode'])





#Append 'good thing' and 'bad thing' columns as percentages of overall answers, rather than raw #s

#get values of total survey respondents per state and store that in a dataframe

st_totals=pd.Series(all_data[state_index].value_counts(), name="Total Answers") 

st_totals.to_frame() 



#merge answer totals with final dataset

df_joined = pd.merge(df_joined,st_totals,  how='left', left_on=['State'], right_index=True)

#create percentage columns

df_joined['Good thing %'] = df_joined['Good thing'] / df_joined['Total Answers']

df_joined['Bad thing %'] = df_joined['Bad thing'] / df_joined['Total Answers']



#Remove states with less than 20 respondents

df=df_joined.query('`Total Answers`>20')



df.head()
import matplotlib.pyplot as plt

x=df['Net Tech Employment as % of Total Workforce']

y=df['Good thing %']



#Plot scatterplot

p1 = plt.scatter(x,y)

plt.ylabel("Internet is mostly a 'Good Thing' for society (% responses)")

plt.xlabel("Net Tech Employment as % of Total Workforce")



#Add best fit line

(m, b) = np.polyfit(x, y, 1) #m = slope, b=intercept

plt.plot(x, m*x + b) #plot best fit line
import matplotlib.pyplot as plt

x=df['Net Tech Employment as % of Total Workforce']

y=df['Bad thing %']



#Plot scatterplot

p1 = plt.scatter(x,y)

plt.ylabel("Internet is mostly a 'Bad Thing' for society (% responses)")

plt.xlabel("Net Tech Employment as % of Total Workforce")



#Add best fit line

(m, b) = np.polyfit(x, y, 1) #m = slope, b=intercept

plt.plot(x, m*x + b) #plot best fit line
#Join datastets based on State column values

df_joined = pd.merge(df_empl,df_self,  how='left', left_on=['State'], right_on = ['State based on self-reported zipcode'])



#convert all numbers to ints

#df.loc[:,'Good thing'] = df.loc[:,'Good thing'].astype(int)

#df.loc[:,'Bad thing'] = df.loc[:,'Bad thing'].astype(int)



#Append 'good thing' and 'bad thing' columns as percentages of overall answers, rather than raw #s

#get values of total survey respondents per state and store that in a dataframe

st_totals=pd.Series(all_data[state_index].value_counts(), name="Total Answers") 

st_totals.to_frame() 



#merge answer totals with final dataset

df_joined = pd.merge(df_joined,st_totals,  how='left', left_on=['State'], right_index=True)

#create percentage columns

df_joined['Good thing %'] = df_joined['Good thing'] / df_joined['Total Answers']

df_joined['Bad thing %'] = df_joined['Bad thing'] / df_joined['Total Answers']



#Remove states with less than 20 respondents

df=df_joined.query('`Total Answers`>20')



import matplotlib.pyplot as plt

x=df['Net Tech Employment as % of Total Workforce']

y=df['Good thing %']



#Plot scatterplot

p1 = plt.scatter(x,y)

plt.ylabel("Internet is a mostly 'Good Thing' for me (% responses)")

plt.xlabel("Net Tech Employment as % of Total Workforce")



#Add best fit line

(m, b) = np.polyfit(x, y, 1) #m = slope, b=intercept

plt.plot(x, m*x + b) #plot best fit line
import matplotlib.pyplot as plt

x=df['Net Tech Employment as % of Total Workforce']

y=df['Bad thing %']



#Plot scatterplot

p1 = plt.scatter(x,y)

plt.ylabel("Internet is a mostly 'Bad Thing' for me (% responses)")

plt.xlabel("Net Tech Employment as % of Total Workforce")



#Add best fit line

(m, b) = np.polyfit(x, y, 1) #m = slope, b=intercept

plt.plot(x, m*x + b) #plot best fit line