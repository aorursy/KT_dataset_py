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
import datetime as dt

import seaborn as sns

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

import plotly.express as px



import sys



if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")
df = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')

df.head()
# drop unwanted columns. 

df = df.drop(['Unnamed: 0', 'Unnamed: 0.1', ' Rocket'], axis=1)





# Cleaning the Datum column by removing Timezone = UTC' 

df['Datum'] = df.Datum.apply(lambda x: pd.Series(x[:-3]))

# convert the Datum column into timedate stamp

df["Datum"] = df["Datum"].apply(pd.to_datetime)



#Create additional column for Country of luanch. 

df['Country'] = df.Location.apply(lambda x: (pd.Series(str(x).split(' ')[-1])))



#create seperate year column from Datum

df['year'] = pd.DatetimeIndex(df['Datum']).year



#create seperate month column from Datum

df['month'] = pd.DatetimeIndex(df['Datum']).month



df.head()

## Would like to group all the failures into one group. 

df['Success_failure'] = df['Status Mission'].apply(lambda x: 'Failure' if 'Failure' in x else x)

df.head()
#Which year of the launch, was most success?



year_list = list(df['year'].unique())



num_launch = []

num_success = []

prob_success = []



# get number of lunchs and success for each year

for n in year_list:

    num_launch.append(((df[df['year']==n]).shape)[0])

    num_success.append((df[(df['year']==n) & (df['Success_failure'] == 'Success')]).shape[0])

    

# get probability of success for each year    

for m in range(len(num_launch)):

    prob_success.append(num_success[m]/num_launch[m])



    

#convert the lists into data dict.    

data = {'year': year_list, 'launchs': num_launch, 'success': num_success, 'probability': prob_success}



#create dataframe

df_year = pd.DataFrame(data=data, columns= ['year', 'launchs', 'success','probability'])



#find top 5 successful launch yearwise. 

df_year.nlargest(5,columns=['probability'], keep='first')
colors = np.random.rand(64)

plt.figure(num=None, figsize=(15,10))



plt.scatter(x= df_year['year'], y= df_year['probability']*100, s= df_year['launchs']*10, c = colors)





plt.xlabel("Year of Launch")

plt.ylabel("%Probability of Success")

plt.title("%Proability of Success vs Year of Launch", loc="center")
fig = px.treemap(df_year.sort_values(by = 'launchs', ascending= False).reset_index(drop = True),

                         path = ['year'], values= 'launchs', height = 700,

                         title = 'Number of launchs year wise',

                         color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label + text+ value'



fig.show()



#Reference https://www.youtube.com/watch?v=ztXcGdLYwe8
plt.figure(num=None, figsize=(15,10))



sns.countplot(x='month', data= df, saturation=0.8, dodge=True)
#Which company was most successful?



company_list = list(df['Company Name'].unique())



num_launch = []

num_success = []

prob_success = []



# get number of lunchs and success for each company

for n in company_list:

    num_launch.append(((df[df['Company Name']== n]).shape)[0])

    num_success.append((df[(df['Company Name']==n) & (df['Success_failure'] == 'Success')]).shape[0])

    

# get probability of success for each company    

for m in range(len(num_launch)):

    prob_success.append(num_success[m]/num_launch[m])



    

#convert the lists into data dict.    

data_co = {'Company': company_list, 'launchs': num_launch, 'success': num_success, 'probability': prob_success}



#create dataframe

df_comp = pd.DataFrame(data=data_co, columns= ['Company', 'launchs', 'success','probability'])



#find top 10 successful companies. 

df_comp.nlargest(10,columns=['launchs'], keep='first')
top10_comp = df_comp.nlargest(10,columns=['launchs'], keep='first')



sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(20, 8))



# Plot the total counts vs year

sns.set_color_codes("pastel")

sns.barplot(x="Company", y="launchs", data=top10_comp,

            label="Total", color="m")



# Plot the crashes where alcohol was involved

sns.set_color_codes("pastel")

sns.barplot(x="Company", y="success", data=top10_comp,

            label="success", color="g")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="upper right", frameon=True)

ax.set(ylabel="Counts of launch", xlabel="Companies")

#sns.despine(left=True, bottom=True)
df_comp

fig = px.treemap(df_comp.sort_values(by = 'launchs', ascending= False).reset_index(drop = True),

                         path = ['Company'], values= 'launchs', height = 700,

                         title = 'Number of launchs Company wise',

                         color_discrete_sequence = px.colors.qualitative.D3)

fig.data[0].textinfo = 'label + text+ value'



fig.show()
df_comp[df_comp['Company'] =='SpaceX']
top10_launchs = df_year.nlargest(10,columns=['launchs'], keep='first')



sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(15, 6))



# Plot the total counts vs year

sns.set_color_codes("pastel")

sns.barplot(x="year", y="launchs", data=top10_launchs,

            label="Total", color="m")



# Plot the crashes where alcohol was involved

sns.set_color_codes("muted")

sns.barplot(x="year", y="success", data=top10_launchs,

            label="success", color="g")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="upper left", frameon=True)

ax.set(ylabel="Counts of launch", xlabel="Year")

#sns.despine(left=True, bottom=True)

fig = px.treemap(df_year.sort_values(by = 'launchs', ascending= False).reset_index(drop = True),

                         path = ['year'], values= 'launchs', height = 700,

                         title = 'Number of launchs year wise',

                         color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label + text+ value'



fig.show()
plt.figure(figsize=(15,10))

sns.lineplot(x="year", y="probability",markers=True, dashes=False, data=df_year)
#Which year of the launch, was least successful?

df_year.nsmallest(5,columns=['probability'], keep='first')
#Which country was most successful?



country_list = list(df['Country'].unique())



num_launch = []

num_success = []

prob_success = []



# get number of lunchs and success for each company

for n in country_list:

    num_launch.append(((df[df['Country']== n]).shape)[0])

    num_success.append((df[(df['Country']==n) & (df['Success_failure'] == 'Success')]).shape[0])

    

# get probability of success for each company    

for m in range(len(num_launch)):

    prob_success.append(num_success[m]/num_launch[m])



    

#convert the lists into data dict.    

data_country = {'Country': country_list, 'launchs': num_launch, 'success': num_success, 'probability': prob_success}



#create dataframe

df_country = pd.DataFrame(data=data_country, columns= ['Country', 'launchs', 'success','probability'])



#find top 5 successful companies. 

df_country.nlargest(5,columns=['launchs'], keep='first')
fig = px.treemap(df_country.sort_values(by = 'launchs', ascending= False).reset_index(drop = True),

                         path = ['Country'], values= 'launchs', height = 700,

                         title = 'Number of launchs country wise',

                         color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label + text+ value'



fig.show()



#Reference https://www.youtube.com/watch?v=ztXcGdLYwe8