import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
ad = pd.read_csv("/kaggle/input/uncover/UNCOVER/covid_tracking_project/covid-statistics-by-us-states-daily-updates.csv")  
ad.head()
ad.describe()
ad.dtypes
ad['tempdate'] = pd.to_datetime(ad['date']) # Creating a new date column with dtype datetime
ad.dtypes
ad.head()
ad['tempdate'].max() # Last date till which data is present
ad['tempdate'].min() # First date from which data is present
ad.columns
ad.head()
ad.fillna(0, inplace=True) # Replacing all the NaN with 0
#ad[(ad.hospitalizedcurrently > 0) & (ad.state == 'NY')]
#ad[(ad.date > '2020-03-15') & (ad.date < '2020-03-25') & (ad.state == 'NY')]
ad[ad.state == 'NY'].head() # Checking sample data from NY state
gr = ad.groupby('tempdate')
gr.first()
gr.get_group('2020-04-28')# Printing the sample grouped data of a date i.e 2020-04-28 
us_datewise = gr.sum()

us_datewise.tail()
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (20,10))
chart = sns.barplot(x= us_datewise.index, y="deathincrease" , 
            data = us_datewise)
chart.set_xticklabels(us_datewise.index.date,rotation=90)
chart.set_xlabel("Date",fontsize = 50)
chart.set_ylabel("No of Deaths",fontsize = 50)
chart.set_title("Daily Deaths in USA", fontsize = 50)
gr1 = ad.groupby('state')
us_statewise = gr1.mean()
us_statewise.head()
fig = plt.figure(figsize = (20,10))
chart1 = sns.barplot(x= us_statewise.index, y="positiveincrease" , data = us_statewise)
chart1.set_xticklabels(us_statewise.index,rotation=45,fontsize = 15)
chart1.set_xlabel("State",fontsize = 50)
chart1.set_ylabel("Average Daily New cases",fontsize = 30)
chart1.set_title("Average Daily New cases in different states of USA", fontsize = 40)
us_datewise.columns
us_datewise['%recovery_outof_outcome'] = us_datewise.recovered/(us_datewise.recovered + us_datewise.death)
us_datewise['%death_outof_outcome'] = us_datewise.death/(us_datewise.recovered + us_datewise.death)
us_datewise['%recovery_outof_outcome'].fillna(1, inplace=True) # Replacing all the NaN with 1 in %recovery_outof_outcome
us_datewise['%death_outof_outcome'].fillna(0, inplace=True) # Replacing all the NaN with 0 in %death_outof_outcome
us_datewise.head()
us_datewise['%recovery_outof_outcome'].plot()
us_datewise['%death_outof_outcome'].plot()
