# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read the Data

data_frame = pd.read_csv("../input/database.csv")



#drop NaN rows

data_frame['Start Date'][89]= None # I need drop this row because invalid date type

data_frame = data_frame.dropna() # drop NaN rows

data_frame.info()
# Parsing Srt type to DateTime type

data_frame['Declaretion Date'] = data_frame['Start Date'].map(lambda x: datetime.datetime.strptime(x,'%m/%d/%Y'))

data_frame['Start Date'] = data_frame['Start Date'].map(lambda x: datetime.datetime.strptime(x,'%m/%d/%Y'))

data_frame['End Date'] = data_frame['End Date'].map(lambda x: datetime.datetime.strptime(x,'%m/%d/%Y'))

data_frame['Close Date'] = data_frame['Close Date'].map(lambda x: datetime.datetime.strptime(x,'%m/%d/%Y'))





#Parse full datetime to Year Only

df = data_frame.copy()

df['Start Date'] = df['Start Date'].map(lambda x: x.year)

df['End Date'] = df['End Date'].map(lambda x: x.year)

df['Close Date'] = df['Close Date'].map(lambda x: x.year)







start_date =  df.groupby(['Start Date']).count().reset_index().iloc[:,range(2)]

start_date.rename(columns={'Declaration Number': 'Total','Start Date':'Date'}, inplace=True)



end_date =  df.groupby(['End Date']).count().reset_index().iloc[:,range(2)]

end_date.rename(columns={'Declaration Number': 'Total','End Date':'Date'}, inplace=True)



close_date =  df.groupby(['Close Date']).count().reset_index().iloc[:,range(2)]

close_date.rename(columns={'Declaration Number': 'Total','Close Date':'Date'}, inplace=True)
g = data_frame.groupby(['Start Date','Declaration Type'], as_index=False).count().iloc[:,range(3)]



g['Month'] = g['Start Date'].apply(lambda x: x.strftime('%m %B'))

g['Year'] = g['Start Date'].apply(lambda x: x.strftime('%Y'))

p=pd.pivot_table(g, values='Declaration Number', index=['Month'] , columns=['Year'], aggfunc=np.sum)





fig,ax = plt.subplots(figsize=(20,15))

ax = sns.heatmap(p,linewidths=.2,cmap='inferno_r')

ax.set_title(label="Disasters By Year")
#Plot data

fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(25,17))





#legends

start_leg = mpatches.Patch(color='red', label='Start Data')

end_leg = mpatches.Patch(color='blue', label='End Data')

close_leg = mpatches.Patch(color='green', label='Close Data')

plt.legend(handles=[start_leg,end_leg,close_leg])



ax1.set_title("Federal Disasters Start Date")

ax2.set_title("Federal Disasters End Date")

ax3.set_title("Federal Disasters Close Date")

ax4.set_title("Federal Disasters All")





plt.xticks(fontsize=12,rotation=45,ha='left')



#Axis



ax1.plot(start_date['Date'], start_date['Total'],'r')

ax2.plot(end_date['Date'], end_date['Total'],'b')

ax3.plot(close_date['Date'], close_date['Total'],'g')



ax4.plot(start_date['Date'], start_date['Total'],'r')

ax4.plot(end_date['Date'], end_date['Total'],'b')

ax4.plot(close_date['Date'], close_date['Total'],'g')



#Calculating the Average of Start to End | End to Close and Start To Close Date

data_frame['Start-End Time'] = data_frame['End Date'] - data_frame['Start Date']

data_frame['End-Close Time'] = data_frame['Close Date'] - data_frame['End Date']

data_frame['Start-Close Time'] = data_frame['Close Date'] - data_frame['Start Date']



print("Start to End Average ", data_frame['Start-End Time'].mean())

print("End to Close Average ",data_frame['End-Close Time'].mean())

print("Start to Close Average ",data_frame['Start-Close Time'].mean())
#Data By State

state = df.groupby(['State']).count().reset_index().iloc[:,range(2)]

state.rename(columns={'Declaration Number': 'Total'}, inplace=True)



#Data By Disaster Type

disaster = df.groupby(['Disaster Type']).count().reset_index().iloc[:,range(2)]

disaster.rename(columns={'Declaration Number': 'Total'}, inplace=True)



#Data By Declaration Type



declaration_type = df.groupby(['Declaration Type']).count().reset_index().iloc[:,range(2)]

declaration_type.rename(columns={'Declaration Number': 'Total'}, inplace=True)
#Ploting Figure



fig, (axis1,axis2) = plt.subplots(2,1,figsize=(25,10))



axis1.set_title('Disasters by Type')

axis2.set_title('Disaster by State')





sns.barplot(x='Disaster Type',y='Total',ax=axis1,data=disaster.sort_values(['Total'],ascending=[0]))

sns.barplot(x='State',y='Total',ax=axis2,data=state.sort_values(['Total'],ascending=[0]))
fig,ax= plt.subplots(figsize=(10,6))

ax.set_title('Declaration by Type')

sns.barplot(x='Declaration Type',y='Total',ax=ax,data=declaration_type.sort_values(['Total'],ascending=[0]))