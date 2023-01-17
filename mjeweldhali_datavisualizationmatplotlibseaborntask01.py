import pandas as pd # Load data

import numpy as np # Scientific Computing

import seaborn as sns # Data Visualization

import matplotlib.pyplot as plt # Data Visualization

import warnings # Ignore Warnings

warnings.filterwarnings("ignore")

sns.set() # Set Graphs Background
data = pd.read_csv('../input/datafile/train (3).csv')

data.head()
data.info()
data['Country_Region'].unique()
data['Country_Region'].nunique()
plt.figure(figsize=(40,10))  # For Figure Resize

sns.barplot(x='Country_Region',y='ConfirmedCases', data=data)

plt.xlabel('Country_Region',fontsize = 35)

plt.ylabel('ConfirmedCases',fontsize = 35)

plt.xticks(rotation=90)  #For X label Value_Name rotation

plt.show() # Show The Plotfontsize = 25
plt.figure(figsize=(20,10))

sns.boxplot(data=data)

plt.show()
plt.figure(figsize=(20,8))

plt.scatter(data['Date'],data['ConfirmedCases'])

plt.title('22 January - 11 April Total ConfirmedCases By Date With Country', fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('ConfirmedCases',fontsize=25)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,8))

plt.scatter(data['Date'],data['Fatalities'])

plt.title('22 January - 11 April Total Deaths By Date With Country', fontsize=25)

plt.xlabel('Date', fontsize=25)

plt.ylabel('Deaths',fontsize=25)

plt.xticks(rotation=90)

plt.show()
data['Country_Region'].value_counts()
print(data['Date'].min())

print(data['Date'].max())
data_15 = data.groupby('Country_Region', as_index=False)['ConfirmedCases','Fatalities'].sum()

data_15 = data_15.nlargest(15,'ConfirmedCases')

data_15
data_15['Active/Recover'] = data_15['ConfirmedCases'] - data_15['Fatalities']

data_15
plt.figure(figsize=(15,8))    # For Figure Resize

sns.barplot(x='Country_Region',y='ConfirmedCases', data=data_15) # For Bar Plot

plt.title("22 January - 11 April Total ConfirmedCases By Country",fontsize = 25)   # For Title For the Graph 

plt.xlabel('Country_Region',fontsize = 25)   # For X-axis Name

plt.ylabel('ConfirmedCases',fontsize = 25)   # For Y-axis Name

plt.xticks(rotation=70)   # For X label Value_Name rotation

plt.show()    # Show The Plot
plt.figure(figsize=(15,8))  # For Figure Resize

sns.barplot(x='Country_Region',y='Fatalities', data=data_15)  # Show The Plot

plt.title("22 January - 11 April Total Deaths By Country",fontsize = 25)  # For Title For the Graph 

plt.xlabel('Country_Region',fontsize = 25)  # For X-axis Name

plt.ylabel('Fatalities',fontsize = 25)   # For Y-axis Name

plt.xticks(rotation=70)   # For X label Value_Name rotation

plt.show()   # Show The Plot
plt.figure(figsize=(15,8))  # For Figure Resize

sns.barplot(x='Country_Region',y='Active/Recover', data=data_15)  # Show The Plot

plt.title("22 January - 11 April Total Active/Recover By Country",fontsize = 25)  # For Title For the Graph 

plt.xlabel('Country_Region',fontsize = 25)  # For X-axis Name

plt.ylabel('Active/Recover',fontsize = 25)   # For Y-axis Name

plt.xticks(rotation=70)   # For X label Value_Name rotation

plt.show()   # Show The Plot
plt.figure(figsize=(15,8))

sns.barplot(x='Country_Region',y='ConfirmedCases', data=data_15, color='green',label='ConfirmedCases')

sns.barplot(x='Country_Region',y='Fatalities', data=data_15,color='red', label='Deaths')

plt.title("22 January - 11 April Total Country_Region Vs ConfirmedCases & Deaths")

plt.xlabel('Country_Region')

plt.ylabel('ConfirmedCases & Fatalities')

plt.xticks(rotation=70)

plt.legend()

plt.show()
bubbol = np.array(data_15['ConfirmedCases']/1500) # For Bubbol Size

plt.figure(figsize=(15,8))

plt.scatter(data_15['Country_Region'],data_15['ConfirmedCases'],c='green',s=bubbol, alpha=0.6)

plt.title("22 January - 11 April Total ConfirmedCases By Country",fontsize = 20)

plt.xlabel('Country_Region',fontsize = 20)

plt.ylabel('ConfirmedCases',fontsize = 20)

plt.xticks(rotation=70)

plt.show()
bubbol = np.array(data_15['Active/Recover']/1500) # For Bubbol Size

plt.figure(figsize=(15,8))

plt.scatter(data_15['Country_Region'],data_15['Active/Recover'],c='blue',s=bubbol, alpha=0.6)

plt.title("22 January - 11 April Total Active/Recover By Country",fontsize = 20)

plt.xlabel('Country_Region',fontsize = 20)

plt.ylabel('ConfirmedCases',fontsize = 20)

plt.xticks(rotation=70)

plt.show()
bubbol = np.array(data_15['Fatalities']/50) # For Bubbol Size

plt.figure(figsize=(15,8))

plt.scatter(data_15['Country_Region'],data_15['Fatalities'],c='red',s=bubbol, alpha=0.6)

plt.title("22 January - 11 April Total Deaths By Country",fontsize = 25)

plt.xlabel('Country',fontsize = 25)

plt.ylabel('Deaths',fontsize = 25)

plt.xticks(rotation=70)

plt.show()
plt.axis('equal')

plt.pie(data_15['ConfirmedCases'],labels=data_15['Country_Region'], radius=2, autopct='%.0f%%',

        shadow=True)

plt.show()
plt.axis('equal')

plt.pie(data_15['Fatalities'],labels=data_15['Country_Region'], radius=2, autopct='%.0f%%',

        shadow=True)

plt.show()
data['Date'] = pd.to_datetime(data['Date'])
data_81 = data.groupby('Date', as_index=False)['ConfirmedCases','Fatalities'].sum()

data_81 = data_81.nlargest(81,'ConfirmedCases')

data_81
data_81['Active/Recover'] = data_81['ConfirmedCases'] - data_81['Fatalities']

data_81
bubbol = np.array(data_81['ConfirmedCases']/1500) # For Bubbol Size

plt.figure(figsize=(15,8))

plt.scatter(data_81['Date'],data_81['ConfirmedCases'],c='blue',s=bubbol, alpha=0.6)

plt.title("22 January - 11 April Total ConfirmedCases By Date",fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('ConfirmedCases',fontsize=25)

plt.show()
bubbol = np.array(data_81['Active/Recover']/1500) # For Bubbol Size

plt.figure(figsize=(15,8))

plt.scatter(data_81['Date'],data_81['Active/Recover'],c='green',s=bubbol, alpha=0.6)

plt.title("22 January - 11 April Total Active/Recover By Date",fontsize = 20)

plt.xlabel('Date',fontsize = 20)

plt.ylabel('Active/Recover',fontsize = 20)

plt.xticks(rotation=70)

plt.show()
bubbol = np.array(data_81['Fatalities']/50) # For Bubbol Size

plt.figure(figsize=(15,8))

plt.scatter(data_81['Date'],data_81['Fatalities'],c='red',s=bubbol, alpha=0.6)

plt.title("22 January - 11 April Total Deaths By Date",fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('Deaths',fontsize=25)

plt.show()
plt.figure(figsize=(15,8))

plt.scatter(data_81['Date'],data_81['ConfirmedCases'],c='blue', alpha=0.6, label='ConfirmedCases')

plt.scatter(data_81['Date'],data_81['Active/Recover'],c='green',alpha=0.6, label='Active/Recover')

plt.scatter(data_81['Date'],data_81['Fatalities'],c='red',alpha=0.6, label='Fatalities')

plt.title("Total ConfirmedCases & Active/Recover & Deaths By Date",fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('ConfirmedCases Active/Recover & Deaths',fontsize=25)

plt.legend(loc=10)

plt.show()
fig, ax = plt.subplots(figsize = (20,10))    

fig = sns.barplot(x ="Date", y ="ConfirmedCases", data = data_81)



x_dates = data_81['Date'].dt.strftime('%Y-%m-%d').sort_values()

ax.set_xticklabels(labels=x_dates, rotation=80)

plt.title("22 January - 11 April Total ConfirmedCases By Date",fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('ConfirmedCases',fontsize=25)

plt.show()
fig, ax = plt.subplots(figsize = (20,10))    

fig = sns.barplot(x ="Date", y ="Fatalities", data = data_81)



x_dates = data_81['Date'].dt.strftime('%Y-%m-%d').sort_values()

ax.set_xticklabels(labels=x_dates, rotation=80)

plt.title("22 January - 11 April Total Date VS Deaths",fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('Deaths',fontsize=25)

plt.show()
plt.figure(figsize=(15,8))

sns.lineplot(x='Date',y='ConfirmedCases', data=data_81)

plt.title("22 January - 11 April Total ConfirmedCases By Date for world",fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('ConfirmedCases',fontsize=25)

plt.show()
plt.figure(figsize=(15,8))

sns.lineplot(x='Date',y='Fatalities', data=data_81)

plt.title("22 January - 11 April Total Deaths By Date for world",fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('Deaths',fontsize=25)

plt.show()
plt.figure(figsize=(15,8))

sns.lineplot(x='Date',y='ConfirmedCases', data=data_81, label='ConfirmedCases')

sns.lineplot(x='Date',y='Active/Recover', data=data_81, label='Active/Recover')

sns.lineplot(x='Date',y='Fatalities', data=data_81, label='Fatalities')

plt.title("Total ConfirmedCases & Active/Recover & Deaths By Date for world",fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('ConfirmedCases & Active/Recover & Deaths',fontsize=25)

plt.legend()

plt.show()
data_all = data.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()

data_all
data_all['Active/Recover'] = data_all['ConfirmedCases'] - data_all['Fatalities']

data_all
data_usa = data_all.query("Country_Region=='US'")

data_usa
bubbol = np.array(data_usa['ConfirmedCases']/500) # For Bubbol Size

plt.figure(figsize=(15,8))

plt.scatter(data_usa['Date'],data_usa['ConfirmedCases'],c='blue',s=bubbol, alpha=0.6)

plt.title("22 January - 11 April Total ConfirmedCases By Date For United State",fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('ConfirmedCases',fontsize=25)

plt.show()
bubbol = np.array(data_usa['Fatalities']/40) # For Bubbol Size

plt.figure(figsize=(15,8))

plt.scatter(data_usa['Date'],data_usa['Fatalities'],c='blue',s=bubbol, alpha=0.6)

plt.title("22 January - 11 April Total Deaths By Date For United State",fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('Deaths',fontsize=25)

plt.show()
plt.figure(figsize=(15,8))

plt.plot(data_usa['Date'],data_usa['ConfirmedCases'],c='blue', alpha=0.6,label='ConfirmedCases')

plt.plot(data_usa['Date'],data_usa['Active/Recover'],c='green', alpha=0.6, label='Active/Recover')

plt.plot(data_usa['Date'],data_usa['Fatalities'],c='red', alpha=0.6, label='Fatalities')

plt.title("Total ConfirmedCases & Active/Recover & Deaths By Date For United State",fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('ConfirmedCases & Active/Recover & Deaths',fontsize=25)

plt.legend(loc=10)

plt.show()
fig, ax = plt.subplots(figsize = (20,10))    

fig = sns.barplot(x ="Date", y ="ConfirmedCases", data = data_usa)



x_dates = data_usa['Date'].dt.strftime('%Y-%m-%d').sort_values()

ax.set_xticklabels(labels=x_dates, rotation=80)

plt.title("22 January - 11 April Total ConfirmedCases By Date For United State",fontsize=25)

plt.xlabel('Date',fontsize=25)

plt.ylabel('ConfirmedCases',fontsize=25)

plt.show()
fig, ax = plt.subplots(figsize = (20,10))    

fig = sns.barplot(x ="Date", y ="Fatalities", data = data_usa)



x_dates = data_usa['Date'].dt.strftime('%Y-%m-%d').sort_values()

ax.set_xticklabels(labels=x_dates, rotation=80)

plt.title("22 January - 11 April Total Deaths By Date For United State",fontsize = 25)

plt.xlabel('Date',fontsize = 25)

plt.ylabel('Deaths',fontsize = 25)

plt.show()