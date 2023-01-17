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
!wget kaggle datasets download -d parulpandey/coronavirus-cases-in-india

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import seaborn as sns
df_covid = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")

df_covid.head()
states = df_covid['State/UnionTerritory'].unique()
m = len(states)
total_cases_statewise = []
total_IndianNational = []
total_ForeignNational = []
deaths = []
cured = []

for st in states:
    total_cases_statewise.append(df_covid['Confirmed'][df_covid['State/UnionTerritory']==st].max())
    total_IndianNational.append(df_covid['ConfirmedIndianNational'][df_covid['State/UnionTerritory']==st].max())
    total_ForeignNational.append(df_covid['ConfirmedForeignNational'][df_covid['State/UnionTerritory']==st].max())
    deaths.append(df_covid['Deaths'][df_covid['State/UnionTerritory']==st].max())
    cured.append(df_covid['Cured'][df_covid['State/UnionTerritory']==st].max())

states = np.array(states).reshape((m,1))
total_cases_statewise = np.array(total_cases_statewise).reshape((m,1))
total_IndianNational = np.array(total_IndianNational).reshape((m,1))
total_ForeignNational = np.array(total_ForeignNational).reshape((m,1))
deaths = np.array(deaths).reshape((m,1))
cured = np.array(cured).reshape((m,1))


states_data = np.hstack([states,total_cases_statewise,total_IndianNational,total_ForeignNational,deaths,cured])
states_data_df = pd.DataFrame(states_data,columns=['states','Confirmed','ConfirmedIndianNational','ConfirmedForeignNational','Deaths','Cured'])

cases_per_state = pd.Series(states_data_df['Confirmed'])
plt.style.use('seaborn')
my_colors = ['hotpink','aquamarine','lightgreen','gold','salmon']
plt.figure(figsize=(20,20))
ax = cases_per_state.plot(kind='barh',color=my_colors,width=1)
ax.set_title('Covid 19 confirmed cases on {}'.format(datetime.date.today()),size=20)
ax.set_xlabel('No. of Confirmed Cases',size=20)
ax.set_ylabel('States & UTs',size=20)
ax.set_xlim(-20,6000)
ax.set_yticklabels(df_covid['State/UnionTerritory'].unique(),size=15)
#ax.grid(True)
my_colors = ['hotpink','aquamarine','lightgreen','gold','salmon']
plt.style.use('seaborn')

rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 10
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)         

#some of the values in the column are "-", replacing them with 0
df_covid['ConfirmedIndianNational'][df_covid['ConfirmedIndianNational'] == "-"] = 0
df_covid['ConfirmedIndianNational'].astype('float32')
df_covid['ConfirmedIndianNational'] = pd.to_numeric(df_covid['ConfirmedIndianNational'])

df_covid['ConfirmedForeignNational'][df_covid['ConfirmedForeignNational'] == "-"] = 0
df_covid['ConfirmedForeignNational'].astype('float32')
df_covid['ConfirmedForeignNational'] = pd.to_numeric(df_covid['ConfirmedForeignNational'])




labels = df_covid['State/UnionTerritory'].unique()
labels.sort()


national = states_data_df['ConfirmedIndianNational']
national[national=='-'] = 0
national = national.values
national = national.astype("float32")

Foreign = states_data_df['ConfirmedForeignNational']
Foreign[Foreign=='-'] = 0
Foreign = Foreign.values
Foreign = Foreign.astype('float32')
width = 0.6  # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots(figsize=(30,20))


ax.bar(labels, national,width, label='National Cases',color='gold')
ax.bar(labels,Foreign, width, bottom=national,label='Foreign Cases',color='orangered')
plt.xticks(rotation=90,size=15)
ax.set_ylabel('Cases Counts',size=20)
ax.set_xlabel('States/UTs',size=20)
ax.set_title('IndianNatioanl Vs ForeignNational Cases ',size=20)
ax.legend()

plt.show()
labels = 'Active Cases', 'Cured', 'Deceased'
total = states_data_df['Confirmed'].sum()
deceased = states_data_df['Deaths'].sum()/total
cured = states_data_df['Cured'].sum()/total
active = (total - (deceased+cured))/total
sizes = [active,cured,deceased]
explode = (0.1,0.1,0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots(figsize=(10,10))


ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90,colors=['dodgerblue','springgreen','red'])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Situation in India as on {}".format(datetime.date.today()),size=(20))

plt.show()
dates = df_covid['Date'].unique()
l = len(dates)
confirmed_total = []
deaths_total = []
cases_per_day=[]
for i,dt in enumerate(dates):
  
    confirmed_total.append(df_covid['Confirmed'][df_covid['Date']==dt].sum())
    deaths_total.append(df_covid['Deaths'][df_covid['Date']==dt].sum())
    cases_per_day.append((df_covid['Confirmed'][df_covid['Date']==dt].sum())-(df_covid['Confirmed'][df_covid['Date']==dates[i-1]].sum()))
    
    
dates = np.array(dates).reshape(l,1)
confirmed_total = np.array(confirmed_total).reshape(l,1)
deaths_total = np.array(deaths_total).reshape(l,1)
cases_per_day[0] = 1
cases_per_day = np.array(cases_per_day).reshape(l,1)


date_wise_df = pd.DataFrame(np.hstack([dates,confirmed_total,deaths_total,cases_per_day]),columns=['Dates','Confirmed','Deaths','cases_per_day'])

plt.style.use('dark_background')
x = np.arange(l)
plt.figure(figsize=(30,20))
plt.plot(x,date_wise_df['cases_per_day'],'bo-',color='gold',linewidth=2)
plt.plot(x,date_wise_df['cases_per_day'],'r-',color='crimson',linewidth=2)
plt.grid(False)
plt.xlabel('Dates',size=20)
plt.ylabel('Cases Counts',size=20)
plt.xticks(x,date_wise_df['Dates'],rotation=90,fontsize=15)
plt.title("New Cases Daily",size=30)


plt.show()
plt.style.use('Solarize_Light2')
x = np.arange(l)
plt.figure(figsize=(25,15))
plt.plot(x,date_wise_df['Deaths'],'ro-',color='orangered',linewidth=2)
plt.plot(x,date_wise_df['Deaths'],'o',color='black',linewidth=2)
plt.grid(False)
plt.xlabel('Dates',size=20)
plt.ylabel('Total Death Counts',size=20)
plt.xticks(x,date_wise_df['Dates'],rotation=90,fontsize=15)
plt.title("Increment in Total Number of Deaths",size=30)
plt.show()

#size of array will be len(states)*len(dates)
m = len(states)
n = len(dates)
death_array = np.zeros((m,n))
confirmed_array = np.zeros((m,n))


for i,st in enumerate(states):
    for j,dt in enumerate(dates):
        
        death_array[i][j] = df_covid['Deaths'][df_covid['State/UnionTerritory']==st[0]][df_covid['Date']==dt[0]].sum()
        confirmed_array[i][j] = df_covid['Confirmed'][df_covid['State/UnionTerritory']==st[0]][df_covid['Date']==dt[0]].sum()
        
        
        


death_array = death_array.astype("int32")
confirmed_array = confirmed_array.astype('int32')

new_death_array = np.zeros(death_array.shape)
new_confirmed_array = np.zeros(confirmed_array.shape)

for i in range(death_array.shape[0]):
    for j in range(death_array.shape[1]):
        
        if(death_array[i][j] - death_array[i][j-1] > 0):
            new_death_array[i][j] = death_array[i][j] - death_array[i][j-1]
        else:
            new_death_array[i][j] = death_array[i][j]
        
        
for i in range(confirmed_array.shape[0]):
    for j in range(confirmed_array.shape[1]):
        
        if(confirmed_array[i][j] - confirmed_array[i][j-1] > 0):
            new_confirmed_array[i][j] = confirmed_array[i][j] - confirmed_array[i][j-1]
        else:
            new_confirmed_array[i][j] = confirmed_array[i][j]
            
            
new_death_array = new_death_array.astype('int32')
new_confirmed_array = new_confirmed_array.astype('int32')
death_array = death_array.astype("int32")

new_array = np.zeros(death_array.shape)
for i in range(death_array.shape[0]):
    for j in range(death_array.shape[1]):
        
        if(death_array[i][j] - death_array[i][j-1] > 0):
            new_array[i][j] = death_array[i][j] - death_array[i][j-1]
        else:
            new_array[i][j] = death_array[i][j]
            
new_array = new_array.astype('int32')
columns = [dt[0] for dt in dates]
#columns = dates
#columns.insert(0,'States')
Confirmed_heatmap = pd.DataFrame(new_confirmed_array,columns=columns,index=states)
Death_heatmap = pd.DataFrame(new_death_array,columns=columns,index=states)

plt.style.use('seaborn')
plt.figure(figsize=(50,20))
plt.yticks(size=20)
plt.xticks(size=20)


sns.heatmap(Confirmed_heatmap,linewidths=0.2,cmap="YlGnBu",annot=True,fmt='d')
plt.xlabel("DATES",size=40)
plt.ylabel('STATES',size=40)
plt.title('Cases Count HeatMap',size=50)
plt.show()
plt.style.use('seaborn')
plt.figure(figsize=(50,25))
plt.yticks(size=20)
plt.xticks(size=20)


sns.heatmap(Death_heatmap,linewidths=0.2,cmap="YlOrBr",annot=True,fmt='d')
plt.xlabel("DATES",size=40)
plt.ylabel('STATES',size=40)
plt.title('Death Count HeatMap',size=50)
plt.show()
df_age = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")
df_age.head()
df_age['Percentage_vals'] = df_age['Percentage']
for i in range(df_age.shape[0]): 
    df_age['Percentage_vals'][i] = float(df_age['Percentage'][i].strip('%'))
    
l = df_age.shape[0]
plt.style.use('seaborn-pastel')
plt.figure(figsize=(15,15))

plt.pie(df_age['Percentage_vals'],labels=df_age['AgeGroup'],startangle=90,explode=[0.1,0.05]*(l//2),autopct="%1.1f%%",shadow=True)
plt.title("Covid19 Impact on Different Age Groups",size=20)
plt.legend(df_age['AgeGroup'],loc='upper right')
plt.show()
print("Chart shows people of age groups 20-29,30-39,40-49 are more prone to disease. This could be possible relying on the fact \n that most of the people of these age groups belongs to working class of the population.")
df_patients = pd.read_csv('../input/covid19-corona-virus-india-dataset/patients_data.csv')
df_patients.info()
df_patients.head(20)
