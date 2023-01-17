import pandas as pd

import numpy as np

import datetime as dt

import arrow as ar

import seaborn as sns

import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode,iplot,iplot_mpl,download_plotlyjs,enable_mpl_offline

import plotly.graph_objs as go



%matplotlib inline



init_notebook_mode(connected=True)
sns.set_style('whitegrid')
noShows = pd.read_csv('../input/No-show-Issue-Comma-300k.csv')
noShows.head(2)
# First let's clean the data a little bit, the appointment date and registration date are formatted as strings.

# Now we will have appointment dates and Registration dates.

# Also our column names should be fixed. Alcoolism? lol

noShows['Registration Date'] = noShows['AppointmentRegistration'].str.split('T').str[0]

noShows['Registration Date'] = noShows['Registration Date'].astype('datetime64[ns]')

noShows['Appointment Date'] = noShows['ApointmentData'].str.split('T').str[0]

noShows['Appointment Date'] = noShows['ApointmentData'].astype('datetime64[ns]')

noShows['Registration Time'] = noShows['AppointmentRegistration'].str.split('T').str[-1].str.replace('Z','')

noShows.drop(['ApointmentData','AppointmentRegistration'],axis=1,inplace=True)

noShows.rename(columns={'Alcoolism':'Alcoholism','HiperTension':'HyperTension','Handcap':'Handicap','Smokes':'Smoker'},inplace=True)
noShows.head(2)
noShows['Month Number'] = noShows['Appointment Date'].dt.month
def dayOfWeek(x):

    if x == 'Monday':

        return 1

    if x == 'Tuesday':

        return 2

    if x == 'Wednesday':

        return 3

    if x == 'Thursday':

        return 4

    if x == 'Friday':

        return 5

    if x == 'Saturday':

        return 6

    if x == 'Sunday':

        return 7

def ageGroup(x):

    if x < 18:

        return "Less than 18"

    elif x >= 18 and x <= 35:

        return "Young Adult"

    elif x > 35 and x <= 64:

        return "Older Adult"

    else:

        return "65+"
noShows['DayNumber'] = noShows['DayOfTheWeek'].apply(dayOfWeek)

noShows['AgeGroup'] = noShows['Age'].apply(ageGroup)
# An issue that I found was that we have a ton of patients with the age of 0. This is a fairly large data set,

# so we can exclude those patients in our analysis.

noShows = noShows[noShows['Age'] > 0]
# First let's check out the Day of the Week that has the largest No-Shows

figure = plt.figure(figsize=(18,8))

ax = figure.add_subplot(1,2,1)

ax.set_title("Weekly No Show Count")

sns.countplot(x="DayNumber",hue="Status",data=noShows,palette='winter',ax=ax)

ax.set_xlabel("Weekday (starts from Monday)")



ax2 = figure.add_subplot(1,2,2)

ax2.set_title("Monthly No Show Count")

sns.countplot(x="Month Number",hue="Status",data=noShows,palette='winter',ax=ax2)

ax2.set_xlabel("Month (starts from January)")



plt.tight_layout

plt.show()
# We need to transform the data a bit, so let's just 

pv = noShows.pivot_table(values=["Diabetes","Alcoholism","HyperTension","Handicap","Smoker","Tuberculosis"],columns="Status",index="Month Number",aggfunc=np.sum)
data = pv.unstack().unstack('Status').reset_index().rename(columns={"level_0":"Category"})
data["No Show Rate"] = data['No-Show'] / (data['No-Show'] + data['Show-Up'])
data.head()
# Now we are ready to see some plots!



figure = plt.figure(figsize=(10,6))

ax = figure.add_axes([0,0,1,1])

sns.boxplot(x="Category",y="Show-Up",data=data)
data[data['Category'] == 'Tuberculosis']
# Let's create a line graph showing the trends. We can alter this to make it a function later but for now,

# we are just going to filter and plot

Diabetes = data[data['Category'] == 'Diabetes']

Alcoholism = data[data['Category'] == 'Alcoholism']

Handicap = data[data['Category'] == 'Handicap']

Hyper = data[data['Category'] == 'HyperTension']

Smoker = data[data['Category'] == 'Smoker']

Tuber = data[data['Category'] == 'Tuberculosis']



trace = go.Scatter(

    x= Diabetes['Month Number'],

    y=Diabetes['No Show Rate']*100,

    mode= 'lines+markers',

    text=Diabetes['Category'],

    name="Diabetes")



trace2 = go.Scatter(

    x= Alcoholism['Month Number'],

    y=Alcoholism['No Show Rate']*100,

    mode= 'lines+markers',

    text=Alcoholism['Category'],

    name="Alcoholism")



trace3 = go.Scatter(

    x= Handicap['Month Number'],

    y=Handicap['No Show Rate']*100,

    mode= 'lines+markers',

    text=Handicap['Category'],

    name='Handicap')



trace4 = go.Scatter(

    x= Hyper['Month Number'],

    y=Hyper['No Show Rate']*100,

    mode= 'lines+markers',

    text=Hyper['Category'],

    name='HyperTension')



trace5 = go.Scatter(

    x= Smoker['Month Number'],

    y=Smoker['No Show Rate']*100,

    mode= 'lines+markers',

    text=Smoker['Category'],

    name="Smoker")



# Tuberculosis was skewing our data and because the count is so minimal, Let's take it out

# trace6 = go.Scatter(

#     x= Tuber['Month Number'],

#     y=Tuber['No Show Rate']*100,

#     mode= 'lines+markers',

#     text=Tuber['Category'],

#     name="Tuberculosis")





layout = dict(title = "No Shows",hovermode='closest',xaxis=dict({"title":"Month (Starts at January)"}),

             yaxis=dict({"title":"No Show Rate in Percent"}))

dat = [trace,trace2,trace3,trace4,trace5]

fig = go.Figure(data=dat,layout=layout)

iplot(fig,filename='basic-line')
pv2 = noShows.pivot_table(values=["Diabetes","Alcoholism","HyperTension","Handicap","Smoker","Tuberculosis"],columns=["Status"],index=["Month Number",'AgeGroup'],aggfunc=np.sum)
data2 = pv2.unstack().unstack('AgeGroup').unstack('Status').reset_index().rename(columns={"level_0":"Category"})
data2['No Show Rate'] = data2['No-Show'] / (data2['No-Show'] + data2['Show-Up'])

data2['Total'] = data2['No-Show'] + data2['Show-Up']
data2[data2['Category'] == 'Alcoholism'].pivot_table(values="No Show Rate",index="AgeGroup",columns='Month Number')
fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(221)

sns.heatmap(data2[data2['Category'] == 'Alcoholism'].pivot_table(values="No Show Rate",index="AgeGroup",columns='Month Number'),

           cmap='viridis',annot=True)

ax.set_title('Alcohol No-Show Rate')



ax2 = fig.add_subplot(222)

sns.heatmap(data2[data2['Category'] == 'Smoker'].pivot_table(values="No Show Rate",index="AgeGroup",columns='Month Number'),

           cmap='viridis',annot=True)

ax2.set_title('Smoker No-Show Rate')



ax3 = fig.add_subplot(223)

sns.heatmap(data2[data2['Category'] == 'Diabetes'].pivot_table(values="No Show Rate",index="AgeGroup",columns='Month Number'),

           cmap='viridis',annot=True)

ax3.set_title('Diabetes No Show Rate')



ax4 = fig.add_subplot(224)

sns.heatmap(data2[data2['Category'] == 'HyperTension'].pivot_table(values="No Show Rate",index="AgeGroup",columns='Month Number'),

           cmap='viridis',annot=True)

ax4.set_title('HyperTension No Show Rate')



plt.tight_layout()

plt.show()
data2[data2['AgeGroup'] == 'Less than 18'].mean()
fig = plt.figure(figsize=(14,9))

ax = fig.add_subplot(321)

sns.heatmap(data2[(data2['Category'] == 'Alcoholism') & (data2['AgeGroup'] != 'Less than 18')].pivot_table(values="No Show Rate",index="AgeGroup",columns='Month Number'),

           cmap='coolwarm',annot=True)

ax.set_title('Alcohol No-Show Rate')



ax2 = fig.add_subplot(322)

sns.heatmap(data2[(data2['Category'] == 'Smoker') & (data2['AgeGroup'] != 'Less than 18')].pivot_table(values="No Show Rate",index="AgeGroup",columns='Month Number'),

           cmap='coolwarm',annot=True)

ax2.set_title('Smoker No-Show Rate')



ax3 = fig.add_subplot(323)

sns.heatmap(data2[(data2['Category'] == 'Diabetes') & (data2['AgeGroup'] != 'Less than 18')].pivot_table(values="No Show Rate",index="AgeGroup",columns='Month Number'),

           cmap='coolwarm',annot=True)

ax3.set_title('Diabetes No Show Rate')



ax4 = fig.add_subplot(324)

sns.heatmap(data2[(data2['Category'] == 'HyperTension') & (data2['AgeGroup'] != 'Less than 18')].pivot_table(values="No Show Rate",index="AgeGroup",columns='Month Number'),

           cmap='coolwarm',annot=True)

ax4.set_title('HyperTension No Show Rate')



ax4 = fig.add_subplot(325)

sns.heatmap(data2[(data2['Category'] == 'Handicap') & (data2['AgeGroup'] != 'Less than 18')].pivot_table(values="No Show Rate",index="AgeGroup",columns='Month Number'),

           cmap='coolwarm',annot=True)

ax4.set_title('Handicap No Show Rate')



plt.tight_layout()

plt.show()
data2.groupby(['AgeGroup','Category'])['No-Show','Show-Up','Total'].sum()
pv3 = noShows.pivot_table(values=["Diabetes","Alcoholism","HyperTension","Handicap","Smoker","Tuberculosis"],columns=["Status",'Gender','AgeGroup'],index="Month Number",aggfunc=np.sum)

gender = pv3.unstack().unstack('Status').reset_index().rename(columns={"level_0":"Category"})

gender['Total'] = gender['No-Show'] + gender['Show-Up']

gender['No Show Rate'] = gender['No-Show'] / gender['Total']
#We have successfully transformed our data and included Gender.

gender.tail(10)
Gd = gender[(gender['AgeGroup'] != 'Less than 18')]

figure = plt.figure(figsize=(10,6))

ax = figure.add_axes([0,0,1,1])

sns.barplot(x="Category",y="No Show Rate",hue='Gender',data=Gd,estimator=np.mean,palette={'r','b'})

ax.set_title("Gender No-Shows")

ax.set_ylabel("No-Show Average")
plt.figure(figsize=(12,8))

sns.set(style="darkgrid", palette="deep", color_codes=True)

sns.pointplot(x='Gender',y='No Show Rate',hue='Category',data=Gd,estimator=np.mean)
Gd.pivot_table('No Show Rate',index='Gender',columns='Month Number',aggfunc=np.mean)
#SMS and Weekly analysis coming up soon. Curious to see how much of a difference sms messaging matters 

# and also what days of the week in which categories have the highest no show rates.

wk = noShows.pivot_table(values=["Diabetes","Alcoholism","HyperTension","Handicap","Smoker","Tuberculosis"],columns=["Status",'Gender','AgeGroup',"DayOfTheWeek"],index=["Month Number"],aggfunc=np.sum)

weekly = wk.unstack().unstack('Status').reset_index().rename(columns={"level_0":"Category"})

weekly['Total'] = weekly['No-Show'] + weekly['Show-Up']

weekly['No Show Rate'] = weekly['No-Show'] / weekly['Total']
weekly[['No-Show',"Show-Up","Total"]] = weekly[['No-Show',"Show-Up","Total"]].fillna(0).astype(int)

weekly['No Show Rate'] = weekly['No Show Rate'].round(2)
weekly = weekly[weekly['AgeGroup'] != 'Less than 18']
sns.barplot("DayOfTheWeek","No Show Rate",data=weekly,estimator=np.mean,order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
wkfig = plt.figure(figsize=(16,9))

wkfig.add_subplot(221)

plt.title('Alcoholism')

sns.barplot("DayOfTheWeek","No Show Rate",data=weekly[(weekly['Category'] == 'Alcoholism') & (~weekly['DayOfTheWeek'].isin(['Saturday','Sunday']))],estimator=np.mean,

            order=['Monday','Tuesday','Wednesday','Thursday','Friday'],palette='winter',hue=weekly['AgeGroup'])



wkfig.add_subplot(222)

plt.title('HyperTension')

sns.barplot("DayOfTheWeek","No Show Rate",data=weekly[(weekly['Category'] == 'HyperTension') & (~weekly['DayOfTheWeek'].isin(['Saturday','Sunday']))],estimator=np.mean,

            order=['Monday','Tuesday','Wednesday','Thursday','Friday'],palette='summer',hue=weekly['AgeGroup'])



wkfig.add_subplot(223)

plt.title('Diabetes')

sns.barplot("DayOfTheWeek","No Show Rate",data=weekly[(weekly['Category'] == 'Diabetes') & (~weekly['DayOfTheWeek'].isin(['Saturday','Sunday']))],estimator=np.mean,

            order=['Monday','Tuesday','Wednesday','Thursday','Friday'],palette='spring',hue=weekly['AgeGroup'])



wkfig.add_subplot(224)

plt.title('Smoker')

sns.barplot("DayOfTheWeek","No Show Rate",data=weekly[(weekly['Category'] == 'Smoker') & (~weekly['DayOfTheWeek'].isin(['Saturday','Sunday']))],estimator=np.mean,

            order=['Monday','Tuesday','Wednesday','Thursday','Friday'],palette='autumn',hue=weekly['AgeGroup'])



wkfig.tight_layout()

plt.show()
g = sns.PairGrid(weekly[~weekly['DayOfTheWeek'].isin(['Saturday','Sunday'])],x_vars=['Gender','AgeGroup','DayOfTheWeek'],y_vars='No Show Rate',aspect=.75,size=6,

                despine=True)

g.map(sns.barplot,palette='pastel')

plt.tight_layout()
weekly.pivot_table("No Show Rate",index='AgeGroup',columns='DayOfTheWeek').sort_values(['Monday','Tuesday','Wednesday','Thursday','Friday'])