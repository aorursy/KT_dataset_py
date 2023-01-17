import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True) #do not miss this line
#py.offline.iplot(fig)
df=pd.read_csv('../input/crime-against-women-20012014-india/crimes_against_women_2001-2014.csv')
df.head()
#Dropping District Column and Unnamed 0 column

df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.drop(['DISTRICT'],axis=1,inplace=True)
print(df['STATE/UT'].unique())
len(df['STATE/UT'].unique())
#Removing repeated entries

def get_case_consistency(row):
    row = row['STATE/UT'].strip()
    row = row.upper()
    #row = row.title()
    return row

df['STATE/UT'] = df.apply(get_case_consistency, axis=1)

df['STATE/UT'].replace("A&N ISLANDS", "A & N ISLANDS", inplace = True)
df['STATE/UT'].replace("D&N HAVELI", "D & N HAVELI", inplace = True)
df['STATE/UT'].replace("DELHI UT", "DELHI", inplace = True)

df['STATE/UT'].unique()
#Check for null values
df.isnull().sum()
def func(total_case_all_years):

    for i in list(df.columns)[2:]:
        total_case_all_years[i]=df.groupby(['Year'])[i].sum()
        
    return total_case_all_years

total_case_all_years=pd.DataFrame()
total_case_all_years=func(total_case_all_years)
total_case_all_years
#Total Number of cases in each year
pd.DataFrame(total_case_all_years.sum(axis=1),columns=['Total Number of Cases'])
fig = px.bar(pd.DataFrame( total_case_all_years.sum(axis=1),columns=['Total Number of Cases']), 
             x=pd.DataFrame( total_case_all_years.sum(axis=1)).index, 
             y='Total Number of Cases',title='Total Number of Crimes in Each year',color_discrete_sequence=['green'])
fig.show()
pd.DataFrame( total_case_all_years.sum(axis=0),columns=['Count']).sort_values(by='Count',ascending=False)
fig = px.pie(pd.DataFrame( total_case_all_years.sum(axis=0),columns=['Count']), values='Count', names=pd.DataFrame( total_case_all_years.sum(axis=0)).index, title='Percentage of Each Crime between 2001 - 2014')
fig.show()
fig = px.bar(total_case_all_years, x=total_case_all_years.index, y='Rape',title='Year Wise Rape Cases',color_discrete_sequence=['black'])
fig.show()
fig = px.bar(total_case_all_years, x=total_case_all_years.index, 
             y='Kidnapping and Abduction',color_discrete_sequence=['brown'],title='Year Wise Kidnapping and Abduction Cases')
fig.show()
fig = px.bar(total_case_all_years, x=total_case_all_years.index, y='Dowry Deaths',color_discrete_sequence=['purple'],title='Year wise Dowry Deaths Cases')
fig.show()
fig = px.bar(total_case_all_years, x=total_case_all_years.index, 
             y='Assault on women with intent to outrage her modesty',color_discrete_sequence=['darkolivegreen'],
            title='Year Wise Assault on women with intent to outrage her modesty Cases')
fig.show()
fig = px.bar(total_case_all_years, x=total_case_all_years.index, 
             y='Insult to modesty of Women',color_discrete_sequence=['brown'],title="Year Wise Insult to modesty of women Cases")
fig.show()
fig = px.bar(total_case_all_years, x=total_case_all_years.index, y='Cruelty by Husband or his Relatives',
             color_discrete_sequence=['black'],title='Year Wise Cruelty by Husband or his Relatives')
fig.show()
fig = px.bar(total_case_all_years, x=total_case_all_years.index, y='Importation of Girls',
             color_discrete_sequence=['darkgreen'],title='Year Wise Importation Of Girls Cases')
fig.show()
#Creating a dataframe that consists number of cases state/ut wise

def func(state_wise_data):

    for i in list(df.columns)[2:]:
        state_wise_data[i]=df.groupby(['STATE/UT'])[i].sum()
        
    return state_wise_data

state_wise_data=pd.DataFrame()
state_wise_data=func(state_wise_data)
state_wise_data
state=pd.DataFrame( state_wise_data.sum(axis=1),columns=['Total Cases'])
#state.head()
pd.DataFrame( state_wise_data.sum(axis=1),columns=['Total Count']).sort_values(by='Total Count',ascending=False).head(10)
pd.DataFrame( state_wise_data.sum(axis=1),columns=['Total Case'] ).sort_values(by='Total Case').head(10)
fig = px.pie(state, values='Total Cases', names=state.index, title='Total Crime Rate state/ut wise Distribution')
fig.show()
pd.DataFrame(state_wise_data.idxmax(),columns=['STATE / UT'])
print('**Top 5 States/UT with highest number of Rape Cases**')
display(pd.DataFrame(state_wise_data['Rape']).sort_values(by='Rape',ascending=False).head())

print('\n\n**Top 5 States/UT with lowest number of Rape Cases**')
pd.DataFrame(state_wise_data['Rape']).sort_values(by='Rape').head()
print('**Top 5 States/UT With Highest Number of Kidnapping And Abduction Cases**')
display(pd.DataFrame(state_wise_data['Kidnapping and Abduction']).sort_values(by='Kidnapping and Abduction',ascending=False).head())

print('\n\n**Top 5 States/UT With Lowest Number of Kidnapping And Abduction Cases**')
pd.DataFrame(state_wise_data['Kidnapping and Abduction']).sort_values(by='Kidnapping and Abduction').head()
print('**Top 5 States/UT With Highest Number of Dowry Deaths Cases**')
display(pd.DataFrame(state_wise_data['Dowry Deaths']).sort_values(by='Dowry Deaths',ascending=False).head())

print('\n\n**Top 5 States/UT With Lowest Number of Dowry Deaths Cases**')
pd.DataFrame(state_wise_data['Dowry Deaths']).sort_values(by='Dowry Deaths').head()
print('**Top 5 States/UT With Highest Number of Assault on women with intent to outrage her modesty Cases**')
display(pd.DataFrame(state_wise_data['Assault on women with intent to outrage her modesty']).sort_values(by='Assault on women with intent to outrage her modesty',ascending=False).head())

print('\n\n**Top 5 States/UT With Lowest Number of Assault on women with intent to outrage her modesty Cases**')
pd.DataFrame(state_wise_data['Assault on women with intent to outrage her modesty']).sort_values(by='Assault on women with intent to outrage her modesty').head()
print('**Top 5 States/UT With Highest Number of Insult to modesty of Women Cases**')
display(pd.DataFrame(state_wise_data['Insult to modesty of Women']).sort_values(by='Insult to modesty of Women',ascending=False).head())

print('\n\n**Top 5 States/UT With Lowest Number of Insult to modesty of Women Cases**')
pd.DataFrame(state_wise_data['Insult to modesty of Women']).sort_values(by='Insult to modesty of Women').head()
print('**Top 5 States/UT With Highest Number of Cruelty by Husband or his Relatives Cases**')
display(pd.DataFrame(state_wise_data['Cruelty by Husband or his Relatives']).sort_values(by='Cruelty by Husband or his Relatives',ascending=False).head())

print('\n\n**Top 5 States/UT With Lowest Number of Cruelty by Husband or his Relatives Cases**')
pd.DataFrame(state_wise_data['Cruelty by Husband or his Relatives']).sort_values(by='Cruelty by Husband or his Relatives').head()
print('**Top 5 States/UT With Highest Number of Importation of Girls Cases**')
display(pd.DataFrame(state_wise_data['Importation of Girls']).sort_values(by='Importation of Girls',ascending=False).head())

print('\n\n**Top 5 States/UT With Lowest Number of Cruelty by Husband or his Relatives Cases**')
pd.DataFrame(state_wise_data['Importation of Girls']).sort_values(by='Importation of Girls').head()
def which_state_you_want_to_analyze(state_name):
    try:
        fig = px.pie(state_wise_data, values=state_wise_data.loc[state_name], 
                     names=state_wise_data.iloc[0,:].index, title='Total Crime Rate Distribution for {}'.format(state_name))
        fig.show()
    except KeyError:
        print('You Entered Wrong STATE/UT Name')
    
state_name=input('Enter Name of State/UT : ').upper()
which_state_you_want_to_analyze(state_name)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,10)
a=np.arange(len(total_case_all_years.T.index))
width=0.35
fig, ax = plt.subplots()
rects1 = ax.bar(a - width/2, total_case_all_years.T[2001], width, label='2001')
rects2 = ax.bar(a + width/2, total_case_all_years.T[2014], width, label='2014')
ax.set_xticks(a)
ax.set_xticklabels(total_case_all_years.T.index)
ax.legend()
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(25,13)
plt.xlabel('\nTYPES OF CRIME')
plt.ylabel('NUMBER OF CRIME ')
plt.title('2001 VS 2014 CRIME RATE COMPARASION')
data_2001=pd.DataFrame()

for i in df.columns[2:]:
    data_2001[i]=df[df.Year==2001].groupby('STATE/UT')[i].sum()//2
    
data_2001.head()
data_2001['Total Cases in 2001']=data_2001.sum(axis=1)


data_2014=pd.DataFrame()

for i in df.columns[2:]:
    data_2014[i]=df[df.Year==2014].groupby('STATE/UT')[i].sum()//2
    
data_2014['Total Cases in 2014']=data_2014.sum(axis=1)
#data_2014.drop(['TELANGANA'],inplace=True)
comparasion_between_2001_2014=pd.DataFrame()

comparasion_between_2001_2014['STATE/UT']=data_2001.index
comparasion_between_2001_2014.set_index('STATE/UT',inplace=True)

comparasion_between_2001_2014['Total Cases in 2001']=data_2001['Total Cases in 2001']
comparasion_between_2001_2014['Total Cases in 2014']=data_2014['Total Cases in 2014']

comparasion_between_2001_2014['Increase in number of crimes in 14 Years']=data_2014['Total Cases in 2014']-data_2001['Total Cases in 2001']
comparasion_between_2001_2014
comparasion_between_2001_2014.sort_values(by='Increase in number of crimes in 14 Years',ascending=False).head()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,10)
a=np.arange(5)
width=0.35
fig, ax = plt.subplots()
rects1 = ax.bar(a - width/2, comparasion_between_2001_2014.sort_values(by='Increase in number of crimes in 14 Years',ascending=False).head()['Total Cases in 2001'], width, label='2001')
rects2 = ax.bar(a + width/2, comparasion_between_2001_2014.sort_values(by='Increase in number of crimes in 14 Years',ascending=False).head()['Total Cases in 2014'], width, label='2014')
ax.set_xticks(a)
ax.set_xticklabels(comparasion_between_2001_2014.sort_values(by='Increase in number of crimes in 14 Years',ascending=False).head().index)
ax.legend()
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(20,10)
plt.xlabel('\nSTATE / UT')
plt.ylabel('Total NUMBER OF CRIMES ')
plt.title('Increase In Total Crimes In 14 Years')

fig = px.bar(comparasion_between_2001_2014, x=comparasion_between_2001_2014.sort_values(by='Increase in number of crimes in 14 Years',ascending=False).head().index, 
             y=comparasion_between_2001_2014.sort_values(by='Increase in number of crimes in 14 Years',ascending=False).head()['Increase in number of crimes in 14 Years'],
            labels={
                'y':'Increase in number of cases in 14 Years',
                'x':'STATES / UT'
            },color_discrete_sequence=['black'])
fig.show()
comparasion_between_2001_2014.sort_values(by='Increase in number of crimes in 14 Years').head()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10,10)
a=np.arange(5)
width=0.35
fig, ax = plt.subplots()
rects1 = ax.bar(a - width/2, comparasion_between_2001_2014.sort_values(by='Increase in number of crimes in 14 Years').head()['Total Cases in 2001'], width, label='2001')
rects2 = ax.bar(a + width/2, comparasion_between_2001_2014.sort_values(by='Increase in number of crimes in 14 Years').head()['Total Cases in 2014'], width, label='2014')
ax.set_xticks(a)
ax.set_xticklabels(comparasion_between_2001_2014.sort_values(by='Increase in number of crimes in 14 Years').head().index)
ax.legend()
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(20,10)
plt.xlabel('\nSTATE / UT')
plt.ylabel('Total NUMBER OF CRIMES ')
plt.title('Increase In Total Crimes In 14 Years')

fig = px.bar(comparasion_between_2001_2014, x=comparasion_between_2001_2014.sort_values(by='Increase in number of crimes in 14 Years').head().index, 
             y=comparasion_between_2001_2014.sort_values(by='Increase in number of crimes in 14 Years').head()['Increase in number of crimes in 14 Years'],
            labels={
                'y':'Increase in number of cases in 14 Years',
                'x':'STATES / UT'
            },color_discrete_sequence=['deepskyblue'])
fig.show()