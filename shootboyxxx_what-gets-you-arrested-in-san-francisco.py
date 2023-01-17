import warnings

warnings.filterwarnings('ignore')



import numpy as np, pandas as pd, matplotlib.pyplot as plt

import seaborn as sns 



data = pd.read_csv('../input/sanfranciso-crime-dataset/Police_Department_Incidents_-_Previous_Year__2016_.csv')
data.head()
data.info()
data.describe()
#deleting incident number, location as the data is already captured

data.drop(columns = ['IncidntNum','Location'], axis = 1, inplace = True)



#feature engineering - extracting month, day and week of the month

data.Date = data.Date.str.split()

data.Date = data.Date.apply(lambda x: x[0])



data['month'] = pd.DatetimeIndex(data.Date).month

data['day'] = pd.DatetimeIndex(data.Date).day



data['iswknd'] = data.DayOfWeek.apply(lambda x: 1 if x in ['Saturday','Sunday'] else 0)



def PartofDay(a):

    if a in ['23','00','01','02']:

        return 'Midnight'

    elif a in ['03','04','05','06']:

        return 'Early Morning'

    elif a in ['07','08','09','10']:

        return 'Morning'

    elif a in ['11','12','13','14']:

        return 'Noon'

    elif a in ['15','16','17','18']:

        return 'Evening'

    else:

        return 'Night'



data['PartofDay'] = data.Time.apply(lambda x: PartofDay(x.split(':')[0]))
a = data.Category.value_counts().reset_index()

plt.figure(figsize = (15,6))

sns.barplot(x = a['index'], y = 'Category', data = a, color = 'red')

plt.xticks(rotation = 90)

plt.xlabel("")

plt.title('Categories of Crime', fontdict = {'color': 'darkred', 'size':16})

plt.text(x = 20, y = 35000,s = "Larceny/Theft : 26% \nNon-Criminal  : 11%", fontsize = 16, color = 'blue')

plt.show()
a = data.DayOfWeek.value_counts(normalize = True).reset_index()

a['DayOfWeek'] = 100*a['DayOfWeek']

plt.figure(figsize = (7,7))

plt.pie(x = a['DayOfWeek'], labels = a['index'], explode = [0.1,0,0,0,0,0,0], shadow = True, autopct = '%1.1f%%')

plt.show()
a = data.PdDistrict.value_counts().reset_index()

plt.figure(figsize = [15,5])

sns.barplot(x = a['index'],y = 'PdDistrict', data = a, palette = ['red']+['grey']*8+['blue'])

plt.xlabel('')

plt.ylabel('Count')

plt.title('District Wise counts of crime', fontdict = {'color': 'darkred', 'size':16})

plt.show()
a = data.Resolution.value_counts().reset_index()

plt.figure(figsize = (10, 6))

sns.barplot(y = a['index'], x = 'Resolution', data = a, color = 'red')

plt.title('Charges for cases', fontdict = {'color': 'darkred', 'size':16})

plt.xlabel('')

plt.ylabel('')

plt.text(x = 40000, y = 1.3, s = '26% of the incidents ends up in Arrest',fontsize = 16, color = 'blue')

plt.show()
a = data.month.value_counts().reset_index()

a.sort_values('index', inplace = True)

m = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']

plt.figure(figsize = (15,4))

sns.barplot(x = a['index'], y = 'month', data = a, color = 'red')

plt.xticks(np.arange(0,12),m)

plt.title('Months vs Crimes', fontdict = {'color': 'darkred', 'size':16})

a1 = data.month.value_counts(normalize = True)

for i in range(1,13):

    plt.text(x = i-1, y = 2000, s = str(round(100*a1[i],2))+'%'+'\n'+str(a.index[i-1]+1), horizontalalignment = 'center', verticalalignment = 'center', color = 'white')

plt.show()
a = data.PartofDay.value_counts().reset_index()

a1 = data.PartofDay.value_counts(normalize = True)

plt.figure(figsize = (15,5))

sns.barplot(x = a['index'], y = 'PartofDay', data = a, color = 'red')

plt.title('Part of the Day vs Crimes', fontdict = {'color': 'darkred', 'size':16})

for i in range(6):

    plt.text(x = i, y = 5000, s = str(round(100*a1[i],2))+'%', horizontalalignment = 'center', color = 'white')

plt.show()
res = pd.pivot_table(columns = data.day,values = 'day' ,index = 'month' , aggfunc = 'count', data = data)

plt.figure(figsize = (15,6))

sns.heatmap(res, cmap = 'Reds')

plt.yticks(np.arange(0.5,12.5) ,['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec'])

plt.xlabel('')

plt.ylabel('')

plt.show()
res = pd.pivot_table(columns = data['PartofDay'] ,index = 'DayOfWeek',values = 'PartofDay' , aggfunc = 'count', data = data)

res = res.reindex(index = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'], columns = ['Early Morning','Morning','Noon','Evening','Night','Midnight'])

plt.figure(figsize = (15,6))

sns.heatmap(res, cmap = 'Reds',annot = True,fmt = 'd')

plt.xlabel('')

plt.ylabel('')

plt.show()
data_arrest = data[data.Resolution.isin(["ARREST, BOOKED","ARREST, CITED"])]
a = data_arrest.Category.value_counts().reset_index()

b = data.Category.value_counts().reset_index()



a = a.merge(b, how = 'inner', on = 'index')

a.columns = ['Category','Arrests','Cases']



a['PercArrests'] = round(100*(a['Arrests']/a['Cases']),2)



a.sort_values('PercArrests',ascending = False, inplace = True)
plt.figure(figsize = (15,8))

sns.barplot(y = 'Category', x = 'PercArrests',data = a, color = 'red')

plt.ylabel('')

plt.xlabel('')

plt.title('What gets you arrested', fontdict = {'color': 'darkred', 'size':16})

plt.show()
a = data_arrest.PdDistrict.value_counts().reset_index()

b = data.PdDistrict.value_counts().reset_index()
a = a.merge(b,how = 'inner', on = 'index')

a.columns = ['PdDistrict', 'Arrests','Cases']

a['PercArrests'] = a['Arrests']/a['Cases']

a['PercArrests'] = 100*a['PercArrests']

a['PercArrests'] = round(a['PercArrests'],2)

a.sort_values('PercArrests', ascending = False, inplace = True)
plt.figure(figsize = (15,4))

sns.barplot(x = 'PdDistrict', y = 'PercArrests',data = a, color = 'red', palette = ['red']+['grey']*8+['blue'])

plt.ylabel('')

plt.xlabel('')

plt.title('Stern Cops : Percentage of Arrests', fontdict = {'color': 'darkred', 'size':16})

for i in range(10):

    plt.text(x = i, y = a.PercArrests.iloc[i], s = str(a.PercArrests.iloc[i])+'%', horizontalalignment = 'center', verticalalignment = 'bottom')

plt.show()
import plotly.express as px
a = (data.Category.value_counts()<1500).reset_index()

l = list(a[a.Category == True]['index'])
data.Category = data.Category.replace(l,'OTHER OFFENSES')

discat = data.groupby(['PdDistrict','Category'])['Descript'].count().reset_index()
discat = discat.rename(columns = {'Descript':'Count'})

fig = px.sunburst(discat, path = ['PdDistrict','Category'], values = 'Count', color = 'Count')

fig.show()