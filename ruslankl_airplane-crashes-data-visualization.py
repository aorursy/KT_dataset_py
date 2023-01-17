#importing the libraries and data

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import date, timedelta, datetime



Data = pd.read_csv('../input/airplane-crashes-since-1908/Airplane_Crashes_and_Fatalities_Since_1908.csv')
np.random.seed(42) 

obs, feat = Data.shape

Data.sample(5)
print(str("Dataset consist of " + str(obs) + " observations (crashes) and " + str(feat) + " features. Features are following:"))
Data.isnull().sum() #calculating missing values in rows
#cleaning up

Data['Time'] = Data['Time'].replace(np.nan, '00:00') 

Data['Time'] = Data['Time'].str.replace('c: ', '')

Data['Time'] = Data['Time'].str.replace('c:', '')

Data['Time'] = Data['Time'].str.replace('c', '')

Data['Time'] = Data['Time'].str.replace('12\'20', '12:20')

Data['Time'] = Data['Time'].str.replace('18.40', '18:40')

Data['Time'] = Data['Time'].str.replace('0943', '09:43')

Data['Time'] = Data['Time'].str.replace('22\'08', '22:08')

Data['Time'] = Data['Time'].str.replace('114:20', '00:00') #is it 11:20 or 14:20 or smth else? 



Data['Time'] = Data['Date'] + ' ' + Data['Time'] #joining two rows

def todate(x):

    return datetime.strptime(x, '%m/%d/%Y %H:%M')

Data['Time'] = Data['Time'].apply(todate) #convert to date type

print('Date ranges from ' + str(Data.Time.min()) + ' to ' + str(Data.Time.max()))



Data.Operator = Data.Operator.str.upper() #just to avoid duplicates like 'British Airlines' and 'BRITISH Airlines'
Temp = Data.groupby(Data.Time.dt.year)[['Date']].count() #Temp is going to be temporary data frame 

Temp = Temp.rename(columns={"Date": "Count"})



plt.figure(figsize=(12,6))

plt.style.use('bmh')

plt.plot(Temp.index, 'Count', data=Temp, color='blue', marker = ".", linewidth=1)

plt.xlabel('Year', fontsize=10)

plt.ylabel('Count', fontsize=10)

plt.title('Count of accidents by Year', loc='Center', fontsize=14)

plt.show()
import matplotlib.pylab as pl

import matplotlib.gridspec as gridspec



gs = gridspec.GridSpec(2, 2)

pl.figure(figsize=(15,10))

plt.style.use('seaborn-muted')

ax = pl.subplot(gs[0, :]) # row 0, col 0

sns.barplot(Data.groupby(Data.Time.dt.month)[['Date']].count().index, 'Date', data=Data.groupby(Data.Time.dt.month)[['Date']].count(), color='lightskyblue', linewidth=2)

plt.xticks(Data.groupby(Data.Time.dt.month)[['Date']].count().index, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.xlabel('Month', fontsize=10)

plt.ylabel('Count', fontsize=10)

plt.title('Count of accidents by Month', loc='Center', fontsize=14)



ax = pl.subplot(gs[1, 0])

sns.barplot(Data.groupby(Data.Time.dt.weekday)[['Date']].count().index, 'Date', data=Data.groupby(Data.Time.dt.weekday)[['Date']].count(), color='lightskyblue', linewidth=2)

plt.xticks(Data.groupby(Data.Time.dt.weekday)[['Date']].count().index, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

plt.xlabel('Day of Week', fontsize=10)

plt.ylabel('Count', fontsize=10)

plt.title('Count of accidents by Day of Week', loc='Center', fontsize=14)



ax = pl.subplot(gs[1, 1])

sns.barplot(Data[Data.Time.dt.hour != 0].groupby(Data.Time.dt.hour)[['Date']].count().index, 'Date', data=Data[Data.Time.dt.hour != 0].groupby(Data.Time.dt.hour)[['Date']].count(),color ='lightskyblue', linewidth=2)

plt.xlabel('Hour', fontsize=10)

plt.ylabel('Count', fontsize=10)

plt.title('Count of accidents by Hour', loc='Center', fontsize=14)

plt.tight_layout()

plt.show()
Temp = Data.copy()

Temp['isMilitary'] = Temp.Operator.str.contains('MILITARY')

Temp = Temp.groupby('isMilitary')[['isMilitary']].count()

Temp.index = ['Passenger', 'Military']



Temp2 = Data.copy()

Temp2['Military'] = Temp2.Operator.str.contains('MILITARY')

Temp2['Passenger'] = Temp2.Military == False

Temp2 = Temp2.loc[:, ['Time', 'Military', 'Passenger']]

Temp2 = Temp2.groupby(Temp2.Time.dt.year)[['Military', 'Passenger']].aggregate(np.count_nonzero)



colors = ['yellowgreen', 'lightskyblue']

plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

patches, texts = plt.pie(Temp.isMilitary, colors=colors, labels=Temp.isMilitary, startangle=90)

plt.legend(patches, Temp.index, loc="best", fontsize=10)

plt.axis('equal')

plt.title('Total number of accidents by Type of flight', loc='Center', fontsize=14)



plt.subplot(1, 2, 2)

plt.plot(Temp2.index, 'Military', data=Temp2, color='lightskyblue', marker = ".", linewidth=1)

plt.plot(Temp2.index, 'Passenger', data=Temp2, color='yellowgreen', marker = ".", linewidth=1)

plt.legend(fontsize=10)

plt.xlabel('Year', fontsize=10)

plt.ylabel('Count', fontsize=10)

plt.title('Count of accidents by Year', loc='Center', fontsize=14)

plt.tight_layout()

plt.show()
Fatalities = Data.groupby(Data.Time.dt.year).sum()

Fatalities['Proportion'] = Fatalities['Fatalities'] / Fatalities['Aboard']



plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

plt.fill_between(Fatalities.index, 'Aboard', data=Fatalities, color="skyblue", alpha=0.2)

plt.plot(Fatalities.index, 'Aboard', data=Fatalities, marker = ".", color="Slateblue", alpha=0.6, linewidth=1)

plt.fill_between(Fatalities.index, 'Fatalities', data=Fatalities, color="olive", alpha=0.2)

plt.plot(Fatalities.index, 'Fatalities', data=Fatalities, color="olive", marker = ".", alpha=0.6, linewidth=1)

plt.legend(fontsize=10)

plt.xlabel('Year', fontsize=10)

plt.ylabel('Amount of people', fontsize=10)

plt.title('Total number of people involved by Year', loc='Center', fontsize=14)



plt.subplot(1, 2, 2)

plt.plot(Fatalities.index, 'Proportion', data=Fatalities, marker = ".", color = 'red', linewidth=1)

plt.xlabel('Year', fontsize=10)

plt.ylabel('Ratio', fontsize=10)

plt.title('Fatalities / Total Ratio by Year', loc='Center', fontsize=14)

plt.tight_layout()

plt.show()
Totals = pd.read_csv('../input/plane-symb/API_IS.AIR.PSGR_DS2_en_csv_v2.csv')

Totals.sample(5)
Totals = Totals.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], axis = 1)

Totals = Totals.replace(np.nan, 0)

Totals = pd.DataFrame(Totals.sum())

Totals = Totals.drop(Totals.index[0:10])

Totals = Totals['1970':'2008']

Totals.columns = ['Sum']

Totals.index.name = 'Year'
Fatalities = Fatalities.reset_index()

Fatalities.Time = Fatalities.Time.apply(str)

Fatalities.index = Fatalities['Time']

del Fatalities['Time']

Fatalities = Fatalities['1970':'2008']

Fatalities = Fatalities[['Fatalities']]

Totals = pd.concat([Totals, Fatalities], axis=1) #joining two data frames into one

Totals['Ratio'] = Totals['Fatalities'] / Totals['Sum'] * 100 #calculating ratio
gs = gridspec.GridSpec(2, 2)

pl.figure(figsize=(15,10))



ax = pl.subplot(gs[0, 0]) 

plt.plot(Totals.index, 'Sum', data=Totals, marker = ".", color = 'green', linewidth=1)

plt.xlabel('Year', fontsize=11)

plt.ylabel('Amount of passengers', fontsize=11)

plt.title('Total amount of air passengers by Year', loc='Center', fontsize=14)

plt.xticks(rotation=90)



ax = pl.subplot(gs[0, 1]) 

plt.plot(Fatalities.index, 'Fatalities', data=Fatalities, color='red', marker = ".", linewidth=1)

plt.xlabel('Year', fontsize=11)

plt.ylabel('Number of Deaths', fontsize=11)

plt.title('Total number of Fatalities by Year', loc='Center', fontsize=14)

plt.xticks(rotation=90)



ax = pl.subplot(gs[1, :]) 

plt.plot(Totals.index, 'Ratio', data=Totals, color='orange', marker = ".", linewidth=1)

plt.xlabel('Year', fontsize=11)

plt.ylabel('Ratio (%)', fontsize=11)

plt.title('Fatalities / Total amount of passegers Ratio by Year', loc='Center', fontsize=14)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
fig =plt.figure(figsize=(12,6))

ax1 = fig.subplots()

ax1.plot(Totals.index, 'Ratio', data=Totals, color='orange', marker = ".", linewidth=1)

ax1.set_xlabel('Year', fontsize=11)

for label in ax1.xaxis.get_ticklabels():

        label.set_rotation(45)

ax1.set_ylabel('Ratio (%)', color='orange', fontsize=11)

ax1.tick_params('y', colors='orange')

ax2 = ax1.twinx()

ax2.plot(Fatalities.index, 'Fatalities', data=Fatalities, color='red', marker = ".", linewidth=1)

ax2.set_ylabel('Number of fatalities', color='red', fontsize=11)

ax2.tick_params('y', colors='r')

plt.title('Fatalities VS Ratio by Year', loc='Center', fontsize=14)

fig.tight_layout()

plt.show()
Data.Operator = Data.Operator.str.upper()

Data.Operator = Data.Operator.replace('A B AEROTRANSPORT', 'AB AEROTRANSPORT')



Total_by_Op = Data.groupby('Operator')[['Operator']].count()

Total_by_Op = Total_by_Op.rename(columns={"Operator": "Count"})

Total_by_Op = Total_by_Op.sort_values(by='Count', ascending=False).head(15)



plt.figure(figsize=(12,6))

sns.barplot(y=Total_by_Op.index, x="Count", data=Total_by_Op, palette="gist_heat", orient='h')

plt.xlabel('Count', fontsize=11)

plt.ylabel('Operator', fontsize=11)

plt.title('Total Count by Opeartor', loc='Center', fontsize=14)

plt.show()
Prop_by_Op = Data.groupby('Operator')[['Fatalities']].sum()

Prop_by_Op = Prop_by_Op.rename(columns={"Operator": "Fatalities"})

Prop_by_Op = Prop_by_Op.sort_values(by='Fatalities', ascending=False)

Prop_by_OpTOP = Prop_by_Op.head(15)



plt.figure(figsize=(12,6))

sns.barplot(y=Prop_by_OpTOP.index, x="Fatalities", data=Prop_by_OpTOP, palette="gist_heat", orient='h')

plt.xlabel('Fatalities', fontsize=11)

plt.ylabel('Operator', fontsize=11)

plt.title('Total Fatalities by Opeartor', loc='Center', fontsize=14)

plt.show()
Prop_by_Op[Prop_by_Op['Fatalities'] == Prop_by_Op.Fatalities.min()].index.tolist()
from PIL import Image

from wordcloud import WordCloud, STOPWORDS



text = str(Data.Summary.tolist())

plane_mask = np.array(Image.open('../input/plane-symb/aircraft-1293790_960_720.jpg'))



stopwords = set(STOPWORDS)

stopwords.add('aircraft')

stopwords.add('plane')



wc = WordCloud(background_color="white", max_words=2000, mask=plane_mask,

               stopwords=stopwords)

wc.generate(text)



plt.figure(figsize=(10,10))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.title('Brief Summary', loc='Center', fontsize=14)

plt.savefig('./aircraft_wordcloud.png', dpi=50)

plt.show()
text = str(Data.Location.tolist())

globe_mask = np.array(Image.open('../input/plane-symb/standing-globe-silhouette-with-support_318-37306.jpg'))



stopwords = set(STOPWORDS)

stopwords.add('nan')

stopwords.add('Near')



wc = WordCloud(background_color="white", max_words=2000, mask=globe_mask,

               stopwords=stopwords)

wc.generate(text)



plt.figure(figsize=(10,10))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.title('Location of Accident', loc='Center', fontsize=14)

plt.show()
Aeroflot = Data[Data.Operator == 'AEROFLOT']



Count_by_Year = Aeroflot.groupby(Data.Time.dt.year)[['Date']].count()

Count_by_Year = Count_by_Year.rename(columns={"Date": "Count"})



plt.figure(figsize=(12,6))

plt.plot(Count_by_Year.index, 'Count', data=Count_by_Year, marker='.', color='red', linewidth=1)

plt.xlabel('Year', fontsize=11)

plt.ylabel('Count', fontsize=11)

plt.title('Count of accidents by Year (Aeroflot)', loc='Center', fontsize=14)

plt.show()