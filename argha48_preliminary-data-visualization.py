import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
import os

destdir = '../input/'

files = [ f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir,f)) ]
files
#df2014 = pd.read_csv('../input/Parking_Violations_Issued_-_Fiscal_Year_2014.csv')

#df2015 = pd.read_csv('../input/Parking_Violations_Issued_-_Fiscal_Year_2015.csv')

df2016 = pd.read_csv('../input/Parking_Violations_Issued_-_Fiscal_Year_2016.csv', nrows = 100000)

#df2017 = pd.read_csv('../input/Parking_Violations_Issued_-_Fiscal_Year_2017.csv')

#df2018 = pd.read_csv('../input/Parking_Violations_Issued_-_Fiscal_Year_2018.csv')
df2016.head(n=2)
df2016.shape
d = {'Unique Entry': df2016.nunique(axis = 0),

        'Nan Entry': df2016.isnull().any()}

pd.DataFrame(data = d, index = df2016.columns.values)
drop_column = ['No Standing or Stopping Violation', 'Hydrant Violation',

               'Double Parking Violation', 'Latitude', 'Longitude',

               'Community Board', 'Community Council ', 'Census Tract', 'BIN',

               'BBL', 'NTA',

               'Street Code1', 'Street Code2', 'Street Code3','Meter Number', 'Violation Post Code',

                'Law Section', 'Sub Division', 'House Number', 'Street Name']

df2016.drop(drop_column, axis = 1, inplace = True)
drop_row = ['Plate ID']

df2016.dropna(axis = 0, how = 'any', subset = drop_row, inplace = True)
df2016['Plate ID'].isnull().any()
df2016.shape
mini2016 = df2016.sample(frac = 0.1, replace = False)
mini2016.shape
x_ticks = mini2016['Registration State'].value_counts().index

heights = mini2016['Registration State'].value_counts()

y_pos = np.arange(len(x_ticks))

fig = plt.figure(figsize=(15,14)) 

# Create horizontal bars

plt.barh(y_pos, heights)

 

# Create names on the y-axis

plt.yticks(y_pos, x_ticks)

 

# Show graphic

plt.show()

pd.DataFrame(mini2016['Registration State'].value_counts()/len(mini2016)).nlargest(10, columns = ['Registration State'])
month = []

for time_stamp in pd.to_datetime(mini2016['Issue Date']):

    month.append(time_stamp.month)

m_count = pd.Series(month).value_counts()



plt.figure(figsize=(12,8))

sns.barplot(y=m_count.values, x=m_count.index, alpha=0.6)

plt.title("Number of Parking Ticket Given Each Month", fontsize=16)

plt.xlabel("Month", fontsize=16)

plt.ylabel("No. of cars", fontsize=16)

plt.show();
violation_code = mini2016['Violation Code'].value_counts()



plt.figure(figsize=(16,8))

f = sns.barplot(y=violation_code.values, x=violation_code.index, alpha=0.6)

#plt.xticks(np.arange(0,101, 10.0))

f.set(xticks=np.arange(0,100, 5.0))

plt.title("Number of Parking Tickets Given for Each Violation Code", fontsize=16)

plt.xlabel("Violation Code [ X5 ]", fontsize=16)

plt.ylabel("No. of cars", fontsize=16)

plt.show();
x_ticks = mini2016['Vehicle Body Type'].value_counts().index

heights = mini2016['Vehicle Body Type'].value_counts().values

y_pos = np.arange(len(x_ticks))

fig = plt.figure(figsize=(15,4))

f = sns.barplot(y=heights, x=y_pos, orient = 'v', alpha=0.6);

# remove labels

plt.tick_params(labelbottom='off')

plt.ylabel('No. of cars', fontsize=16);

plt.xlabel('Car models [Label turned off due to crowding. Too many types.]', fontsize=16);

plt.title('Parking ticket given for different type of car body', fontsize=16);

df_bodytype = pd.DataFrame(mini2016['Vehicle Body Type'].value_counts() / len(mini2016)).nlargest(10, columns = ['Vehicle Body Type'])
df_bodytype
df_bodytype.sum(axis = 0)/len(mini2016)
vehicle_make = mini2016['Vehicle Make'].value_counts()



plt.figure(figsize=(16,8))

f = sns.barplot(y=np.log(vehicle_make.values), x=vehicle_make.index, alpha=0.6)

# remove labels

plt.tick_params(labelbottom='off')

plt.ylabel('log(No. of cars)', fontsize=16);

plt.xlabel('Car make [Label turned off due to crowding. Too many companies!]', fontsize=16);

plt.title('Parking ticket given for different type of car make', fontsize=16);



plt.show();
pd.DataFrame(mini2016['Vehicle Make'].value_counts() / len(mini2016)).nlargest(10, columns = ['Vehicle Make'])
timestamp = []

for time in mini2016['Violation Time']:

    if len(str(time)) == 5:

        time = time[:2] + ':' + time[2:]

        timestamp.append(pd.to_datetime(time, errors='coerce'))

    else:

        timestamp.append(pd.NaT)

    



mini2016 = mini2016.assign(Violation_Time2 = timestamp)

mini2016.drop(['Violation Time'], axis = 1, inplace = True)

mini2016.rename(index=str, columns={"Violation_Time2": "Violation Time"}, inplace = True)
hours = [lambda x: x.hour, mini2016['Violation Time']]
# Getting the histogram

mini2016.set_index('Violation Time', drop=False, inplace=True)

plt.figure(figsize=(16,8))

mini2016['Violation Time'].groupby(pd.TimeGrouper(freq='30Min')).count().plot(kind='bar');

plt.tick_params(labelbottom='on')

plt.ylabel('No. of cars', fontsize=16);

plt.xlabel('Day Time', fontsize=16);

plt.title('Parking ticket given at different time of the day', fontsize=16);

violation_county = mini2016['Violation County'].value_counts()



plt.figure(figsize=(16,8))

f = sns.barplot(y=violation_county.values, x=violation_county.index, alpha=0.6)

# remove labels

plt.tick_params(labelbottom='on')

plt.ylabel('No. of cars', fontsize=16);

plt.xlabel('County', fontsize=16);

plt.title('Parking ticket given in different counties', fontsize=16);
sns.countplot(x = 'Unregistered Vehicle?', data = mini2016)
mini2016['Unregistered Vehicle?'].unique()
pd.DataFrame(mini2016['Vehicle Year'].value_counts()).nlargest(10, columns = ['Vehicle Year'])
plt.figure(figsize=(20,8))

sns.countplot(x = 'Vehicle Year', data = mini2016.loc[(mini2016['Vehicle Year']>1980) & (mini2016['Vehicle Year'] <= 2018)]);
plt.figure(figsize=(16,8))

sns.countplot(x = 'Violation In Front Of Or Opposite', data = mini2016);
# create data

names = mini2016['Violation In Front Of Or Opposite'].value_counts().index

size = mini2016['Violation In Front Of Or Opposite'].value_counts().values

 

# Create a circle for the center of the plot

my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.figure(figsize=(8,8))

from palettable.colorbrewer.qualitative import Pastel1_7

plt.pie(size, labels=names, colors=Pastel1_7.hex_colors)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
