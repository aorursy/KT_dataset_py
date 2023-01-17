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
import pandas as pd

boston= pd.read_csv('../input/crimes-in-boston/crime.csv',encoding='ISO-8859-1',engine='python')

boston.columns = map(str.lower, boston.columns)
boston.drop_duplicates(['incident_number'],inplace=True)
new=boston[['lat','long']]

new.describe()
boston[boston.lat>= 40]

boston[boston.long<=-70]

boston.shooting.unique()# Output: ([nan, 'Y'])

boston['shooting'] = boston.shooting.map({'NaN':0, 'Y':1}) #I change missing values is 0 because if we dont know whether there is a shooting in the area or not, probably there is no shooting incident in a area.

boston.shooting=pd.get_dummies(boston.shooting)

boston.shooting.value_counts()
total = boston.isnull().sum().sort_values(ascending=False)

percent = (boston.isnull().sum()/boston.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data# I can't impute missignees in lat, long street,distinct and ucr_part varibles. If ı drop them, my dataset wont be reduced too much.
boston=boston.dropna()

boston.info()# number of non-null is the same for all variables
import pandas as pd

boston['date']=pd.to_datetime(boston['occurred_on_date']).dt.floor('d')
boston.date = pd.to_datetime(boston.date, format='%Y-%m-%d')
boston['date'].describe().to_frame()
yearpercentages=boston.groupby(['year'])['year'].count()/boston.year.count()*100
yearpercentages.plot.pie(autopct="%.1f%%",title="Amount of the data by years",figsize=(20,10))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

df = boston.groupby(['date'])["incident_number"].count().reset_index()

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111)

p=sns.lineplot(x=df.iloc[:,0], y=df.iloc[:,1], data=df)

p.set_ylabel("No. of Crimes Occurred")

p.set_xlabel("Date")

plt.tight_layout()

plt.show()
plt.figure(figsize=(15,7))

boston.groupby(['year','month']).count()['incident_number'].plot.barh(color = ['green', 'brown', 'orange', 'blue'])
shooting_monthly=boston.groupby(['month'])['month'].count()/len(boston['incident_number'])*100

shooting_monthly.plot.pie(autopct="%.1f%%",title="Yearly Shooting Percentages between 2015 and 2018",figsize=(20,10))
from pandas.api.types import CategoricalDtype

cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

cat_type = CategoricalDtype(categories=cats, ordered=True)

boston['day_of_week'] =boston['day_of_week'].astype(cat_type)

crime_2015=boston[boston['year']==2015]

crime_2016=boston[boston['year']==2016]

crime_2017=boston[boston['year']==2017]

crime_2018=boston[boston['year']==2018]
plt.figure(figsize=(20,5))

plt.title("Bar plot for incident numbers in Boston in 2015 by month and day of week")

crime_2015.groupby(['month','day_of_week']).count()['incident_number'].plot.bar(color = ['blue', 'green', 'yellow', 'orange','black','red','brown'])
plt.figure(figsize=(20,5))

plt.title("Bar plot for incident numbers in Boston in 2016 by month and day of week")

crime_2016.groupby(['month','day_of_week']).count()['incident_number'].plot.bar(color = ['blue', 'green', 'yellow', 'orange','black','red','brown'])
plt.figure(figsize=(20,5))

plt.title("Bar plot for incident numbers in Boston in 2017 by month and day of week")

crime_2017.groupby(['month','day_of_week']).count()['incident_number'].plot.bar(color = ['blue', 'green', 'yellow', 'orange','black','red','brown'])
plt.figure(figsize=(20,5))

plt.title("Bar plot for incident numbers in Boston in 2018 by month and day of week")

crime_2018.groupby(['month','day_of_week']).count()['incident_number'].plot.bar(color = ['blue', 'green', 'yellow', 'orange','black','red','brown'])
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

offenses = boston.groupby('offense_code_group')['incident_number'].count().sort_values(ascending = False).to_frame()

offenses.reset_index(inplace = True)

plt.figure(figsize=(25,8))

ax = sns.barplot(x="offense_code_group", y="incident_number", data=offenses)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90);
offenses1 = crime_2015.groupby('offense_code_group')['incident_number'].count().sort_values(ascending = False).to_frame()

offenses1.reset_index(inplace = True)

offenses2 = crime_2016.groupby('offense_code_group')['incident_number'].count().sort_values(ascending = False).to_frame()

offenses2.reset_index(inplace = True)

offenses3 = crime_2017.groupby('offense_code_group')['incident_number'].count().sort_values(ascending = False).to_frame()

offenses3.reset_index(inplace = True)

offenses4 = crime_2018.groupby('offense_code_group')['incident_number'].count().sort_values(ascending = False).to_frame()

offenses4.reset_index(inplace = True)

o2015 = offenses1.rename(columns = {'offense_code_group': 'offense_code_group2015'}, inplace = False)

o2016=offenses2.rename(columns = {'offense_code_group': 'offense_code_group2016'}, inplace = False)

o2017=offenses3.rename(columns = {'offense_code_group': 'offense_code_group2017'}, inplace = False)

o2018=offenses4.rename(columns = {'offense_code_group': 'offense_code_group2018'}, inplace = False)

df_col = pd.concat([o2015,o2016,o2017,o2018], axis=1).drop('incident_number',axis=1)

df_col.head(5)
fig = plt.figure(figsize=(20,10))

order2 = boston['offense_code_group'].value_counts().sort_values(ascending=False).head(6).index

sns.countplot(data =boston, x='offense_code_group',hue='district', order = order2,palette='Paired' );

plt.ylabel("Offense Amount");
import matplotlib.pyplot as plt

import seaborn as sns



fig, axes = plt.subplots(1,4, figsize = (40,10))

df2015 = crime_2015.groupby(["district"])["incident_number"].count().reset_index()

df2015

sns.set(style="whitegrid")

sns.swarmplot(x='district',y='incident_number',data=df2015,ax=axes[0]).set_title('Incident numbers in 2015 by distinct')



df2016 = crime_2016.groupby(["district"])["incident_number"].count().reset_index()

df2016

sns.set(style="whitegrid")

sns.swarmplot(x='district',y='incident_number',data=df2016,ax=axes[1]).set_title('Incident numbers in 2016 by distinct')



df2017 = crime_2017.groupby(["district"])["incident_number"].count().reset_index()

df2017

sns.set(style="whitegrid")

sns.swarmplot(x='district',y='incident_number',data=df2017,ax=axes[2]).set_title('Incident numbers in 2017by distinct')



df2018 = crime_2018.groupby(["district"])["incident_number"].count().reset_index()

df2018

sns.set(style="whitegrid")

sns.swarmplot(x='district',y='incident_number',data=df2018,ax=axes[3]).set_title('Incident numbers in 2018 by distinct')

import matplotlib.pyplot as plt

import seaborn as sns



fig, axes = plt.subplots(1,4, figsize = (40,6))

df2015 = crime_2015.groupby(["district"])["shooting"].count().reset_index()

df2015

sns.set(style="whitegrid")

sns.swarmplot(x='district',y='shooting',data=df2015,ax=axes[0]).set_title('Shooting numbers in 2015 by distinct')



df2016 = crime_2016.groupby(["district"])['shooting'].count().reset_index()

df2016

sns.set(style="whitegrid")

sns.swarmplot(x='district',y='shooting',data=df2016,ax=axes[1]).set_title('Shooting numbers in 2016 by distinct')



df2017 = crime_2017.groupby(["district"])["shooting"].count().reset_index()

df2017

sns.set(style="whitegrid")

sns.swarmplot(x='district',y='shooting',data=df2017,ax=axes[2]).set_title('Shooting numbers in 2017 by distinct')



df2018 = crime_2018.groupby(["district"])["shooting"].count().reset_index()

df2018

sns.set(style="whitegrid")

sns.swarmplot(x='district',y='shooting',data=df2018,ax=axes[3]).set_title('Shooting numbers in 2018 by distinct')
fig = plt.figure(figsize=(12,5))

crime_street = boston.groupby('street')['shooting'].count().nlargest(10)

crime_street.plot(kind='bar', color ="saddlebrown")

plt.xlabel("street")

plt.ylabel("Shooting Count")

plt.show()
hour_nums = boston.groupby(['hour']).count()['incident_number'].to_frame().reset_index()

sns.set(rc={'figure.figsize':(20,5)})

ax = sns.barplot(x = 'hour' , y="incident_number", data = hour_nums)
boston_map=boston.groupby(['lat','long'])['incident_number'].count().reset_index()

import plotly.express as px

fig = px.density_mapbox(boston_map, lat='lat', lon='long', z='incident_number', radius=6,center=dict(lat=42, lon=-71), zoom=7, mapbox_style="stamen-terrain")

fig.show()