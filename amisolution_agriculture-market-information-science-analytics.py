import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import os
# print(os.listdir("../input"))
#data = pd.read_csv("../input/oecd-crsw-crop-production-since-1990/oecd-CRSW-crop-production.csv", index_col='TIME', parse_dates=True)
data = pd.read_csv("../input/oecd-crsw-crop-production-since-1990/oecd-CRSW-crop-production.csv")
amis = pd.read_csv("../input/amis-fao-crop-production-crsw/CRSW-AMIS.csv")

print('data: \nRows: {}\nCols: {}'.format(data.shape[0],data.shape[1]))
print(data.columns)

print('\nAmis data: \nRows: {}\nCols: {}'.format(amis.shape[0],amis.shape[1]))
print(amis.columns)
amis.index
import numpy as np
import pandas as pd
import math
import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
data.describe()
#amis.describe()
#data.describe();
amis.describe()
data.head(5)
data.sample(10)
#amis.sample()
#data.sample()
amis.sample(2)
pivoted = amis.pivot_table('Value', index="Year", columns="Country/Region Name")
pivoted.iloc[:40, :60]
pivoted = data.pivot_table('Value', index="LOCATION", columns="TIME")
pivoted.iloc[:400000, :200000]
data.index
rice_countries = ['ARG','AUS','BRA','BRICS',
                 'CHN','IND','JPN','KOR',
                  'OECD','PAK','PHL','THA',
                    'TUR','UKR','USA','VNM']

colors = ['blue','xkcd:medium grey','red','green','pink',
          'xkcd:muted blue','yellow','magenta','brown',
          'orange','xkcd:tan','xkcd:seafoam','tab:olive',
          'xkcd:turquoise','xkcd:mauve','xkcd:acid green',
          'xkcd:bland','xkcd:coral','xkcd:chocolate','xkcd:red purple',
          'xkcd:bright lilac','xkcd:heather']

#years=np.sort(data.year.unique())
years=np.sort(data.dtypes.unique())

rice_df = pd.DataFrame()
for LOCATION in rice_countries:
    rice_df=rice_df.append(data[data.LOCATION.isin([LOCATION])])
fig= plt.figure(figsize=(15,7))
region=[]
sub_region=[]
for LOCATION in rice_countries:
    region.append(rice_df[rice_df.LOCATION.isin([LOCATION])]["OECD"].unique()[0])
    sub_region.append(rice_df[rice_df.LOCATION.isin([LOCATION])]["BRICS"].unique()[0])
plt.subplot2grid((1,2),(0,0))
sns.countplot(pd.Series(region))
plt.title("Rice countries distribution according to OECD")
plt.subplot2grid((1,2),(0,1))
sns.countplot(pd.Series(sub_region))
plt.title("Rice countries distribution according to BRICS")
def extract_country_by_record(data,LOCATION,SUBJECT):
    country_foot_print=data[data.country.isin([LOCATION])]
    country_by_record = country_foot_print [country_foot_print.record.isin([SUBJECT])]
    return country_by_record

def extract_countries_feature_by_year(data,LOCATION,Value,TIME,SUBJECT="RICE"):
    excluded_countries=[]
    feature_values=[]
    available_countries=[]
    for i in range (0,len(LOCATION)):
        country_by_record = extract_country_by_record(data,LOCATION[i],SUBJECT)
        feature_value = country_by_record.loc[lambda df1: country_by_record.year == year][Value].values
        if  feature_value.size==0 or math.isnan(feature_value[0]) :
            excluded_countries.append(LOCATION[i])
        else:
            feature_values.append(feature_value[0]) 
            available_countries.append(LOCATION[i])
        return feature_values, available_countries, excluded_countries
    
def print_excluded_countries (excluded_countries,year):
    if len(excluded_countries) != 0:
        print("excluded countries from dataset in {0} are : ".format(year))
        for i in excluded_countries:
            print(i)   
            
def calculate_growth_rate(present,past,period):
    #present : present year , past: past year , period: number of years between present and past
    percentage_growth_rate = ((present - past)/(past*period))*100
    return percentage_growth_rate
fig = plt.figure(figsize=(15,20))
plt.subplot2grid((2,1),(0,0))
population,available_countries,excluded_countries=extract_countries_feature_by_year(rice_df,rice_countries,'population',2014)
population_df = pd.DataFrame({'countery':available_countries,'population':population}).sort_values(by='population',ascending=False)
population_list = list (population_df['population'])
# to avoid overlap of labels at the small slices in the chart the explode added 
# the explode len must be the same len of the pie data and adding excluded values to the equivalent pos of the required elements
# and the remaining elements left 0, the explode could be list or set
explode = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.3,0.5,0.7)
wedges, texts, autotexts = plt.pie(population_list, autopct= '%.1f%%',textprops=dict(color="black"),
                           colors= colors, labels= list(population_df['countery']),explode=explode,labeldistance =1.03)
# plt.legend(wedges, list(df['countery']),
#           title="Countries",
#           loc="center left",
#           bbox_to_anchor=(1, 0, 0.5, 1))
# plt.setp(autotexts, size=10, weight="bold")
plt.title("Population distribution in Rice countries(without Sudan)")
plt.subplot2grid((2,1),(1,0))
available_countries.append("Sudan")
population.append(37737900)
population_df = pd.DataFrame({'countery':available_countries,'population':population}).sort_values(by='population',ascending=False)
population_list = list (population_df['population'])
explode = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.3,0.5,0.7)
wedges, texts, autotexts = plt.pie(population_list, autopct= '%1.1f%%',textprops=dict(color="black"),
                           colors= colors, labels= list(population_df['countery']),explode=explode,labeldistance =1.03)
plt.title("Population distribution in Rice countries (Sudan added)")
print_excluded_countries(excluded_countries,2014)
fig = plt.figure(figsize=(14,9))
ax = sns.barplot(population_df['population'],population_df['countery'], palette="Blues_d")
population_list = list (np.array(population_df['population'])/10**6)
list_counter = 0
# annotating the values
for p in ax.patches:        
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[:,1]
    ax.annotate(str(population_list[list_counter])+" M" , (x.max()+500000, y.mean()), 
                horizontalalignment='left',
                verticalalignment='center')
    list_counter += 1
plt.ylabel("")
plt.title("Rice countries by population [2014]")

print ("The total Population of Rice countries in 2014 according to the available data and with adding Sudan is: {0}".format(np.sum(population_df['population'])))
data.head()
#amis.head()
#data.index
#amis.index
amis.head()
#data.index
#amis.index
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import os
data = pd.read_csv("../input/oecd-crsw-crop-production-since-1990/oecd-CRSW-crop-production.csv", index_col='TIME', parse_dates=True)
amis = pd.read_csv("../input/amis-fao-crop-production-crsw/CRSW-AMIS.csv", index_col='Year', parse_dates=True)

print('data: \nRows: {}\nCols: {}'.format(data.shape[0],data.shape[1]))
print(data.columns)

print('\nAmis data: \nRows: {}\nCols: {}'.format(amis.shape[0],amis.shape[1]))
print(amis.columns)
#data.index
amis.index
data.index
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import os
data = pd.read_csv("../input/oecd-crsw-crop-production-since-1990/oecd-CRSW-crop-production.csv", index_col='SUBJECT')
print('data: \nRows: {}\nCols: {}'.format(data.shape[0],data.shape[1]))
print(data.columns)
data['LOCATION'].unique()
# Production Volume of each Country/Entity
data.Value.replace(['-'],0.0,inplace = True)
data.Value = data.Value.astype(float)
#data.Value = data.Value.astype()
area_list = list(data['LOCATION'].unique())
vol_country_ratio = []
for i in area_list:
    x = data[data['LOCATION']==i]
    vol_country_rate = sum(x.Value)/len(x)
    vol_country_ratio.append(vol_country_rate)
data1 = pd.DataFrame({'area_list': area_list,'vol_country_ratio':vol_country_ratio})
new_index = (data1['vol_country_ratio'].sort_values(ascending=False)).index.values
sorted_data = data1.reindex(new_index)

# visualization
#plt.figure(figsize=(15,10))
plt.figure(figsize=(10,5))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['vol_country_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('Country/Entity')
plt.ylabel('Production Volume')
plt.title('AMIS CRSW Crop production volume ratio per Country/Group')
%matplotlib inline
data.resample('Q').sum().plot();
data['LOCATION'] = pd.to_datetime(data['TIME'],infer_datetime_format=True)
plt.clf()
data['TIME'].map(lambda y: y.year).plot(kind='hist')
plt.show()
%matplotlib inline
#amis.resample('Q').sum().plot();
#amis.columns['']
amis.plot();
data.resample('Q').sum().rolling(4).sum().plot();
data = pd.read_csv('../input/oecd-CRSW-crop-production.csv', index_col='TIME', parse_dates=['TIME'])
#data.index = pd.to_datetime(data.index)
data.index
#data['Total'] = data['Value'] + data['TIME']
#ax = data.resample('Q').sum().rolling(4).sum().plot();
#ax.set_ylim(0, None);
#return data
#data.head ()
#data.plot();
data.groupby(data.index.date).size().plot();
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn')
amis.plot();
data.groupby(data.index.date).mean().plot();
plt.clf()
data.groupby('SUBJECT').size().plot(kind='bar')
plt.show()
plt.clf()
data.groupby('LOCATION').sum().plot(kind='bar')
plt.show()
plt.clf()
data.groupby('SUBJECT').agg('Value').sum().plot(kind='bar')
plt.show()
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.clf()
data.groupby('LOCATION').agg('Value').mean().plot(kind='bar')
plt.show()
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
pivoted = data.pivot_table('Value', index="TIME", columns="LOCATION")
pivoted.iloc[:40, :40]
pivoted.plot(legend=False, alpha=0.5);
pivoted = amis.pivot_table('Value', index="Year", columns="Country/Region Name")
pivoted.iloc[:40, :40]

#data.columns = ['OECD', 'BRICS']
#data['Total'] = data['OECD'] + data['BRICS']
#return data
pivoted.plot(legend=False, alpha=0.5);
pivoted.plot(legend=True, alpha=0.5);
X = pivoted.fillna(0).T.values
X.shape
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
#X2 = PCA(2, svd_solver='full').fit_transform(X)
X2 = PCA(10, svd_solver='full').fit_transform(X)
X2.shape
plt.scatter(X2[:, 0], X2[:, 1]);
gmm = GaussianMixture(2).fit(X)
labels = gmm.predict(X)
plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap='rainbow')
plt.colorbar();
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

pivoted.T[labels == 0].T.plot(legend=False, alpha=0.5, ax=ax[0]);
pivoted.T[labels == 1].T.plot(legend=False, alpha=0.4, ax=ax[1]);

ax[0].set_title('Purple Cluster')
ax[1].set_title('Red Cluster');