import pandas as pd
import numpy as np
# Change the path to the dataset file if needed. 
PATH = '../input/athlete_events.csv'
data = pd.read_csv(PATH)
data.head()
data.shape
data.info()
data.describe(include="all",percentiles=[0.01,0.05,0.1,0.25,0.5,0.9,0.95,0.99])
import missingno
missingno.matrix(data)
data.shape
data.drop_duplicates().shape
data[data.Year == 1996].groupby(by=['Year','Season'],as_index=False)['ID'].count()
data[data.Year == 1996].groupby(by=['Sex'],as_index=False)['Age'].min()

columns_person = ['ID','Name','Sex','Age','Height','Weight','NOC',"Sport"]
data_new = data.loc[np.logical_and(data.Year == 2000,data.Sex == 'M'), columns_person].drop_duplicates()
data_new.head()
list_sport = data.Sport.unique()
list_sport_Gymnastics = [sport for sport in list_sport if ('gymnastic' in sport.lower())]
list_sport_Gymnastics
print("Validating unique sportsmen: ")
print(data_new.shape[0])
print(data_new.ID.count())
data_new["flg_Sgymnastic"] = data_new["Sport"].isin(list_sport_Gymnastics).astype(int)
(data_new.flg_Sgymnastic.value_counts(normalize=True)*100).round(1)

columns_person = ['ID','Name','Sex','Age','Height','Weight','NOC',"Sport"]
data_new = data.loc[(data.Year == 2000)&(data.Sex == 'F')&(data.Sport == 'Basketball'), columns_person].drop_duplicates()
data_new.head()
data_new.Height.describe().round(1)

data[data.Year == 2002].groupby(by=['Year','Season'],as_index=False)['ID'].count()
max_Weight = data[data.Year == 2002]['Weight'].max()
data.loc[(data.Year == 2002)&(data.Weight == max_Weight),:]
data.loc[(data.Year == 2002)&(data.Weight == max_Weight),:].Sport

data.loc[data.Name == 'Pawe Abratkiewicz'].Year.nunique()

data[data.Year == 2000].groupby(by=['Year','Season'],as_index=False)['ID'].count()
data[(data.Year == 2000)&(data.Sport == 'Tennis')&(data['NOC'].isin(['AUS','ANZ']))].groupby(by=['Team','Medal'], as_index = False)['ID'].count()
data.head(2)
def n_medal(country): 
    print(country,':',data[(data.Year==2016)&(data.Team==country)&(data.Medal.notnull())]['Medal'].count(),"medals")
n_medal('Switzerland')
n_medal('Serbia')
data.head(2)
data[(data.Year==2014)]["Age"].describe()
# Considerando duplicados
pd.cut(data.loc[(data.Year==2014),['ID','Age']]['Age'],bins = 4,duplicates='drop',right=False).value_counts()
# Real: # Sin considerar duplicados
pd.cut(data.loc[(data.Year==2014),['ID','Age']].drop_duplicates()['Age'],bins = 4,duplicates='drop',right=False).value_counts()
data.head(2)
def Olympics(city): 
    return(data.loc[(data.City==city),['City','Year','Season']].drop_duplicates())
Olympics('Lake Placid')
Olympics('Sankt Moritz')

data.groupby(by=['Year','Season'],as_index=False)['ID'].count()
(data[data.Year == 2016]).groupby(by=['Year','Season'],as_index=False)['Sport'].nunique()
data[data.Year == 1995].groupby(by=['Year','Season'],as_index=True)['Sport'].nunique()
