# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
dataevents = pd.read_csv("../input/athlete_events.csv")
dataregions = pd.read_csv("../input/noc_regions.csv")
dataevents.head()
dataevents.info()
dataregions.head()
dataevents.Sex.unique()
dataregions.info()
#list(dataevents.Name.unique())
dataevents[dataevents['Year']==1996].groupby(["Sex"])["Age"].min()
def fdistribucion(base,decimales):
    cont = 0
    for i in lista_variables:
        cont = cont + 1
        print("------------------------------------------------------")
        print(str(cont),". Var: ",i, sep = "")
        print(pd.concat([pd.DataFrame(base[:][i].value_counts(dropna = False).index, columns = ['Atributo']),
               pd.DataFrame(base[:][i].value_counts(dropna = False).values, columns = ['Cantidad']),
               pd.DataFrame(np.round(100*base[:][i].value_counts(dropna = False).values/len(base),decimales), columns = ['%Total'])], axis = 1))
lista_variables = ['Sport']
fdistribucion(dataevents[(dataevents['Year'] ==2000) & (dataevents['Sex'] == 'M')].drop_duplicates(subset=['Name']),1)
devent2=dataevents[(dataevents['Year']==2000) & (dataevents['Sex']=='F')  & (dataevents['Sport']=='Basketball') ]
np.round(devent2.groupby(["Sport"])["Height"].mean())
np.round(devent2.groupby(["Sport"])["Height"].std(),1)
dataevents[(dataevents['Year'] == 2002)]['Weight'].max()
dataevents[(dataevents['Year'] == 2002) & (dataevents['Weight'] == 123.0)]['Sport']
dataevents.Sport.unique()
devent5=dataevents[(dataevents['Name']=='Pawe Abratkiewicz')  ]
devent5.Year.unique()
devent6=dataevents[(dataevents['Year']==2000) & (dataevents['Team']=='Australia') &  (dataevents['Sport']=='Tennis') &  (dataevents['Medal']=='Silver')]
print('cantidad de medallas: ',devent6['Medal'].value_counts())
devent6.head()
devent7=dataevents[(dataevents['Year']==2016) & (dataevents['Team']=='Switzerland') ] #
devent7.Medal.count()
devent7=dataevents[(dataevents['Year']==2016) & (dataevents['Team']=='Serbia') ] #
devent7.Medal.count()
bins = [15, 25, 35,45,55] 
dataevents['age_bin'] = pd.cut(dataevents['Age'], bins = bins,right = True)
dataevents[(dataevents['Year'] ==2014) ].agg(['min','max'])['Age']
list(dataevents.age_bin.unique())
p4=dataevents[(dataevents['Year'] ==2014) ]
p4.groupby(["age_bin"])["Age"].count()
lista_variables= ['Age']
fdistribucion(dataevents[(dataevents['Year'] ==2014) ],1)
dataevents[(dataevents['Season']=='Summer') & (dataevents['City']=='Lake Placid') ].City.unique()
dataevents[(dataevents['Season']=='Winter') & (dataevents['City']=='Sankt Moritz')  ].City.unique()
#& (dataevents['City']=='Lake Placid')
dataevents.City.unique()
a=dataevents[(dataevents['Year']==1995) ].Sport.unique()
a=pd.DataFrame(a)
a=a.count()
d=dataevents[(dataevents['Year']==2016) ].Sport.unique()
d=pd.DataFrame(d)
d=d.count()
np.abs(a-d)