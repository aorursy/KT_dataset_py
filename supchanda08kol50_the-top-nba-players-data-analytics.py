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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
df= pd.read_csv('../input/NBA_player_of_the_week.csv',usecols=['Position','Season short'])
#display(df)
yearArray = dict()
for key,value in df.groupby('Season short'):
    yearArray[key]=value['Position']
for each in list(yearArray.keys()):
    #print(pd.Series(yearArray[each]))
#print(yearArray)
    #print('PDSERIES: ',(pd.Series(yearArray[each])).unique())
    dfPosition = pd.DataFrame((pd.Series(yearArray[each])).unique(),index= (pd.Series(yearArray[each])).unique(),columns=['Positions'])
    dfFreq = pd.DataFrame(pd.value_counts(pd.Series(yearArray[each]))).rename(columns={'Position':'Freq'})
    #display(dfPosition)
    #display(dfFreq)
    dfNew= pd.concat([dfPosition,dfFreq],axis=1,sort=False).sort_values(by=['Freq'],ascending= False).reset_index(drop=True)
    display(dfNew)
    plt.figure(figsize=(16,6))
    sns.barplot(x=dfNew['Positions'],y=dfNew['Freq'],palette = 'ocean')
    plt.xlabel(each,fontsize=18)
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
df= pd.read_csv('../input/NBA_player_of_the_week.csv').head(100)
#display(df)
plt.figure(figsize=(16,6))
sns.stripplot(x=df['Season short'],y=df['Age'],linewidth=1.0)
plt.xlabel('Year',fontsize=18)
plt.ylabel('Ages_Participated_by_an_individual_in_that_year',fontsize=16)
plt.show()
dfAge = pd.DataFrame((df['Age']).unique(),index=(df['Age']).unique(),columns=['Age'])
dfAgeFreq = pd.DataFrame(pd.value_counts(df['Age'])).rename(columns={'Age':'AgeFreq'})
#display(dfAge)
#display(dfAgeFreq)
dfCombined = pd.concat([dfAge,dfAgeFreq],axis=1).sort_values(by=('AgeFreq'),ascending=False).reset_index(drop=True)
#display(dfCombined)
plt.figure(figsize=(16,10))
plt.pie(x=dfCombined['AgeFreq'],labels=dfCombined['Age'],autopct='%.2f')
center_circle = plt.Circle((0,0),0.70,color='black',fc='white', linewidth=1.25)
fig= plt.gcf()
fig.gca().add_artist(center_circle)
plt.legend()
plt.axis('equal')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from IPython.display import display
df= pd.read_csv('../input/NBA_player_of_the_week.csv',usecols=['Player','Seasons in league','Season short']).head(200)
dfOld= df.groupby(['Season short'])
dfSeason = df['Season short'].unique()
df= df.groupby(['Season short','Player'])
k= dict()
SeasonList=[]
for each,value in dfOld:
    for elem,newValue in value.groupby('Player'):
        #print(elem)
        #display(pd.DataFrame(newValue))
        #print(pd.Series(value['Seasons in league']))
        summedValue=np.sum([newValue['Seasons in league'].values],axis=1)
        k[elem] = summedValue
    newDataFrame= pd.DataFrame(list(k.values()),list(k.keys()),columns=['Total_Seasons_Played in ' + str(each)]).sort_values(by='Total_Seasons_Played in ' + str(each),ascending=False)[0:1]
    for each in newDataFrame['Total_Seasons_Played in ' + str(each)].values:
        SeasonList.append(each)
plt.figure(figsize=(16,10))
sns.barplot(x=list(dfSeason),y=SeasonList,palette='gist_rainbow')
plt.xlabel('Year',fontsize=18)
plt.ylabel('Maximum_Season_Participated_by_an_individual_in_that_year',fontsize=16)
plt.show()
plt.figure(figsize=(16,6))
plt.pie(x=SeasonList,labels =list(dfSeason),autopct='%.2f')
x=plt.Circle((0,0),0.70,fc='white',color='yellow',linewidth=0.05)
fig= plt.gcf()
fig.gca().add_artist(x)
plt.legend()
plt.axis('equal')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from IPython.display import display
df= pd.read_csv('../input/NBA_player_of_the_week.csv',usecols=['Age','Season short']).head(200)
plt.figure(figsize=(16,6))
sns.boxplot(x=df['Season short'],y= df['Age'],palette='gist_earth')
plt.xlabel('Year/Season',fontsize=16)
plt.ylabel('Age of Players',fontsize=16)
plt.show()
