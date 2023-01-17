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
df= pd.read_csv('../input/records-for-2011.csv').fillna(0).head(180000)
#df = pd.concat([df['Location'],df['Incident Type Description']],axis=1)
df= pd.DataFrame(pd.Series(list(zip(df['Location'],df['Incident Type Description']))),columns=['LocWithIncident']).sort_values(by=['LocWithIncident'],ascending=True)
df.reset_index(drop=True,inplace=True)
#display(df)

dfNew= df.groupby('LocWithIncident')

locAccidentSize= np.array([])
for key,value in dfNew:
    locAccidentSize= np.append(locAccidentSize,value.size)
df.drop_duplicates(keep='first',inplace=True)
df.reset_index(drop=True,inplace=True)
locAccidentSize = pd.DataFrame(pd.Series(locAccidentSize),columns=['locAccidentSize'])
dfNewest= pd.concat([df['LocWithIncident'],locAccidentSize['locAccidentSize']],axis=1).sort_values(by=['locAccidentSize'],ascending= False)
#display(dfNewest[0:10])
plt.figure(figsize=(16,10))
sns.barplot(y=dfNewest['LocWithIncident'][0:10],x=dfNewest['locAccidentSize'][0:10] , palette='spring')
plt.show()
plt.figure(figsize=(10,8))
plt.pie(dfNewest['locAccidentSize'][0:10],labels=dfNewest['LocWithIncident'][0:10],explode=(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1),autopct = '%.2f',shadow=True)
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
df= pd.read_csv('../input/records-for-2011.csv').fillna(0).head(100)
dfAccType = pd.DataFrame(df['Incident Type Description'])
#display(dfAccType)
dfGrouped = dfAccType.groupby('Incident Type Description')
listArray  = np.array([])
listNames = pd.DataFrame(dfAccType['Incident Type Description'].unique(),index=list(dfAccType['Incident Type Description'].unique()),columns=['Incident Type Description'])
#display(listNames)
listSize= pd.DataFrame(pd.value_counts(dfAccType['Incident Type Description'])).rename(columns={'Incident Type Description':'Incident Count'})
#display(listSize)
'''for key,group in dfGrouped:
    np.append(listNames, key)
    np.append(listArray,group.size)
print(listArray)'''
dfNewest = pd.concat([listNames,listSize],axis=1,sort=True).sort_values(by=('Incident Count'),ascending=False).reset_index(drop=True,inplace=False)
display(dfNewest[0:10])
plt.figure(figsize=(16,6))
sns.barplot(y=dfNewest['Incident Type Description'][0:10],x=dfNewest['Incident Count'][0:10],palette='autumn')
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv('../input/records-for-2011.csv',usecols=['Location']).fillna(0).head(10)
df=df.sort_values(by=('Location'),ascending=True)
#display(df)
dfFreq= pd.DataFrame(pd.value_counts(df['Location'])).reset_index(drop=True)
#display(dfFreq)
dfName= pd.DataFrame((df['Location']).unique(),columns=['LocName'])
#display(dfName)
dpNew= pd.concat([dfName,dfFreq],axis=1)
#display(dpNew)
plt.figure(figsize=(16,8))
sns.barplot(x=dpNew['LocName'],y=dpNew['Location'],palette='gist_heat')
plt.xlabel('Location Name',fontsize=16)
plt.ylabel('Crime Frequency In Location ',fontsize=16)
plt.show()
