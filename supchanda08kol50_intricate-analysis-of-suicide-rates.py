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
from IPython.display import  display
df= pd.read_csv('../input/who_suicide_statistics.csv').fillna(0)
dfNew= df.groupby('year')

for e,k in dfNew:
    yearList= [(str(e)+':'+ str(i)) for i in range(6)]
    yearDT=pd.DataFrame(yearList,columns=['Year'])
    suicidecount=np.array([])
    #display(k)
    i=0
    ageUnique=pd.DataFrame(k['age'].unique(),columns=['Age'])
    #display(ageUnique)
    dfSuicideAge = k.groupby('age')
    for elem,value in dfSuicideAge:
        suicideArray = 0
        #display(value)
        for each in value['suicides_no']:
            #print(int(each))
            suicideArray+=int(each)
            #print('suicideArray:',suicideArray)
        suicidecount= np.append(suicidecount,suicideArray)
        sCountDFrame = pd.DataFrame(list(suicidecount),columns=['SuicideCount']).reset_index(drop=True)
    concatPd = pd.concat([yearDT,ageUnique,sCountDFrame],axis=1)
    display(concatPd)
    plt.figure(figsize=(16,6))
    pivotSucide = concatPd.pivot('Age','Year','SuicideCount')
    sns.heatmap(pivotSucide,annot=True,cmap='RdYlGn',cbar=False,linewidth=0.06,fmt='.2f')
    plt.title('Total Suicide Count in every year w.r.t. Age',fontsize=15)
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import  display
df= pd.read_csv('../input/who_suicide_statistics.csv',usecols=['sex'])
dfSex= pd.DataFrame(df['sex'].unique(),columns=['Gender'],index=df['sex'].unique())
dfCount = pd.DataFrame(pd.value_counts(df['sex']))
#display(dfSex)
#display(dfCount)
dpConcat= pd.concat([dfSex,dfCount],axis=1,sort=True)
display(dpConcat)

plt.figure(figsize=(16,10))
plt.pie(dfCount['sex'],labels=list(df['sex'].unique()),autopct='%.2f',explode=(0.05,0.05))
circlePlotted = plt.Circle((0,0),0.60,fc='white',linewidth=0.05)
fig= plt.gcf()
plt.gca().add_artist(circlePlotted)
plt.axis('equal')
plt.title('Suicide Count w.r.t. Sex',fontsize=18)
plt.legend()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
df= pd.read_csv('../input/who_suicide_statistics.csv',usecols=['country','suicides_no']).fillna(0)
dfcountry = df.groupby('country')
pdConcatOverall=pd.DataFrame()
for each,value in dfcountry:
    k= pd.DataFrame(pd.Series(int(np.sum(value['suicides_no']))),columns=['Total_Suicide_Count'])
    #display(k)
    country = pd.DataFrame(pd.Series(str(each)),columns=['Country'])
    #display(country)
    pdConcat= pd.concat([country,k],axis=1)
    #display(pdConcat)
    pdConcatOverall =pd.concat([pdConcatOverall,pdConcat],ignore_index=True)
#print(type(pdConcatOverall))
pdConcatOverall.sort_values(by=['Total_Suicide_Count'],ascending=False,inplace=True)
#display(pdConcatOverall)
plt.figure(figsize=(20,15))
plt.pie(x=pdConcatOverall['Total_Suicide_Count'][0:10],labels=pdConcatOverall['Country'][0:10],autopct='%.1f')
CRKle = plt.Circle((0,0),0.65,fc='white',linewidth=1.5,color='black')
x=plt.gcf()
x.gca().add_artist(CRKle)
plt.legend()
plt.axis('equal')
plt.title('Top 10 Country having Suicide rates\' percentage',fontsize=20)
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import  display
df= pd.read_csv('../input/who_suicide_statistics.csv').fillna(0)
dfNew= df.groupby('year')
overallConcatPd=pd.DataFrame()
for e,k in dfNew:
    yearListTotal= [str(e) for i in range(6)]
    yearDTNew=pd.DataFrame(yearListTotal,columns=['YearName'])
    suicidecount=np.array([])
    #display(k)
    i=0
    ageUnique=pd.DataFrame(k['age'].unique(),columns=['Age'])
    #display(ageUnique)
    dfSuicideAge = k.groupby('age')
    for elem,value in dfSuicideAge:
        suicideArray = 0
        #display(value)
        for each in value['suicides_no']:
            #print(int(each))
            suicideArray+=int(each)
            #print('suicideArray:',suicideArray)
        suicidecount= np.append(suicidecount,suicideArray)
        sCountDFrame = pd.DataFrame(list(suicidecount),columns=['SuicideCount']).reset_index(drop=True)
    concatPd = pd.concat([yearDTNew,ageUnique,sCountDFrame],axis=1)
    overallConcatPd = pd.concat([overallConcatPd,concatPd],axis=0)
plt.figure(figsize=(33,20))
pivotSuicideTotal = overallConcatPd.pivot('Age','YearName','SuicideCount')
sns.heatmap(pivotSuicideTotal,annot=True,cmap='RdYlGn',cbar=False,linewidth=2.0,fmt='.0f')
plt.xlabel('Year',fontsize=20)
plt.ylabel('Age',fontsize=20)
plt.title('Total Suicide Count w.r.t. Age',fontsize=30)
plt.show()
