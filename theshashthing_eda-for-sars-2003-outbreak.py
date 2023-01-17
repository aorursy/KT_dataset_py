import pandas as pd

import numpy as np
data=pd.read_csv("/kaggle/input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv")
data.shape
data.head()
len(data['Country'].unique())
print(data['Country'].unique().tolist())
countryWise=data.groupby('Country').sum().reset_index()
countryWise
nosCases=countryWise.iloc[:,0:2].copy()
nosCases.head()
nosCases['log(C nos of cases)']=np.log(nosCases['Cumulative number of case(s)'])
nosCases.head()
nosCases.plot.bar(x='Country',y='log(C nos of cases)',rot=90)
deathRec=countryWise.iloc[:,[0,2,3]].copy()
deathRec.head()
deathRec['Deaths(log version)']=deathRec['Number of deaths']+1

deathRec['Deaths(log version)']=np.log(deathRec['Deaths(log version)'])
deathRec['Recovered(log version)']=deathRec['Number recovered']+1

deathRec['Recovered(log version)']=np.log(deathRec['Recovered(log version)'])
d=deathRec.iloc[:,3]

r=deathRec.iloc[:,4]
df=pd.DataFrame({'NosDeaths':d

                 ,'NosRecovered':r

                ,'Country':deathRec.iloc[:,0]})
df.set_index('Country').plot.bar(rot=90)
top10Cases=countryWise.sort_values('Cumulative number of case(s)',ascending=False).head(10).reset_index()

top10Cases.drop(columns=(['index']),inplace=True)

top10Cases
top10Cases['Deaths(log)']=top10Cases['Number of deaths']+1

top10Cases['Deaths(log)']=np.log(top10Cases['Deaths(log)'])

top10Cases['Recovered(log)']=np.log(top10Cases['Number recovered'])
top10Cases
deathRec=top10Cases.iloc[:,[0,4,5]].set_index('Country')
deathRec.plot.bar(rot=90)