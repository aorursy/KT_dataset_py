import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
d=pd.read_csv("../input/Production by product.csv")
df1 = d.drop(d.columns[[0,16,24]],axis=1)
df1

plt.figure(figsize=(30, 30))
plt.subplot(2, 3, 1)
plt.xticks(())
plt.yticks(())
plt.plot(df1["Other market services"], color='darkred',label=" Other market services")
plt.plot(df1["Hotel and restaurant services"], color='green',label="Hotel and restaurant services")
plt.plot(df1["Non-market services"], color='grey',label="Non-market services")
plt.legend(loc='upper left', prop={'size': 20})

plt.subplot(2, 3, 2)
plt.xticks(())
plt.yticks(())
plt.plot(df1["Building materials, ceramics and glass"], color='green',label="Building materials, ceramics and glass")
plt.plot(df1["Maintenance and repair"], color='grey',label="Maintenance and repair")
plt.plot(df1["Mechanical and electrical industries"], color='black',label="Mechanical and electrical industries")
plt.plot(df1["Building and civil engineering"], color='darkred',label="Building and civil engineering")

plt.legend(loc='upper left', prop={'size': 20})


plt.subplot(2, 3, 3)
plt.xticks(())
plt.yticks(())
plt.plot(df1["Textile, Clothing and Leather"], color='grey',label="Textile, Clothing and Leather")
plt.legend(loc='upper left', prop={'size': 20})

plt.subplot(2, 3, 4)
plt.xticks(())
plt.yticks(())
plt.plot(df1["Tobacco Industry"], color='green',label=" Tobacco Industry")
plt.plot(df1["Agro-food Industry"], color='darkred',label="Agro-food Industry")
plt.plot(df1["Various industries"], color='black',label=" Various industries")
plt.plot(df1["Agriculture and Fisheries"], color='grey',label="Agriculture and Fisheries")
plt.plot(df1["Water"], color='magenta',label="Water")
plt.legend(loc='upper left', prop={'size': 20})

plt.subplot(2, 3, 5)
plt.xticks(())
plt.yticks(())
plt.plot(df1["Financial Services"], color='darkred',label="Financial Services")
plt.plot(df1["Transport"], color='green',label="Transport")
plt.plot(df1["Post and telecommunications"], color='grey',label="Post and telecommunications")
plt.legend(loc='upper left', prop={'size': 20})

plt.subplot(2, 3, 6)
plt.xticks(())
plt.yticks(())
plt.plot(df1["Oil refining"], color='grey',label=" Oil refining")
plt.plot(df1["Chemical industries"], color='darkred',label="Chemical industries")
plt.plot(df1["Mines"], color='green',label="Mines")
plt.legend(loc='upper left', prop={'size': 20})       

plt.tight_layout()
plt.show()
np.corrcoef(df1)
plt.matshow(df1.corr())
plt.xticks(range(len(df1.columns)), df1.columns,rotation=90)
plt.yticks(range(len(df1.columns)), df1.columns)
plt.colorbar()
plt.show()
  

max=df1.loc[df1["Maintenance and repair"].idxmax()]
max=pd.DataFrame(max)
max
dfM = pd.DataFrame(index=range(18), columns=range(22))
dfM = dfM.fillna(0)

for j in range(22):
    for i in range(18):
        dfM.iloc[i,j]=(df1.iloc[i,j]/max.iloc[i,0])*100
dfM.columns=['Agriculture and Fisheries', 'Agro-food Industry', 'Tobacco Industry',
       'Textile, Clothing and Leather', 'Various industries', 'Oil refining',
       'Chemical industries', 'Building materials, ceramics and glass',
       'Mechanical and electrical industries', 'Oil and natural gas', 'Mines',
       'Electricity and Gas', 'Water', 'Building and civil engineering',
       'Maintenance and repair', 'Hotel and restaurant services', 'Transport',
       'Post and telecommunications', 'Financial Services',
       'Other market services', 'Non-market services',
       'Territorial Correction']
dfM
plt.figure(figsize=(30, 30))
plt.subplot(2, 3, 1)
plt.xticks(())
plt.yticks(())
plt.plot(dfM["Other market services"], color='darkred',label=" Other market services")
plt.plot(dfM["Hotel and restaurant services"], color='green',label="Hotel and restaurant services")
plt.plot(dfM["Non-market services"], color='grey',label="Non-market services")
plt.legend(loc='upper left', prop={'size': 20})

plt.subplot(2, 3, 2)
plt.xticks(())
plt.yticks(())
plt.plot(dfM["Building materials, ceramics and glass"], color='green',label="Building materials, ceramics and glass")
plt.plot(dfM["Maintenance and repair"], color='grey',label="Maintenance and repair")
plt.plot(dfM["Mechanical and electrical industries"], color='black',label="Mechanical and electrical industries")
plt.plot(dfM["Building and civil engineering"], color='darkred',label="Building and civil engineering")

plt.legend(loc='upper left', prop={'size': 20})


plt.subplot(2, 3, 3)
plt.xticks(())
plt.yticks(())
plt.plot(dfM["Textile, Clothing and Leather"], color='grey',label="Textile, Clothing and Leather")
plt.legend(loc='upper left', prop={'size': 20})

plt.subplot(2, 3, 4)
plt.xticks(())
plt.yticks(())
plt.plot(dfM["Tobacco Industry"], color='green',label=" Tobacco Industry")
plt.plot(dfM["Agro-food Industry"], color='darkred',label="Agro-food Industry")
plt.plot(dfM["Various industries"], color='black',label=" Various industries")
plt.plot(dfM["Agriculture and Fisheries"], color='grey',label="Agriculture and Fisheries")
plt.plot(dfM["Water"], color='magenta',label="Water")
plt.legend(loc='upper left', prop={'size': 20})

plt.subplot(2, 3, 5)
plt.xticks(())
plt.yticks(())
plt.plot(dfM["Financial Services"], color='darkred',label="Financial Services")
plt.plot(dfM["Transport"], color='green',label="Transport")
plt.plot(dfM["Post and telecommunications"], color='grey',label="Post and telecommunications")
plt.legend(loc='upper left', prop={'size': 20})

plt.subplot(2, 3, 6)
plt.xticks(())
plt.yticks(())
plt.plot(dfM["Oil refining"], color='grey',label=" Oil refining")
plt.plot(dfM["Chemical industries"], color='darkred',label="Chemical industries")
plt.plot(dfM["Mines"], color='green',label="Mines")
plt.legend(loc='upper left', prop={'size': 20})       

plt.tight_layout()
plt.show()
df1 = StandardScaler().fit_transform(df1)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df1)
principalComponents
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf 
plt.subplot(1,1,1)
plt.xticks(())
plt.yticks(())
plt.plot(principalDf["principal component 1"], color='grey',label="principal component 1")
plt.plot(principalDf["principal component 2"], color='green',label="principal component 2")
plt.legend(loc='upper left', prop={'size': 20})
