import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython.core.display import display, HTML
plt.style.use('default')
display(HTML("<style>.container { width:90% !important; }</style>"))
df = pd.read_csv('../input/events.csv',low_memory=False)
#convieto las fechas a formato fecha 
df['timestamp']= df['timestamp'].astype('datetime64')
visitsplatform = df.groupby(['device_type'])['event'].value_counts().reset_index(drop= True, level = 1)
visitsplatform.sort_values(ascending = False,inplace= True)
visitsplatform = visitsplatform.to_frame().reset_index()
visitsplatform
fig, ax = plt.subplots(1,2,figsize=(10, 2))
ax[0].barh(visitsplatform['device_type'],visitsplatform['event'], color = 'rgb',edgecolor='black')
ax[0].set(xlabel='Total de visitas', ylabel='Dispositivo', title='Dispositivos usados para ingresar al sitio')
ax[1].barh(visitsplatform['device_type'],visitsplatform['event']/visitsplatform['event'].sum()*100, color = 'rgb',edgecolor='black')
ax[1].set(xlabel='% Total de visitas', ylabel='Dispositivo', title='Dispositivos usados para ingresar al sitio')
plt.tight_layout()

visitsos= df.groupby(['device_type','operating_system_version'])['event'].value_counts()
visitsos = visitsos.reset_index(drop = True, level = 2).reset_index()
computeros = visitsos.loc[visitsos['device_type']=='Computer']
tabletos = visitsos.loc[visitsos['device_type']=='Tablet']
smartphoneos = visitsos.loc[visitsos['device_type']=='Smartphone']
OS = computeros.groupby(['operating_system_version'])['event'].sum()
tOS = tabletos.groupby(['operating_system_version'])['event'].sum()
sOS = smartphoneos.groupby(['operating_system_version'])['event'].sum()


Linux = OS.filter(like = 'Linux').sum()  + OS.filter(like = 'Ubuntu').sum()
Mac = OS.filter(like = 'Mac').sum()
Windows = OS.filter(like = 'Windows').sum()

oses = pd.DataFrame(data = {'cant' :[Linux, Mac, Windows]}, index = ['Linux', 'Mac', 'Windows'])

oses
fig, ax = plt.subplots(1,2,figsize=(10, 2))
ax[0].barh(oses.index,oses['cant'], color = 'rgb',edgecolor='black')
ax[0].set(xlabel='Total de visitas', ylabel='Sistema Operativo', title='Sistemas Operativo usados (computadora)')
ax[1].barh(oses.index,oses['cant']/oses['cant'].sum()*100, color = 'rgb',edgecolor='black')
ax[1].set(xlabel='% Total de visitas', ylabel='Sistema Operativo', title='Sistemas Operativo usados (computadora)')
plt.tight_layout()
iOS = tOS.filter(like = 'iOS').sum() 
Android = tOS.filter(like = 'Android').sum()
toses = pd.DataFrame(data = {'cant' :[iOS, Android]}, index = ['iOS', 'Android'])
toses
fig, ax = plt.subplots(1,2,figsize=(10, 2))
ax[0].barh(toses.index,toses['cant'], color = 'rgb',edgecolor='black')
ax[0].set(xlabel='Total de visitas', ylabel='Sistema Operativo', title='Sistemas Operativo usados (tablet)')
ax[1].barh(toses.index,toses['cant']/toses['cant'].sum()*100, color = 'rgb',edgecolor='black')
ax[1].set(xlabel='% Total de visitas', ylabel='Sistema Operativo', title='Sistemas Operativo usados (tablet)')
plt.tight_layout()
sOS.reset_index()['operating_system_version'].unique()
Android = sOS.filter(like = 'Android').sum()  + OS.filter(like = 'Ubuntu').sum()
iOS = sOS.filter(like = 'iOS').sum()
Windows = sOS.filter(like = 'Windows').sum()
Other = sOS.filter(like = 'Other').sum()

soses = pd.DataFrame(data = {'cant' :[Android, iOS, Windows, Other]}, index = ['Android', 'iOS', 'Windows','Other'])

soses
fig, ax = plt.subplots(1,2,figsize=(10, 2))
ax[0].barh(soses.index,soses['cant'], color = 'rgb',edgecolor='black')
ax[0].set(xlabel='Total de visitas', ylabel='Sistema Operativo', title='Sistemas Operativo usados (smartphone)')
ax[1].barh(soses.index,soses['cant']/soses['cant'].sum()*100, color = 'rgb',edgecolor='black')
ax[1].set(xlabel='% Total de visitas', ylabel='Sistema Operativo', title='Sistemas Operativo usados (smartphone)')
plt.tight_layout()
visitsbrowser= df.groupby(['browser_version'])['event'].value_counts()
visitsbrowser.sort_values(ascending = False,inplace= True)
visitsbrowser.to_frame().reset_index(drop=True,level = 1).reset_index().head(10)
visitsscreen= df.groupby(['screen_resolution','device_type' ])['event'].value_counts()
visitsscreen.sort_values(ascending = False,inplace= True)
screens = visitsscreen.to_frame().reset_index(drop=True, level = 2).reset_index()
screens.loc[screens['device_type']=='Computer'].head(10).reset_index(drop = True)
screens.loc[screens['device_type']=='Smartphone'].head(10).reset_index(drop = True)
screens.loc[screens['device_type']=='Tablet'].head(10).reset_index(drop = True)