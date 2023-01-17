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
df = pd.read_csv('/kaggle/input/all-india-mobile-data-speed-for-july-2020-dataset/All India Mobile Data Speed Measurement for July 2020(TRAI).csv')
df.head()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df.isnull().sum()
df = df.dropna()
a = df.groupby(['Service_provider']).mean()
a
plt.figure(figsize = (10,5))
plt.bar(a['Data_Speed(Kbps)'].index,a['Data_Speed(Kbps)'].values,color = list('rgbymc'), edgecolor = 'black');
plt.title('Average Data speed (in kbps) of \ndifferent networks', fontsize = 20)
#plt.xlabel('networks', fontsize = 15)
plt.ylabel('Data speed',fontsize = 15)
plt.xticks(fontsize = 15)
for i,j in zip(a['Data_Speed(Kbps)'].index,a['Data_Speed(Kbps)'].values):
    plt.text(i,j+70,int(j),fontsize = 15,horizontalalignment='center')
plt.box()
plt.gca().axes.get_yaxis().set_visible(False)
#plt.gca().axes.get_xaxis().set_visible(False)
b = df.copy()
b = b.groupby(['Service_provider','Technology']).mean()
b = b.drop(index = '3G', level = 1)
b.index.droplevel(1)
plt.figure(figsize = (10,5))
plt.bar(b.index.droplevel(1),b['Data_Speed(Kbps)'].values,color = list('rgbymc'), edgecolor = 'black');
plt.title('Average Data speed (in kbps) of \ndifferent 4G networks', fontsize = 20)
#plt.xlabel('networks', fontsize = 15)
plt.ylabel('Data speed',fontsize = 15)
plt.xticks(fontsize = 15)
for i,j in zip(b.index.droplevel(1),b['Data_Speed(Kbps)'].values):
    plt.text(i,j+70,int(j),fontsize = 15,horizontalalignment='center')
plt.box()
plt.gca().axes.get_yaxis().set_visible(False)
#plt.gca().axes.get_xaxis().set_visible(False)
c = df.copy()
c = c.groupby(['Download_Upload','Technology','Service_provider']).mean()
c = c.drop(index = '3G',level = 1)
c
plt.figure(figsize = (15,5))
plt.plot(['AIRTEL','CELLONE','IDEA','JIO','VODAFONE'], c['Data_Speed(Kbps)'][:5].values, color = 'navy', marker = 'o', label = 'Download Speed');
plt.plot(['AIRTEL','CELLONE','IDEA','JIO','VODAFONE'], c['Data_Speed(Kbps)'][5:10].values, color = 'red', marker = 'o', label = 'Upload speed');
plt.legend(loc = 0,fontsize = 15);
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Service provider',fontsize = 15)
plt.title('Downloading speed v/s Uploadidng speed\nof different 4G network',fontsize = 15)
d = df.groupby(['Service_Area','Technology']).mean()
d = d.drop(index = '3G',level = 1)
d = d.reset_index()
d = d.drop(['Technology','Signal_strength'],axis = 1)
d = d.sort_values('Data_Speed(Kbps)',ascending = False)
d = d.drop(14)
d
plt.figure(figsize = (25,8))
plt.bar(d['Service_Area'],d['Data_Speed(Kbps)'],width = 0.5,color = 'blue');
plt.xticks(rotation = 90, fontsize = 20)
plt.yticks(fontsize = 20)
plt.title('Best to worst performing states\nin terms of avg. network speed',fontsize = 25)

e = df.where(df['Service_Area']=='Rajasthan')
e = e.where(e['Technology']=='4G')
e = e.dropna()
e
f = e.groupby(['Service_provider','Technology']).mean()
f
plt.figure(figsize = (6,4))
plt.bar(f.index.droplevel(1),f['Data_Speed(Kbps)'].values,color = ['blue','red','yellow']);
plt.title('Average Data speed (in kbps) of \ndifferent 4G networks in Rajasthan', fontsize = 20)
#plt.xlabel('networks', fontsize = 15)
plt.ylabel('Data speed',fontsize = 15)
plt.xticks(fontsize = 15)
for i,j in zip(f.index.droplevel(1),f['Data_Speed(Kbps)'].values):
    plt.text(i,j+100,int(j),fontsize = 15,horizontalalignment='center')
plt.box()
plt.gca().axes.get_yaxis().set_visible(False)
g = e.groupby(['Download_Upload','Technology','Service_provider']).mean()
g

plt.figure(figsize = (15,5))
plt.plot(['AIRTEL','JIO','VODAFONE'], g['Data_Speed(Kbps)'][:3].values, color = 'red', marker = 'o', label = 'Download Speed');
plt.plot(['AIRTEL','JIO','VODAFONE'], g['Data_Speed(Kbps)'][3:6].values, color = 'navy', marker = 'o', label = 'Upload speed');
plt.legend(loc = 0,fontsize = 15);
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Service provider',fontsize = 15)
plt.title('Downloading speed v/s Uploading speed\nof different 4G network',fontsize = 15)
