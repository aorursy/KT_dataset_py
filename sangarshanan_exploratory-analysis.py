import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/2015_16_Statewise_Elementary.csv')
data.head()
col = ['STATNAME', 'DISTRICTS', 'TOTPOPULAT','SCHTOT','SCHTOTG','SCHTOTGR','SCHTOTPR','ENRTOT','ENRTOTG','ENRTOTGR', 
       'ENRTOTPR', 'TCHTOTG', 'TCHTOTP', 'SCLSTOT', 'STCHTOT', 'ROADTOT', 'SPLAYTOT', 'SWATTOT',  'SELETOT',] 

df = pd.DataFrame(data, columns=col)
df.head()
##correct the wrong value(population of west bengal)
df.loc[18,'TOTPOPULAT'] = df.loc[18,'TOTPOPULAT']/10
plt.figure(figsize=(10,12))
sns.barplot( df['TOTPOPULAT'],df['STATNAME'], alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Population', fontsize=14)
plt.ylabel('States in India', fontsize=14)
plt.title("Population wrt states in India", fontsize=16)
plt.show()
plt.figure(figsize=(10,12))
sns.barplot(df['SCHTOT'], df['STATNAME'],alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Number of Schools', fontsize=14)
plt.ylabel('States in India', fontsize=14)
plt.title("Number of schools wrt states in India", fontsize=16)
plt.show()
plt.figure(figsize=(20,12))
for i in range(1,len(data)):
    plt.subplot(4,9,i)
    plt.title(df['STATNAME'][i])
    top = ['Gov','pri']
    uttar = data.loc[df['STATNAME'] == df['STATNAME'][i],:]
    value =[float(uttar['SCHTOTG']/uttar['SCHTOT'])*100,float(uttar['SCHTOTPR']/uttar['SCHTOT'])*100]
    plt.pie(value, labels=top, autopct='%1.1f%%',startangle=140)
    plt.axis('equal')
plt.show()
plt.figure(figsize=(10,12))
sns.barplot(data['OVERALL_LI'], data['STATNAME'],alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel("Literacy rate", fontsize=16)
plt.title('Literacy rate with respect to state')
plt.show()
data['good'] = data['ROADTOT'] +data['SPLAYTOT'] + data['SWATTOT'] +data['SELETOT']
data['goodpercent'] = data['good']/data['SCHTOT']
plt.figure(figsize=(10,12))
sns.barplot(data['goodpercent'], data['STATNAME'],alpha=1)
plt.xticks(rotation='vertical')
plt.xlabel("Literacy rate", fontsize=16)
plt.title('Literacy rate with respect to state')
plt.show()

