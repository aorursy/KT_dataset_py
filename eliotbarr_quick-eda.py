import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as DT
import numpy as np

%matplotlib inline
athletes = pd.read_csv('../input/athletes.csv')
athletes.head()
athletes.dtypes
now = pd.Timestamp(DT.datetime.now())
athletes['dob'] = pd.to_datetime(athletes['dob'])   
athletes['dob'] = athletes['dob'].where(athletes['dob'] < now, athletes['dob'] -  np.timedelta64(100, 'Y'))   
athletes['age'] = (now - athletes['dob']).astype('<m8[Y]')    
athletes.dtypes
fig, axs = plt.subplots(ncols=2)

athletes.groupby("sex").weight.hist(alpha=0.4, ax=axs[1], bins=50)
athletes.groupby("sex").height.hist(alpha=0.4, ax=axs[0], bins=50)
sns.jointplot(athletes['height'], athletes['weight'], kind="kde", color="#4CB391")
fig, axs = plt.subplots(ncols=1)

athletes.groupby("sex").age.hist(alpha=0.4, bins=50)
sns.countplot(x = 'sport', data=athletes)
plt.title('Sports Count')
plt.xticks(rotation=90)
athletes_count = athletes['nationality'].value_counts()
athletes['nationality'].value_counts()[:30]
gold = athletes.groupby('nationality').sum()['gold']
silver = athletes.groupby('nationality').sum()['silver']
bronze = athletes.groupby('nationality').sum()['bronze']

total_medals = gold + silver + bronze
gold_ratio = gold.divide(athletes_count, axis='index')
silver_ratio = silver.divide(athletes_count,axis ='index')
bronze_ratio = bronze.divide(athletes_count,axis ='index')
total_medals_ratio = total_medals.divide(athletes_count,axis = 'index')
gold_ratio.sort(ascending=False)
silver_ratio.sort(ascending=False)
bronze_ratio.sort(ascending=False)
total_medals_ratio.sort(ascending=False)
gold_ratio.head()
silver_ratio.head()
bronze_ratio.head()
total_medals_ratio.head()