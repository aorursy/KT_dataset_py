import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_apps = pd.read_csv('../input/googleplaystore.csv')
data_reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')
data_reviews.head()
data_apps.head()
for i in range(22):
    print(data_apps.Installs.values[i])
category_list = list(data_apps.groupby(['Category']).groups.keys())
category_list 
len(category_list)
data_apps.sort_values(by=['Rating'],ascending=False)
Install_list = list(data_apps.groupby(['Installs']).groups.keys())
n_count=[]
for str in Install_list:
    n_count.append(data_apps[(data_apps['Rating']==5.0) & (data_apps['Installs']==str)].shape[0])
plt.bar(np.arange(len(Install_list)),np.array(n_count))
plt.xticks(np.arange(len(Install_list)), Install_list,rotation='vertical')
plt.show()
Install_list = list(data_apps.groupby(['Installs']).groups.keys())
len(Install_list)
print(Install_list)
n_count=[]
for str in Install_list:
    n_count.append(data_apps[(data_apps['Category']=='ART_AND_DESIGN') & (data_apps['Installs']==str)].shape[0])
plt.bar(np.arange(len(Install_list)),np.array(n_count))
plt.xticks(np.arange(len(Install_list)), Install_list,rotation='vertical')
plt.show()
n_count=[]
for str in Install_list:
    n_count.append(data_apps[(data_apps['Category']=='WEATHER') & (data_apps['Installs']==str)].shape[0])
plt.bar(np.arange(len(Install_list)),np.array(n_count))
plt.xticks(np.arange(len(Install_list)), Install_list,rotation='vertical')
plt.show()
n_count=[]
for str in Install_list:
    n_count.append(data_apps[(data_apps['Category']=='VIDEO_PLAYERS') & (data_apps['Installs']==str)].shape[0])
plt.bar(np.arange(len(Install_list)),np.array(n_count))
plt.xticks(np.arange(len(Install_list)), Install_list,rotation='vertical')
plt.show()
top_apps = data_apps[data_apps.Installs == '1,000,000,000+']
top_apps.sort_values(['Rating'],ascending=False)
top_apps['Reviews'] = top_apps['Reviews'].astype(int)
top_apps.sort_values(['Reviews'],ascending=False)
plt.hist(top_apps.Rating)
top_apps.shape


