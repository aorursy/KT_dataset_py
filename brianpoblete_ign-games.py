import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



dataset = pd.read_csv("../input/ign.csv")



%matplotlib inline

import seaborn

seaborn.set() 

platform_data = dataset[dataset['release_year']>=2011]['platform']

ax = platform_data.value_counts().plot(kind='bar')

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.title('Platforms with the most games reviewed from 2011')

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.xlabel('Number of games reviewed')

plt.ylabel('Platform name')

plt.show()

genre_data = dataset[dataset['release_year'] == 2016]['genre']

temp = genre_data.value_counts()

print("Number of games reviewed in 2016:",dataset[dataset['release_year']==2016]['title'].count())

genre_data_top10 = temp.head(10)

genre_data_top10['remaining {0} items'.format(len(temp)-10)] = sum(temp[10:])

genre_data_top10.plot(kind='pie')

plt.title('Most common genres in 2016')

plt.pie(genre_data_top10,autopct='%1.0f%%')

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.show()
genre_data = dataset[dataset['release_year'] == 2006]['genre']

print("Number of games reviewed in 2006:",dataset[dataset['release_year']==2006]['title'].count())

temp = genre_data.value_counts()

genre_data_top10 = temp.head(10)

genre_data_top10['remaining {0} items'.format(len(temp)-10)] = sum(temp[10:])

genre_data_top10.plot(kind='pie')

plt.title('Most common genres in 2006')

plt.pie(genre_data_top10,autopct='%1.0f%%')

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.show()
ec_year = dataset[dataset['editors_choice']=='Y']['release_year'].value_counts()

df_ecyear = pd.DataFrame([ec_year])

df_ecyear.index = [""]

ax2 = df_ecyear.plot(kind='bar',stacked=False, figsize=(10,6), title="Number of Editor's Choice games by Year")

for p in ax2.patches:

    ax2.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))





from IPython.display import display

display(df_ecyear)