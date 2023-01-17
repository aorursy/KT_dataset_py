import pandas as pd
df = pd.read_csv("../input/app-store-apple-data-set-10k-apps/AppleStore.csv")
df.head()

df_rating_not_0 = df.loc[(df['user_rating_ver'] != 0)]
genre_mean = df_rating_not_0.groupby(['prime_genre']).mean()['user_rating_ver'] 

genre_mean = round(genre_mean.sort_values(ascending=False),2) 

df_genre_mean = pd.DataFrame(genre_mean)
df_genre_mean


#The .sort_values(ascending=False) command is to sort the result in descending order, based in the mean value

#The round(xxxx,2) command is to round the result in 2 decimal places
from matplotlib import pyplot as plt

df_lifestyle = df_rating_not_0.loc[(df_rating_not_0['prime_genre'] == 'Lifestyle')]
df_lifestyle['user_rating_ver'].hist()
plt.title('Lifestyle apps histogram')
plt.xlabel('User rating')
plt.ylabel('Frequency (n)')
plt.show()
values = []
for i in range(1000):
    selected_sample = df_rating_not_0.loc[(df_rating_not_0['prime_genre'] == 'Lifestyle')]['user_rating_ver'].sample(n=50, replace=True)
    mean_sample = selected_sample.mean()
    values += [mean_sample]

plt.hist(values)
plt.title('Lifestyle apps histogram')
plt.xlabel('User rating')
plt.ylabel('Frequency (n)')
plt.show() 

import numpy as np
genres = df.prime_genre.unique()
genre_mean = []

for names in genres:
    values = []
    for i in range(1000):
        selected_sample = df_rating_not_0.loc[(df_rating_not_0['prime_genre'] == names)]['user_rating_ver'].sample(n=50, replace=True)
        mean_sample = selected_sample.mean()
        values += [mean_sample]
        
    total_mean = sum(values)/len(values) 
    SD = np.std(values)
    genre_mean += [[names,total_mean,SD]]
    
    
def getKey(item):
    return item[1]
genre_mean = sorted(genre_mean, key=getKey, reverse=True)
rank_num = 1
rank = [['Book',1]]
for i in range(1,len(genre_mean)):
    SE = ((((genre_mean[i-1][2])**2)/1000)+(((genre_mean[i][2])**2)/1000))**0.5
    D = SE*1.96    #I'm working with the interval of 95% of confidence
    M = genre_mean[i-1][1]-genre_mean[i][1]
    if M > D:
        rank += [[genre_mean[i][0],rank_num]]
    else:
        rank_num = rank_num + 1
        rank += [[genre_mean[i][0],rank_num]]
df_rank = pd.DataFrame(rank, columns =['Genre', 'Rank'])
df_rank
df_lifestyle = df_rating_not_0.loc[(df_rating_not_0['prime_genre'] == 'Lifestyle')]
df_lifestyle['user_rating_ver'].hist()
plt.title('Lifestyle apps histogram')
plt.xlabel('User rating')
plt.ylabel('Frequency (n)')
plt.show()
df_games = df_rating_not_0.loc[(df_rating_not_0['prime_genre'] == 'Games')]
df_games['user_rating_ver'].hist()
plt.title('Games apps histogram')
plt.xlabel('User rating')
plt.show()
genres = df.prime_genre.unique()
probability_genres = []
for names in genres:
    df_selected = df_rating_not_0.loc[(df_rating_not_0['prime_genre'] == names)]['user_rating_ver']
    probability_genres += [round((sum(df_selected[df_selected>=4].value_counts())/len(df_selected))*100,1)]
frame = {'Genre': genres,'Probability %': probability_genres}
df_probability = pd.DataFrame(frame)
df_probability = df_probability.sort_values(['Probability %'],ascending=[True])
df_probability
df_google = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
#Droping NaN values
df_google = df_google.dropna()

#Converting reviews to numeric
df_google['Reviews'] = pd.to_numeric(df_google['Reviews'], errors='coerce')

#From 'Installs' column, I'm going to remove the '+' and convert it to numeric
df_google['Installs'] = df_google['Installs'].str.replace('+','')
df_google['Installs'] = df_google['Installs'].str.replace(',','')
df_google['Installs'] = pd.to_numeric(df_google['Installs'], errors='coerce')

#Selecting just the information that it is now needed for me
df_google_select = df_google[['Category','Rating','Reviews','Installs']]

#Creating a new column 'Review_rate'. It is the relationship(ratio) between the number of
# user reviewn and the number if downloads
df_google_select['Review_rate'] = df_google_select['Reviews']/df_google_select['Installs']


#Finally, organizing the information by number of downloads.
installs_bins = df_google_select.groupby(['Installs']).mean()
installs_bins['Review_rate'] = round(installs_bins['Review_rate']*100,2)

#and printing just the information for more than 5000 downloads
df_more_5000 = installs_bins.iloc[7: ,2: ]
df_more_5000
import numpy as np
bars = (5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000,500000000,1000000000)
y_pos = np.arange(len(df_more_5000))
plt.bar(y_pos,df_more_5000['Review_rate'])
plt.xticks(y_pos, bars, rotation=45)
plt.axhline(df_more_5000['Review_rate'].mean(), color='r', linestyle='-')
plt.title('Review rate for app with more than 5000 downloads ')
plt.xlabel('Installs')
plt.ylabel('Review Rate (%)')
plt.show()
df_apple = df
df_apple['Rating_Total'] = df_apple['rating_count_tot'] + df_apple['rating_count_ver']
df_apple['Downloads_Estimate'] = 35.7 * df_apple['Rating_Total']

bins = [5000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000,500000000,1000000000]
labels = ['5000+','50000+','100000+','500000+','1000000+','5000000+','10000000+','50000000+','100000000+','500000000+']
df_apple['bins'] = pd.cut(df_apple['Downloads_Estimate'], bins=bins, labels=labels)
installs = df_apple.groupby(['bins']).mean()['user_rating_ver']

df_apple_installs = pd.DataFrame(installs)
df_apple_installs
df_apple['bins'].value_counts()
