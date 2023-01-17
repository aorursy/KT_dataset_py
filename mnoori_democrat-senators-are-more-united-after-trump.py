#importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

import requests

from bs4 import BeautifulSoup
urls=[]



#first url in the first session of 115th Congress. It runs from Jan 03 untill Dec 21, 2017.

first_url_1stsession='https://www.senate.gov/legislative/LIS/roll_call_votes/vote1151/vote_115_1_00001.xml'



for i in range(1,326):

    url=re.sub('0+([0-9]+).xml$',str(i).zfill(5)+'.xml',first_url_1stsession)

    urls.append(url)
first_url_2ndsession='https://www.senate.gov/legislative/LIS/roll_call_votes/vote1152/vote_115_2_00001.xml'



for i in range(1,16):

    url=re.sub('0+([0-9]+).xml$',str(i).zfill(5)+'.xml',first_url_2ndsession)

    urls.append(url)



# First 3 urls of final list    

urls[:3]
def votes_scraper(urls):

    votes_dict={}

    for url in urls:

        try:

            names=[]

            votes=[]

            doc_name=''

            

            #loading url contents 

            page=requests.get(url)

            

            #instatiating BeautifulSoup

            soup=BeautifulSoup(page.content, 'html.parser')

            

            #creating unique keys for dictionary

            cong_year=soup.find('congress_year').text

            vote_num=soup.find('vote_number').text

            doc_name=cong_year+'_'+vote_num.zfill(3)

            

            #loading senator names and their votes

            names_tag=soup.find_all('member_full')

            votes_tag=soup.find_all('vote_cast')

            names=[names_tag[i].text for i in range(len(names_tag))]

            votes=[votes_tag[i].text for i in range(len(votes_tag))]

        

            #storing data in a dictionary

            votes_dict[doc_name]={k:v for k, v in zip(names, votes)}

        except:

            print(url)

            pass

        

    return votes_dict
#Note: at the time of this analysis, I recieve an error on getting access to the urls. The error...

#...says there is a potential security risk. I assume it is becuase this function loads a few...

#...urls in a short perio of time.



#Uncomment in your own code

#votes=votes_scraper(urls)
# Feel free to uncomment following lines after executing the above cell. 

#votes_df=pd.DataFrame(votes)



votes_df=pd.read_csv('../input/../input/votes_115thCongress.csv')

votes_df.head()
# replacing null values with Not Voting

votes_df=votes_df.replace('Not Voting',np.nan)



# extracting party affiliations 

votes_df['Party']=votes_df['index'].apply(lambda x: re.findall('\(([A-za-z])',x)[0])



# reseting the index. This will add a new column with our old index's.

votes_df=votes_df.set_index('index',drop=True)



#Percent of missing data in each each row.

num_na_row=votes_df.isnull().sum(axis=1).sort_values(ascending=False)/votes_df.shape[1]*100

num_na_row.head()
drop_sens=['Jones (D-AL)','Smith (D-MN)','Sessions (R-AL)']

votes_df=votes_df.drop(drop_sens,axis=0)
votes_df.shape
map_dict={'Yea':1,

         np.nan:0.5,

         'Present':0.5,

         'Nay':0}

votes_df_numeric=votes_df.replace(map_dict)



#Let's check if there is any non numerical values in our dataframe!

votes_df_numeric.select_dtypes(include=['object']).head()
votes_df_numeric['Party'].value_counts()
from sklearn.cluster import KMeans



X=votes_df_numeric.iloc[:,1:(votes_df_numeric.shape[1]-1)]

kmeans= KMeans(n_clusters=2, random_state=1)



senator_distances=kmeans.fit_transform(X)



#labeling each senator based on the kmeans algorithm.

labels=kmeans.labels_



pd.crosstab(votes_df_numeric['Party'],labels)
dis=pd.DataFrame(senator_distances)

dis.columns=['Distance from 1st cluster','Distance from 2nd cluster']

dis['Actual_Party']=votes_df_numeric.reset_index()['Party']
sns.set_style("dark")

sns.set_context("talk")

sns.lmplot(x='Distance from 1st cluster',y='Distance from 2nd cluster',data=dis,

           hue='Actual_Party',scatter=True,fit_reg=False,size=6, scatter_kws={"s": 100},

           legend=False,palette=sns.color_palette(['red','blue','gray']))

plt.title('Distance from clusters',fontsize=20)

plt.legend(loc=0,title='Actual Party',fontsize =12)

plt.show()
extremism=((senator_distances)**3).sum(axis=1)

votes_df_numeric['extremism']=extremism

votes_df_numeric.sort_values('extremism',inplace=True,ascending=False)

votes_df_numeric.head(10)
urls_obama=[]



first_url_1stsession_113='https://www.senate.gov/legislative/LIS/roll_call_votes/vote1131/vote_113_1_00001.xml'

for i in range(1,292):

    url=re.sub('0+([0-9]+).xml$',str(i).zfill(5)+'.xml',first_url_1stsession_113)

    urls_obama.append(url)



first_url_2ndsession_113='https://www.senate.gov/legislative/LIS/roll_call_votes/vote1132/vote_113_2_00001.xml'

for i in range(1,367):

    url=re.sub('0+([0-9]+).xml$',str(i).zfill(5)+'.xml',first_url_2ndsession_113)

    urls_obama.append(url)

    

first_url_1stsession_114='https://www.senate.gov/legislative/LIS/roll_call_votes/vote1141/vote_114_1_00001.xml'

for i in range(1,340):

    url=re.sub('0+([0-9]+).xml$',str(i).zfill(5)+'.xml',first_url_1stsession_114)

    urls_obama.append(url)



first_url_2ndsession_114='https://www.senate.gov/legislative/LIS/roll_call_votes/vote1142/vote_114_2_00001.xml'

for i in range(1,164):

    url=re.sub('0+([0-9]+).xml$',str(i).zfill(5)+'.xml',first_url_2ndsession_114)

    urls_obama.append(url)