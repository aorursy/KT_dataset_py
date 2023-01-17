# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
#Data cleaning

oscar=pd.read_csv('/kaggle/input/the-oscar-award/the_oscar_award.csv')

#print(oscar.info())





oscar['category']=oscar['category'].apply(lambda x:x.strip())

oscar['name']=oscar['name'].apply(lambda x:x.strip().replace("(","").replace(")",""))

oscar['film']=oscar['film'].apply(lambda x: np.NaN if str(x)=='nan' else str(x).strip())



oscar['category']=oscar['category'].apply(lambda x:x.capitalize())







category=oscar['category'].str.split('(',n=1,expand=True)

category.columns=['category1','category2']

category['category2']=category['category2'].apply(lambda x:str(x).capitalize().replace(")",""))

oscar=pd.concat([oscar,category],axis=1)

#print(oscar['name'])







nameRole=oscar['name'].str.split(',',expand=True)

nameRole=nameRole.rename(columns={0:'one_of_the_names'})

nameRole['one_of_the_names']=nameRole['one_of_the_names'].apply(lambda x: x if x.find(":")< 0 else x[x.index(':')+1:] )



oscar=pd.concat([oscar,nameRole['one_of_the_names']],axis=1)
print(oscar.info())
print(oscar.head(10))
oscar[oscar["year_ceremony"] < 1950]["category"].unique()

#len == 52
nominations = oscar[((oscar["year_ceremony"] > 1950)

                     &(oscar["year_ceremony"] < 1970))]["category"].unique()



print (len(nominations), nominations)
nominations = oscar[((oscar["year_ceremony"] > 1970)

                     &(oscar["year_ceremony"] < 2000))]["category"].unique()



print (len(nominations), nominations)
nominations = oscar["category"].unique()

for word in nominations:

    if word.startswith("Writing"):

        print (word)
nominations = oscar[oscar["year_ceremony"] > 1970]["category"].unique()



print (len(nominations), nominations)
nominations = oscar[oscar["year_ceremony"] > 2000]["category"].unique()



print (len(nominations), nominations)
nominations = oscar["category"].unique()



print (len(nominations), nominations)
#Change from year to year

year_record=oscar[['year_ceremony','category1','name','film']].groupby(by='year_ceremony').nunique()

del year_record['year_ceremony']

year_record['true_winner']=oscar.loc[oscar.winner==1,['year_ceremony','winner']].groupby(by='year_ceremony').count()

year_record['false_winner']=oscar.loc[oscar.winner==0,['year_ceremony','winner']].groupby(by='year_ceremony').count()





#print(year_record.describe().round(4))



f, axes = plt.subplots(1,3, figsize=(16,6))

year_number=year_record.index.tolist()

yCategory=year_record['category1'].tolist()

yName=year_record['name'].tolist()

yFilm=year_record['film'].tolist()

vis1 = sns.lineplot(x=year_number, y=yCategory,ax=axes[0])

vis1.set(xlabel='Year', ylabel='Number Of Main Category')

axes[0].set_title('Number Of Main Category Change')

vis2 = sns.lineplot(x=year_number, y=yName,ax=axes[1])

vis2.set(xlabel='Year', ylabel='Number Of Participant')

axes[1].set_title('Number Of Nominees Change')

vis3 = sns.lineplot(x=year_number, y=yFilm,ax=axes[2])

vis3.set(xlabel='Year', ylabel='Number Of Film')

axes[2].set_title('Number Of Nominated Film Change')

plt.show()
#description

year_record['nominate']=oscar[['year_ceremony','winner']].groupby(by='year_ceremony').count()



yearDescr=round(year_record.describe(),4)

print(yearDescr)
#Change in award-winning rate per year

plt.subplot(2,1,1)

true_winner_line,false_winner_line=plt.plot(

    year_number,year_record['true_winner'].to_list(),

    year_number,year_record['false_winner'].to_list())





plt.setp(true_winner_line,color='r')

plt.setp(false_winner_line,color='g')

plt.ylabel('Number Of Record')

plt.xlabel('Year')

plt.title('Winner and Nominee Change')



plt.legend(handles = [true_winner_line,false_winner_line], labels = ['Winner', 'Only Nomined'], loc = 'best')





year_record['winning_probability']=year_record['true_winner']/year_record['nominate']

yWinning_probability=year_record['winning_probability'].to_list()

vis1 = sns.lineplot(x=year_number, y=yWinning_probability,ax=plt.subplot(2,1,2))

vis1.set(xlabel='Year', ylabel='Winning Probability')

vis1.set_title('Winning Probability Change')



plt.show()
#Film information

All_movie=oscar[['film','winner']].groupby(by='film').sum().sort_values(by='winner',ascending=False)

All_movie['Awards']=All_movie['winner'].apply(lambda x:int(x))

del All_movie['winner']

All_movie['Nominations']=oscar[['film','winner']].groupby(by='film').count()

All_movie['Winnin_rate']=All_movie['Awards']/All_movie['Nominations']



print(All_movie.head(10))
#Top 10 Nominated Movies

nominated_movie=All_movie.sort_values(by='Nominations',ascending=False).head(10)

nominated_movie['Diff']=nominated_movie['Nominations']-nominated_movie['Awards']

nominated_movie=nominated_movie.sort_values(by='Nominations')

vis1=plt.barh(nominated_movie.index.tolist(), nominated_movie['Awards'], 0.5,color = 'pink', label = 'Awards')

vis1=plt.barh(nominated_movie.index.tolist(), nominated_movie['Diff'],0.5, color = 'c', left = nominated_movie['Awards'], label = 'Only nominated')



plt.xlabel('Number Of Record')

plt.ylabel('Movie')



plt.xticks(range(0,max(nominated_movie['Nominations']),2))

plt.legend(loc='lower right',fontsize=8)

plt.title('Top 10 Nominated Movies')



plt.show()
#Top 10 Award-winning Movies

Award_winning_movie=All_movie.sort_values(by='Awards',ascending=False).head(10)

Award_winning_movie=Award_winning_movie.sort_values(by='Awards')

movie_name=Award_winning_movie.index.tolist()



count_award=Award_winning_movie['Awards'].to_list()





vis2= plt.barh(movie_name,count_award,height=0.5, color = 'pink')

plt.xlabel('Number Of Record')

plt.ylabel('Movie')

plt.title('Top 10 Award-winning Movies')

plt.xticks(range(0,max(Award_winning_movie['Awards'])+2,2))

plt.show()
#Time Distribution Of The Top 10 Award-winning Films

topTenMovieInfo=oscar.loc[oscar['film'].isin(movie_name),['year_ceremony','year_film','name','film','category1']]





plt.pie(topTenMovieInfo.year_film.value_counts(),labels=topTenMovieInfo.year_film.value_counts().index,

        autopct='%1.1f%%',shadow=False,startangle=50)



plt.title('Time Distribution Of The Top 10 Award-winning Films')



plt.show()
#winner infomation

nominated_person=oscar['one_of_the_names'].value_counts()

nominated_person=pd.DataFrame(nominated_person)



nominated_person['Awards']=oscar.loc[oscar.winner==1,['one_of_the_names','winner']].groupby(by='one_of_the_names').count()



print(nominated_person.head(10))
#Top 10 Nominates

topTenPerson=nominated_person.head(10)

topTenPerson=pd.DataFrame(topTenPerson)

topTenPerson=topTenPerson.sort_values(by='one_of_the_names')



topTenPerson=topTenPerson.rename(columns={'one_of_the_names':'Total nominations'})



topTenPerson['Awards']=oscar.loc[oscar.winner==1,['one_of_the_names','winner']].groupby(by='one_of_the_names').count()

topTenPerson['Only Nomined']=topTenPerson['Total nominations']-topTenPerson['Awards']



vis1= plt.barh(topTenPerson.index.tolist(),topTenPerson['Awards'].to_list(),height=0.5, color = 'pink',label='Awards')

vis1= plt.barh(topTenPerson.index.tolist(),topTenPerson['Only Nomined'].to_list(),height=0.5, left=topTenPerson['Awards'].to_list(),color = 'c',label='Only Nomined')

plt.xlabel('Number Of Record')

plt.ylabel('Person Or Organization')

plt.title('Top 10 Nominates')

plt.legend(loc='lower right',fontsize=8)



plt.show()
#Top 10 Winner

topTenAwardsP=nominated_person.sort_values(by='Awards',ascending=False).head(10)

topTenAwardsP=topTenAwardsP.sort_values(by='Awards')

#print(topTenAwardsP)

person_name=topTenAwardsP.index.tolist()



count_award=topTenAwardsP['Awards'].to_list()





vis2= plt.barh(person_name,count_award,height=0.5, color = 'pink')

plt.xlabel('Number Of Record')

plt.ylabel('Person Or Organization')

plt.title('Top 10 Winner')

#plt.xticks(range(0,int(max(topTenAwardsP['Awards']))+2))



plt.show()