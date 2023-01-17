import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sys

reload (sys)

sys.setdefaultencoding('utf-8')

data=pd.read_csv("../input/movie_metadata.csv")

data.head()

data.describe()
import pylab as pl



imdbScore=[[]]

x=[]



for i in pl.frange(1,9.5,.5):

    imdbScore.append(len(data.imdb_score[(data.imdb_score>=i) & (data.imdb_score<i+.5)]))

    x.append(i)



del(imdbScore[0])



plt.figure(figsize=(15,12))

plt.title("Histogram Of IMDB Score")

plt.ylabel("IMDB Score")

plt.xlabel('Frequency')

plt.barh(x,imdbScore,height=.45 ,color='green')

plt.yticks(x)

plt.show()
plt.figure(figsize=(12,10))

plt.title("IMDB Score Vs Director Facebook Popularity")

plt.xlabel("IMDB Score")

plt.ylabel("Facebook Popularity")

tmp=plt.scatter(data.imdb_score,data.director_facebook_likes,c=data.imdb_score,vmin=3,vmax=10)

plt.yticks([i*2500 for i in range(11)])

plt.colorbar(tmp,fraction=.025)

plt.show()
plt.figure(figsize=(12,10))

plt.title("IMDB Score Vs Cast Facebook Popularity")

plt.xlabel("IMDB Score")

plt.ylabel("Facebook Popularity")

tmp=plt.scatter(data.imdb_score,data.cast_total_facebook_likes,c=data.imdb_score,vmin=3,vmax=10)

plt.yticks([i*70000 for i in range(11)])

plt.colorbar(tmp,fraction=.025)

plt.show()
data=data.dropna()

year=(data.title_year.unique()).astype(int)

year=sorted(year)



yearImdbScore=[]

numOfMovieByYear=[]

for i in year:

    tmp=data.imdb_score[data.title_year==i]

    yearImdbScore.append(tmp)

    numOfMovieByYear.append(len(tmp))





#plt.figure(2)

plt.figure(figsize=(15,10))

plt.title("IMDB Score Vs Year")

plt.ylabel("IMDB Score")

plt.xlabel('Year')

plt.boxplot(yearImdbScore,widths=.75)

year=list(np.insert(year,0,0))

plt.xticks(range(len(year)),year,rotation=90,fontsize=10)

plt.show()



del(year[0])




country=data.country.unique()

countryImdbScore=[]



for i in country:

    countryImdbScore.append(data.imdb_score[data.country==i])





country=np.insert(country,0,'')



plt.figure(figsize=(14,10))

plt.title("IMDB Score Vs Country")

plt.ylabel("IMDB Score")

plt.xlabel('Country')

plt.boxplot(countryImdbScore,widths=.75,)

plt.xticks(range(len(country)),country,rotation=90,fontsize=8)

plt.show()
correlation=data.corr()

correlation
plt.figure(figsize=(10,10))

tmp=plt.matshow(correlation,fignum=1)

plt.xticks(range(len(correlation.columns)),correlation.columns,rotation=90,fontsize=8)

plt.yticks(range(len(correlation.columns)),correlation.columns,fontsize=8)

plt.colorbar(tmp,fraction=0.035)

plt.show()
avgimdbScore=[sum(i)/len(i) for i in yearImdbScore]

convertedNumMovie=[i/float(20) for i in numOfMovieByYear]

plt.figure(figsize=(14,11))

plt.title("Average Of IMDB Score & Number Of Movies Through Year")

plt.xlabel('Year')

plt.plot(year,avgimdbScore)

plt.plot(year,convertedNumMovie,',r-')

plt.legend(['Average IMDB Score','Number Of Movies'],loc='lower left')

plt.yticks(range(10))

plt.show()
director=list(data.director_name.unique())

df=pd.DataFrame(columns=['director','directorScoreMean','directorImdbScore','directorMovieNUm'])



for i in director:

    tmp=list(data.imdb_score[data.director_name==i])

    if len(tmp)>1:

        df=df.append({'director': i,'directorScoreMean': sum(tmp)/len(tmp),'directorImdbScore': tmp,'directorMovieNUm' :len(tmp)},ignore_index=True)

    

tmp=(df.sort_values(['directorScoreMean'],ascending=False)).head(25)

directorByMeanScore=list(tmp.director)

directorByMeanScore.reverse()

ScoreByMeanScore=list(tmp.directorImdbScore)

ScoreByMeanScore.reverse()





tmp=(df.sort_values(['directorMovieNUm'],ascending=False)).head(25)

directorByMovieNum =list(tmp.director)

directorByMovieNum.reverse()

ScoreByMovieNum=list(tmp.directorImdbScore)

ScoreByMovieNum.reverse()
#directorByMovieNum

plt.figure(figsize=(11,11))

for i in range(len(directorByMovieNum)):

    for j in ScoreByMovieNum[i]:

        plt.scatter(i,j,c=j,vmin=3,vmax=10)





#tmp=plt.scatter(c=ScoreByMovieNum)

plt.colorbar(fraction=.04)

plt.title("Top Director vs Their IMDB Score\n Interm Of The Number Of Movies They Make ")

plt.ylabel('Movies IMDB Score')

plt.xlabel("\nDirector's Name")

plt.xticks(range(25),directorByMovieNum,rotation=90)



plt.show()
#directorByMeanScore

plt.figure(figsize=(11,11))

for i in range(len(directorByMeanScore)):

    for j in ScoreByMeanScore[i]:

        plt.scatter(i,j,c=j,vmin=6,vmax=10)



plt.xticks(range(25),directorByMeanScore,rotation=90)

plt.title("Top Director vs Their IMDB Score\n Interm Of Their Average IMDB Score ")

plt.ylabel('Movies IMDB Score')

plt.xlabel("\nDirector's Name")

plt.colorbar(fraction=.04)

plt.show()