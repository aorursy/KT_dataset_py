#Libraries for Data Manipulation
import pandas as pd
import numpy as np

#library for requesting url/ making HTTP requests 
import requests as rq

#Beautiful Soup is a Python package for parsing HTML and XML documents.
from bs4 import BeautifulSoup
url = "https://www.imdb.com/search/title/?groups=top_250&sort=user_rating"

html = rq.get(url)

#An HTMLParser instance is fed HTML data and calls handler methods when start tags, end tags, text, comments, and other markup elements are encountered. The user should subclass HTMLParser and override its methods to implement the desired behavior
soup = BeautifulSoup(html.text,"html.parser")
type(soup)
#creating a container which search the html file and include the div with the class: lister-item mode-advanced
movie_containers = soup.find_all('div',{'class':"lister-item mode-advanced"})

#print the number of container that is being shown in the page
print(len(movie_containers))

#It would display 50 as the number of movie that is being shown in the page
first_movie = movie_containers[0]

#Now we can access the various tag within the first_movie as below

first_movie.h3.a

#The above code display the tag <a> within the <h3> tag
# By adding .text in above code, we now can access the text in the <a> tag
name1 = first_movie.h3.a.text
#we can extract the rank in the same way
#.replace is used to delete the '.' which is being displayed after the rank
rank = first_movie.h3.span.text.replace('.','')

#Converting text into float
rank1 = float(rank)
#Extracting ratings from the <strong> tag and converting it into float
imdb_ratings1 = first_movie.strong.text
imdb_ratings1 = float(imdb_ratings1)
#Extracting the Movie released year from the <span> tag with the class name lister-item-year text-muted unbold
year1 = first_movie.h3.find('span', {'class':'lister-item-year text-muted unbold'}).text

#First of all by using the strip method, we are deleting '()' from the output
#Then converting it into int
year1 = int(year1.strip('()'))
# Just as the process above the votes, genres and metascore of the movie are also extracted
vote = first_movie.find('span', {'name':'nv'}).text 
genres = container.p.find('span' ,{'class' : 'genre'}).text.strip()
metascore = first_movie.find('span',{'class':'metascore favorable'})
#creating empty list
ranks = []
names = []
years = []
imdb_ratings = []
meta_scores = []
votes = []
gross = []
genre = []
for container in movie_containers:
        ranking = container.h3.span.text.replace('.','')
        #append is used to add all the rank of the movies in the list
        ranks.append(ranking)
        name = container.h3.a.text
        names.append(name)
        year = container.h3.find('span', {'class':'lister-item-year text-muted unbold'}).text.strip('()')
        years.append(year)
        imdb = container.strong.text
        imdb_ratings.append(imdb)
        metascore = container.find('span',{'class':'metascore favorable'})
        meta_scores.append(metascores)
        vote = container.find('span', {'name':'nv'}).text
        votes.append(vote)  
        genres = container.p.find('span' ,{'class' : 'genre'}).text.strip()
        genre.append(genres)
#creating empty list
ranks = []
names = []
years = []
imdb_ratings = []
meta_scores = []
votes = []
gross = []
genre = []
for container in movie_containers:
        ranking = container.h3.span.text.replace('.','')
        ranks.append(ranking)
        name = container.h3.a.text
        names.append(name)
        year = container.h3.find('span', {'class':'lister-item-year text-muted unbold'}).text.strip('()')
        years.append(year)
        imdb = container.strong.text
        imdb_ratings.append(imdb)
        if container.find('span',{'class':'metascore favorable'}):
            metascore = container.find('span',{'class':'metascore favorable'}).text
            meta_scores.append(metascore)
        else:
            meta_scores.append(metascore)
        vote = container.find('span', {'name':'nv'}).text
        votes.append(vote)  
        genres = container.p.find('span' ,{'class' : 'genre'}).text.strip()
        genre.append(genres)
imdb = pd.DataFrame({
    'Rank' : ranks,
    'Movie' : names,
    'Year' : years,
    'IMDB' : imdb_ratings,
    'Metascore' : meta_scores,
    'Votes' : votes,
    'Genre' : genre
})
print(imdb.info())
pages = np.arange(1,250,50)
print(pages)
page = rq.get("https://www.imdb.com/search/title/?groups=top_250&sort=user_rating,desc&start=" + str(page) + "&ref_=adv_nxt")
from time import sleep
from random import randint

sleep(randint(2,10))
import requests as  rq
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from time import sleep
from random import randint
ranks = []
names = []
years = []
imdb_ratings = []
meta_scores = []
votes = []
gross = []
genre = []

pages = np.arange(1,250,50)
for page in pages:
    page = rq.get("https://www.imdb.com/search/title/?groups=top_250&sort=user_rating,desc&start=" + str(page) + "&ref_=adv_nxt")
    soup = BeautifulSoup(page.text,'html.parser')
    
    movie_containers = soup.find_all('div',{'class':"lister-item mode-advanced"})
    
    sleep(randint(2,10))
    
    for container in movie_containers:
            ranking = container.h3.span.text.replace('.','')
            ranks.append(ranking)
  
            name = container.h3.a.text
            names.append(name)

            year = container.h3.find('span', {'class':'lister-item-year text-muted unbold'}).text.strip('()')
            years.append(year)

            imdb = container.strong.text
            imdb_ratings.append(imdb)
            
            if container.find('span',{'class':'metascore favorable'}):
                metascore = container.find('span',{'class':'metascore favorable'}).text
                meta_scores.append(metascore)
            else:
                meta_scores.append(metascore)

            vote = container.find('span', {'name':'nv'}).text
            votes.append(vote)  

            genres = container.p.find('span' ,{'class' : 'genre'}).text.strip()
            genre.append(genres)

imdb = pd.DataFrame({
    'Rank' : ranks,
    'Movie' : names,
    'Year' : years,
    'IMDB' : imdb_ratings,
    'Metascore' : meta_scores,
    'Votes' : votes,
    'Genre' : genre
})  
print(imdb.info())
#save the data as csv file using .to_csv to your defined location in your local computer
data = imdb.to_csv(r"C:\Users\pokha\Desktop\Data science\Imdb.csv", index = False, header=True)