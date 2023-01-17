import urllib3

import os

import requests

import re

from bs4 import BeautifulSoup

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
url="https://www.imdb.com/chart/top?ref_=nv_mv_250"

os.environ['NO_PROXY'] = 'imdb.com'

req = requests.get(url)

page = req.text



soup = BeautifulSoup(page, 'html.parser')

#print (soup.prettify())
links=[]

for a in soup.find_all('a'): #, href=True):

    links.append(a.get('href'))

links=['https://www.imdb.com'+a.strip() for a in links if a is not None and a.startswith('/title/tt') ]



#---------------------------Remove duplicates in links

top_250_links=[]

for c in links:

    if c not in top_250_links:

        top_250_links.append(c)

#top_250_links=top_250_links[2:]



print(len(top_250_links))
column_list=['Rank','Movie_name' ,'URL' ,'Release_Year' ,'IMDB_Rating' ,

'Reviewer_count' ,'Censor_Board_Rating' ,'Movie_Length' ,'Genre_1' ,

'Genre_2' ,'Genre_3' ,'Genre_4' ,'Release_Date' ,'Story_Summary' ,

'Director' ,'Writer_1' ,'Writer_2' ,'Writer_3' ,'Star_1' ,

'Star_2' ,'Star_3' ,'Star_4' ,'Star_5' ,'Plot_Keywords' ,'Budget' ,

'Gross_USA' ,'Cum_Worldwide_Gross' ,'Production_Company' 

]

df = pd.DataFrame(columns=column_list)#,index=t) 

df


for x in np.arange(0, len(top_250_links)):

    #---------------------------Load html page for 1st movie in top 250 movies

    url=top_250_links[x]

    req = requests.get(url)

    page = req.text

    soup = BeautifulSoup(page, 'html.parser')

    

    #---------------------------Retrieve Movie details from html page

    Movie_name=(soup.find("div",{"class":"title_wrapper"}).get_text(strip=True).split('|')[0]).split('(')[0]

        

    year_released=((soup.find("div",{"class":"title_wrapper"}).get_text(strip=True).split('|')[0]).split('(')[1]).split(')')[0]

        

    imdb_rating=soup.find("span",{"itemprop":"ratingValue"}).text

    

    reviewer_count=soup.find("span",{"itemprop":"ratingCount"}).text

    

    subtext= soup.find("div",{"class":"subtext"}).get_text(strip=True).split('|') #Censor_rating

    if len(subtext)<4:

        censor_rating='Not Rated'

        movie_len=subtext[0]

        genre_list=subtext[1].split(',')

        while len(genre_list)<4:         genre_list.append(" ")

        genre_1,genre_2,genre_3,genre_4=genre_list

        release_date=subtext[2]

    else:

        censor_rating=subtext[0]

        movie_len=subtext[1]

        genre_list=subtext[2].split(',')

        while len(genre_list)<4:         genre_list.append(" ")

        genre_1,genre_2,genre_3,genre_4=genre_list

        release_date=subtext[3]

        

    story_summary=soup.find("div",{"class":"summary_text"}).get_text(strip=True).strip()

    

    #---------------------------Director,Writer and Actor details

    b=[]

    for a in soup.find_all("div",{"class":"credit_summary_item"}):

        c=re.split(',|:|\|',a.get_text(strip=True))         #print("c - ",c)

        b.append(c)                                         #print(''.join(a.get_text(strip=True)))

    stars=b.pop()

    writers=b.pop()

    directors=b.pop()

    if 'See full cast & crew»' in stars: stars.remove('See full cast & crew»')

    if '1 more credit»' in writers: writers.remove('1 more credit»') 

    if '1 more credit»' in directors: directors.remove('1 more credit»')

    stars=stars[1:]

    writers=writers[1:]

    directors=directors[1:]

    while len(stars)<5:         stars.append(" ")

    while len(writers)<3:         writers.append(" ")



    star_1,star_2,star_3,star_4,star_5=stars

    

    writer_1,writer_2,writer_3=writers

    

    director=directors[0]

    

    #---------------------------Plot Keywords

    b=[]

    for a in soup.find_all("span",{"class":"itemprop"}):     b.append(a.get_text(strip=True))  

    

    plot_keywords='|'.join(b)

    

    #---------------------------Commercial details and Prod Company

    

    

    b=[]                    #---------------------------Remove unwanted entries

    d={'Budget':'', 'Opening Weekend USA':'','Gross USA':'','Cumulative Worldwide Gross':'','Production Co':''}

    for a in soup.find_all("div",{"class":"txt-block"}):

        c=a.get_text(strip=True).split(':')

        if c[0] in d:

            b.append(c)



    for i in b:             #---------------------------Update default values if entries are found

            if i[0] in d: 

                d.update({i[0]:i[1]})                

        #print(d)



    production_company=d['Production Co'].split('See more')[0]

    cum_world_gross=d['Cumulative Worldwide Gross'].split(' ')[0]

    gross_usa=d['Gross USA'].split(' ')[0]

    budget=d['Budget']

    

    #---------------------------Dictionary to holds all details

    movie_dict={

        'Rank':x+1,

        'Movie_name' : Movie_name,

        'URL' : url,

        'Release_Year' : year_released,

        'IMDB_Rating' : imdb_rating,

        'Reviewer_count' : reviewer_count,

        'Censor_Board_Rating' : censor_rating,

        'Movie_Length' : movie_len,

        'Genre_1' : genre_1,

        'Genre_2' : genre_2,

        'Genre_3' : genre_3,

        'Genre_4' : genre_4,

        'Release_Date' : release_date,

        'Story_Summary' : story_summary,

        'Director' : director,

        'Writer_1' : writer_1,

        'Writer_2' : writer_2,

        'Writer_3' : writer_3,

        'Star_1' : star_1,

        'Star_2' : star_2,

        'Star_3' : star_3,

        'Star_4' : star_4,

        'Star_5' : star_5,

        'Plot_Keywords' : plot_keywords,

        'Budget' : budget,

        'Gross_USA' : gross_usa,

        'Cum_Worldwide_Gross' : cum_world_gross,

        'Production_Company' : production_company

        }

    #print(movie_dict['Rank'],":",movie_dict['Movie_name'])

    

    #---------------------------Append rows to dataframes using dictionary

    df = df.append(pd.DataFrame.from_records([movie_dict],columns=movie_dict.keys() ) )

df=df[column_list]  

df=df.set_index(['Rank'], drop=False) #should be run only once

df.head(20)
df.tail(20)