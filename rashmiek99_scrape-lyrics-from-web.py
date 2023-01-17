!pip install google 

from googlesearch import search

from bs4 import BeautifulSoup

import requests

def scrape_lyrics(details):

    for x in search(details,tld='co.in',lang='en',start=0,stop=1):

        

        site = x

        req = requests.get(site)

        

        if req.status_code != 200:

            print("Error getting information")

            return

        soup = BeautifulSoup(req.content,'html.parser')

        lyrics = ''

        

        for link in soup.find_all('p'):

            lyrics += link.text

            

    return lyrics    



site_song = 'hinditracks Dil Ne Kaha'    

            

#site_song = 'genius 'believer

                

print(scrape_lyrics(site_song))