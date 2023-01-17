from __future__ import unicode_literals
from facebook_scraper import get_posts 
from pymongo import MongoClient
import codecs
import unicodecsv as csv
import datetime
import csv
import codecs
import shutil
import io  
import requests
from PIL import Image
import gridfs

words_filter=[ ] #filtering words (Here the context is the daeth of the tunisian president)
page_id=''
Mongodb_uri='mongodb://127.0.0.1:27017'

#Verify if a word/words in in texte
def containAtLeastOneWord(text, words):
    for oneWord in words:
        if oneWord in text:
            return True
    return False

#Mongodb connect
def connect_mongodb(Mongodb_uri):
    client = MongoClient(Mongodb_uri)
    db=client.facebook_scrape
    serverStatusResult=db.command("serverStatus")
    #print(serverStatusResult)  
    print("Connected...")  
    return db

def insert_image(db,image_url):
    r = requests.get(image_url, stream=True)
    img = Image.open(r.raw)
    #img.show()
    fs = gridfs.GridFS(db)
    fs.put(img, filename="postimage")
    
#Scrapping Facebook page/group   
def ScrapeFacebookPage(page_id):
    file_name=page_id+'_facebook_status.csv'
    
    file = codecs.open(file_name, "wb", "utf-8")
    file_writer = csv.writer(file)
    file_writer.writerow(['post_id', 'text', 'post_text','time', 'image','likes','comments', 'shares', 'reactions','post_url', 'link'])
    db=connect_mongodb(Mongodb_uri)
    
    for post in get_posts(group='kaissaied', pages=20):
        #Filter only content that repnds to di
        if(containAtLeastOneWord(post, words_filter)):   
            #Insert images in Mongodb instance
            image_url=post['image']
            if (image_url!=None):
                print(image_url)
                print("\n")
                insert_image(db,image_url)
            #insert posts in csv file
            file_writer.writerow(post.values())
            #insert file into mongodb Instance
            db.collection.insertOne(post)
    
    file.close()
    print("Scrapping & strorage done...") 
    
if __name__ == '__main__':
    ScrapeFacebookPage(page_id)

