# Importing required libraries 
import requests
from bs4 import BeautifulSoup
import pandas as pd 
# Target URL to scrap 
url = "https://www.goibibo.com/hotels/5-star-hotels-in-lucknow-cg/" 
headers= { }
response=requests.request("GET",url)
response
type(response.text)
# Parse - syntatical analysis - understand the symbols , strings and mke it in usable data form and give information
# Beautiful Soup is a Python library for getting data out of HTML, XML, and other markup languages. 
data=BeautifulSoup(response.text,'html.parser')
# Find all sections with specific class name 
cards_data=data.find_all('div',attrs={'class','HotelCardstyles__WrapperSectionMetaDiv-sc-1s80tyk-2 cPtMpq'})
#Totat number of cards found 
print("Total number of cards found: ", len(cards_data))
#for cards in cards_data:
    #print(cards)
# Extract the hotel name and price per room 
for cards in cards_data:
    
    #get the hotel name , found in a tag , only ine tag if this kind for each card 
    hotel_name = cards.find('a')
    
    #get the room price , found in p tag , more than one tag so need to mention the class name along with the tag 
    room_price = cards.find('p',attrs={'class':'HotelCardstyles__CurrentPrice-sc-1s80tyk-27 ieJkCi'})
    print(hotel_name.text,room_price.text)  
# So now final is to get this into CSV or JSON file :
# Creating a list to store the data 
scraped_data= []

for cards in cards_data:
    
    # initialize the dictionary
    card_details = {}
    
    #Get the hotel name 
    hotel_name = cards.find('a')
    
    #Get the room price
    room_price = cards.find('p',attrs={'class':'HotelCardstyles__CurrentPrice-sc-1s80tyk-27 ieJkCi'})
   
    #Add data to dictionary
    card_details['hotel_name']=hotel_name.text
    card_details['room_price']=room_price.text
    scraped_data.append(card_details)  #This is the list of dictionaries , adding dictionaries to the list 
scraped_data
#Create dataframe from the list of dictionaries 
final= pd.DataFrame.from_dict(scraped_data)
# Save the scraped data as CSV File 
final.to_csv('LKO_5Star_hotels.csv',index=False,sep=',')
# Single web-page scrapping can be done using some commands but for multiple pages it is not easy , have to write the script
# Scrap website URLs and email IDs 
# Marketing Teams scrap Email IDs in bulk 
# Web Scraping URLs and Email IDs 

#Importing required libraries 
import urllib.request 
from bs4 import BeautifulSoup

#URL to Scrap 
Logistic = "https://dlca.logcluster.org/display/public/DLCA/4.1+Nepal+Government+Contact+List"
#Query the website and get the response stored in the variable page 
PageR=urllib.request.urlopen(Logistic)
#Parse the HTML using Beautiful Soup 
soup = BeautifulSoup(PageR,features='html.parser')
PageR.url ,type(PageR)
print('\n\nPage Scraped!!!\n\n')
print("Title of the page is : ", soup.title.string)
print("All the URLS in page")
All_U=soup.find_all('a')
print("Total number of URLs present = ",len(All_U))
print("Last five urls in the page are :")

if len(All_U)>5:
    last_5=All_U[len(All_U)-5:] #You want only last five URLS from this list , start point and end will happen


for url in last_5:
    print(url.get('href'))  # Get the exact link from the hyperlink reference 

last_5
emails=[]    #List to store all the email IDs , hyperlink reference

for url in All_U:
    if(str(url.get('href')).find('@')>0):
        emails.append(url.get('href'))         #Store all the href of email IDs
        
print("The total number of Email IDs present are :",len(emails))
print("\n\nSome of the Email IDs are :\n\n")
print(emails[:5]) #Just get five email IDs
# Web Scraping - Scrap Images 
# Importing required libraries 
import requests
from bs4 import BeautifulSoup
# Target URL to scrap 
Iurl = "https://www.goibibo.com/hotels/5-star-hotels-in-lucknow-cg/" 
res=requests.request("GET",Iurl)
res
type(res.text)
# Beautiful Soup is a Python library for getting data out of HTML, XML, and other markup languages. 
Data=BeautifulSoup(res.text,'html.parser')
# Find all with the image tag 
images = Data.find_all('img',src=True)

print("Number of Images :",len(images))
link=Data.find(itemprop="image")
link
#for image in images:
 #   print(image)
#src part of your link contains the image format and it is as hyperlink
#Select src tag
image_src=[x['src'] for x in images]

#This is to select only the jpg format
images_src=[x for x in image_src if x.endswith('.png')]


for image in images_src:
    print(image)
image_src[4]
image_count = 1 
for image in images_src:
    with open('image_'+str(image_count)+'.gif','wb') as f:
        res= requests.get(image)
        f.write(res.content)
    image_count=image_count+1
# Scrape data om Page load , load more buttom etc , inspect the page to find how to load the data
