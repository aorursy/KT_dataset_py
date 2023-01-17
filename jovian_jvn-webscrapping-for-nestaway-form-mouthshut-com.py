import requests

from bs4 import BeautifulSoup
search_query="Nestaway-com-reviews-925868223"

base_url="https://www.mouthshut.com/websites/"
url= base_url+search_query

print(url)
search_response=requests.get(url)## pass the url to mouthshut an requesting the response

print(search_response)
print(search_response.status_code)

print(search_response.content)
reviews=BeautifulSoup(search_response.content)

print(reviews)
reviews.title
review=reviews.find("div",{"class": "left-panel"})
data=review.find("div", {"class":"read-review-holder"})

print(data)


new=[]

def test(review):

    for x in review:

        new.append(x.text)

    return new  

reviews_1=data.findAll('div', {"class":"more reviewdata"})

print(reviews_1)

print(len(reviews_1))
def extractReview(Review):

    r=[]

    for  i in Review:

        r.append(i.text)

    return r
review_1= extractReview(reviews_1)

print(review_1)
readmore=data.find("a",{"onclick":"bindreviewcontent('3041453',this,false,'I found this review of Nestaway.com pretty useful',925868223,'.png','I found this review of Nestaway.com pretty useful %23WriteShareWin','https://www.mouthshut.com/review/Nestaway-com-review-mqlroqpmplo','Nestaway.com',' 1/5','mqlroqpmplo');"})
# Data manipulation

import pandas as pd

import numpy as np



# Options for pandas

pd.options.display.max_columns = None

pd.options.display.max_rows = None



pd.options.display.max_colwidth=-1



# Display all cell outputs

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'



from IPython import get_ipython

ipython = get_ipython()

import re
total_reviews=reviews.find("span",{"id":"ctl00_ctl00_ContentPlaceHolderFooter_ContentPlaceHolderBody_lblnoofrecords"})
total_review=total_reviews.text

total_review
total_review=total_review.replace("Showing: 1 - 20 of ","")

print(total_review)

total_review=re.sub(" Reviews.*","",total_review)

print(total_review)
total_review=int(total_review)
total_pages=total_review//20



if total_pages%20!=0:

    total_pages=total_pages+1

print(total_pages)
reviews_url=[]

for more_div in reviews_1:

    review_url=more_div.find("a")['onclick']

    reviews_url.append(review_url)



reviews_url
urls=[]



for review_url in reviews_url:

    review_url=review_url.replace('bindreviewcontent',"")

    review_url=review_url.replace(";","")

    review_url=review_url.replace('"',"")

    review_url=review_url.strip("()")

    ## Convert the review url string to tuple or split on comma 

    review_url_split=review_url.split(",")

    #print(review_url_split)

    #If a value after splitting starts with "https:" it is tghe url of the review

    url=[value for value in review_url_split if "www.mouthshut.com/review" in value][0]

    urls.append(url)
for review_url in reviews_url:

    review_url=review_url.replace('bindreviewcontent',"")

    review_url=review_url.replace(";","")

    review_url=review_url.replace('"',"")

    review_url=review_url.strip("()")

    ## Convert the review url string to tuple or split on comma 

    review_url_split=review_url.split(",")

    
review_url1=urls[0]

review_url1
def cleancontent(urls):

    print(urls)

    search_response1=requests.get(urls)

    print(search_response1)

    search_response1.status_code

    #search_response1.content

    reviews=BeautifulSoup(search_response.content)

    reviews.prettify

    return reviews
review_url1=cleancontent(urls[0])
exactreview=review_url1.find("div",{"class": "left-panel"})
exact=exactreview.find("div", {"class":"read-review-holder"})

print(exact)
exactrow=exact.find('div',{"class":"row review-article"})

print(exactrow)
reviews_exact=exactrow.find('div', {"class":"rev-main-content"})

print(reviews_exact)
urls=[]

for i in range(1,19):

    url="https://www.mouthshut.com/websites/"

    query="Nestaway-com-reviews-925868223""-page-"+str(i)

    print (url+query)

    urls.append(url+query)


def getreviewUrl(urls):

    reviews=cleancontent(urls)   

    review=reviews.find("div",{"class": "left-panel"})

    data=review.find("div", {"class":"read-review-holder"})

    reviews_1=data.findAll('div', {"class":"more reviewdata"})

    reviews= extractReview(reviews_1)

    

    rating=review.findAll('div',{"class":"rating"})

    ratings=[]

    for x in rating:

        rating=x.findAll("i",{"class":"icon-rating rated-star"})

        ratings.append(len(rating))

    #print(len(review))

    #print(review)

        

    nestaway=pd.DataFrame()

    nestaway['ratings']=ratings

    nestaway['reviews']=reviews

    return nestaway

    
nestaway=getreviewUrl(urls[0])
rating=review.findAll('div',{"class":"rating"})

ratings=[]

for x in rating:

    rating=x.findAll("i",{"class":"icon-rating rated-star"})

    ratings.append(len(rating))
print(len(ratings))
nestaway
nestaway1=getreviewUrl(urls[1])
nestaway2=getreviewUrl(urls[2])
nestaway3=getreviewUrl(urls[3])
nestaway4=getreviewUrl(urls[4])
nestaway5=getreviewUrl(urls[5])
nestaway6=getreviewUrl(urls[6])
nestaway7=getreviewUrl(urls[7])
nestaway8=getreviewUrl(urls[8])
nestaway9=getreviewUrl(urls[9])
nestaway10=getreviewUrl(urls[10])
nestaway11=getreviewUrl(urls[11])
nestaway12=getreviewUrl(urls[12])
nestaway13=getreviewUrl(urls[13])
nestaway14=getreviewUrl(urls[14])
nestaway15=getreviewUrl(urls[15])
nestaway16=getreviewUrl(urls[16])
nestaway17=getreviewUrl(urls[17])
nestaway0=getreviewUrl(url)
nestaway_reviews=pd.concat([nestaway0, nestaway, nestaway1, nestaway2, nestaway3, nestaway4, nestaway5, nestaway6, nestaway7, nestaway8, nestaway9, nestaway10, nestaway11, nestaway12, nestaway13, nestaway14, nestaway15, nestaway16, nestaway17 ])
nestaway_reviews.shape
nestaway_reviews.to_csv("nestaway.csv")