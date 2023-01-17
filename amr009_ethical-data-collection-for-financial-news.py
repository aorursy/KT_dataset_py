import time
import csv
import os.path
import numpy as np 
import pandas as pd 
import requests 
from bs4 import BeautifulSoup 
def request_with_check(url):
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36 , For: a Tutorial kernel By: elamraoui sohayb'}
    page_response = requests.get(url, headers=headers, timeout=60)
    if page_response.status_code>299:
        raise AssertionError("page content not found, status: %s"%page_response.status_code)
    
    return page_response    
page_test = request_with_check('https://www.investing.com/news/commodities-news')
# Cheking the first 5000 charchters of the HTML code
page_test.text[:5000]
def get_details(single_article):
    
    # A title is in <a></a> with the 'class' attribute set to: title
    title = single_article.find('a',{'class':'title'})

    # A safeguard against some empty articles in the deeper pages of the site
    if title == None:
        #print('Empty Article')
        return None
    
    # the link to an article is the Href attribute
    link = title['href']
    
    # A safeguarde against embedded Advertisment articles
    if (('/news/'and category_name) not in link):
        #print('Ad Article found')
        return None       
        
    title = title.text
    
    # The first Paragraph is in <p></p>
    first_p = single_article.find('p').text
    
    # the Source is in <span></span>, with Class == articleDetails
    source_tag = single_article.find_all('span',{'class':'articleDetails'})
    source = str(source_tag[0].span.text)
    
    #date is also in <span></span> withe the Class == date
    date = single_article.find('span',{'class':'date'}).text
    
    return title, link, first_p, source, date  
def single_page(Url_page,page_id = 1):

    news_list = []

    #Making the Http request
    page = request_with_check(Url_page)
    
    #Calling the Html.parser to start extracting our data
    html_soup = BeautifulSoup(page.text, 'html.parser')
    
    # The Articles Class
    articles = html_soup.find('div',{'class':'largeTitle'})
    
    # The single Articles List
    articleItems = articles.find_all('article' ,{'class':'articleItem'})

    # Looping, for each single Article
    for article in articleItems:
        if get_details(article) == None:
            continue
        
        title, link, first_p, source_tag, date = get_details(article)
        news_list.append({'id_page':page_id,
                          'title':title,   
                          'date':date,
                          'link': link,
                          'source':source_tag,
                          'first_p':first_p})

    return news_list
def dict_to_csv (filename,news_dict):
    
    #Setting the Dataframe headers
    fields = news_dict[0]
    fields = list(fields.keys())
    
    #Checking if the file already exists, if Exists we woulb pe appending, if Not we creat it
    has_header = False
    if os.path.isfile(filename):
        with open(filename, 'r') as csvfile:
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(csvfile.read(2048))
    
    with open(filename, 'a',errors = 'ignore', encoding= 'utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        if(has_header == False):
            writer.writeheader()  
        for row in range(len(news_dict)):
            item = news_dict[row]
            writer.writerow(item)
def parsing_category_pages(category_name,base_url,number_pages):
    start_time = time.time()
    
    #getting the start page
    page = request_with_check(base_url)

    #Calling the Html Parser
    html_soup = BeautifulSoup(page.text, 'html.parser')
    
    #Finding the Laste page
    last_page = int(html_soup.findAll(class_='pagination')[-1].text)

    if number_pages > last_page:
        number_pages = last_page

    #Looping over the specified nupber of Pages:
    for p in range(1,number_pages,1):
        category_page = base_url+'/'+str(p)
        print('Parsing: ',category_page)
        page_news = single_page(category_page,p)
        
        #Saving to a CSV
        dict_to_csv(category_name+'.csv',page_news)
        
        #Time sleep
        time.sleep(10)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    return True
URL = 'https://www.investing.com/news/'
category_name = 'commodities-news'
base_url = URL+category_name
parsing_category_pages ('commodities-news',base_url,number_pages=5)
data = pd.read_csv('../working/commodities-news.csv')
data.head(100)
