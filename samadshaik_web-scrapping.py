from bs4 import BeautifulSoup

import requests

import pandas as pd
# Url for 

url = "https://boston.craigslist.org/search/npo"


npo_jobs = {}

job_no = 0

#A while loop which itters until it reaches the last page of the site

while True:

    # save the response from the site got using request library in a response library

    response = requests.get(url)

    # save the text in data variable

    data = response.text

    

    # convert the text to a soup object



    soup = BeautifulSoup(data,'html.parser')

    

    jobs = soup.find_all('p',{'class':'result-info'})

    

    for job in jobs:

        

        title = job.find('a',{'class':'result-title'}).text

        location_tag = job.find('span',{'class':'result-hood'})

        location = location_tag.text[2:-1] if location_tag else "N/A"

        date = job.find('time', {'class': 'result-date'}).text

        link = job.find('a', {'class': 'result-title'}).get('href')

        

        job_response = requests.get(link)

        job_data = job_response.text

        job_soup = BeautifulSoup(job_data, 'html.parser')

        job_description = job_soup.find('section',{'id':'postingbody'}).text

        job_attributes_tag = job_soup.find('p',{'class':'attrgroup'})

        job_attributes = job_attributes_tag.text if job_attributes_tag else "N/A"

        

        job_no+=1

        npo_jobs[job_no] = [title, location, date, link, job_attributes, job_description]

        

        

#       print('Job Title:', title, '\nLocation:', location, '\nDate:', date, '\nLink:', link,"\n", job_attributes, '\nJob Description:', job_description,'\n---')

        

    url_tag = soup.find('a',{'title':'next page'})

    if url_tag.get('href'):

        url= 'https://boston.craigslist.org' + url_tag.get('href')

        print(url)

    else:

        break
print("Total Jobs:", job_no)
npo_jobs_df = pd.DataFrame.from_dict(npo_jobs, orient = 'index', columns = ['Job Title','Location','Date', 'Link', 'Job Attributes', 'Job Description'])
npo_jobs_df.size
npo_jobs_df.to_csv('npo_jobs.csv')
from bs4 import BeautifulSoup

import requests 

import pandas as pd



datadictionary = {}



# coding for Crowling a web site



# Url of Monster India /

url = "https://www.car.com/sports-cars/"

# save the response from the site got using request library in a response

response = requests.get(url)



    # save the text in data variable

data = response.text

    

     # convert the text to a soup object

soup = BeautifulSoup(data,'html.parser')



    # Find the block of content in our case it is a dic with class listItem__h2--grid

body = soup.find('ul',{'class':'trim-tile-container'})

frame = body.find_all('li',{'class':'tile-item reflow'})

i = 1

for a in frame:

    div1 = a.find('div',{'class':'caption-1'})

    div2 = div1.find('div')

    h2 = div2.find('h2')

    car_name = h2.small.text

    car_model = h2.span.text

    

    

    div3 = a.find('div',{'class':'caption-2'})

    div4 = div3.find('div')

    car_price = div4.span.text

    

    print(car_name+"  "+car_model + " "+ car_price)

    datadictionary[i] = [car_name,car_model,car_price]

    i=i+1

    cars = pd.DataFrame.from_dict(datadictionary, orient = 'index',columns=['Car','Model','Price'])

    cars.to_csv('cars.csv')