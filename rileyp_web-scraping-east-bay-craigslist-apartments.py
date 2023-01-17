#import get to call a get request on the site

from requests import get



#get the first page of the east bay housing prices

response = get('https://sfbay.craigslist.org/search/eby/apa?hasPic=1&availabilityMode=0') #get rid of those lame-o's that post a housing option without a pic using their filter



from bs4 import BeautifulSoup

html_soup = BeautifulSoup(response.text, 'html.parser')



#get the macro-container for the housing posts

posts = html_soup.find_all('li', class_= 'result-row')

print(type(posts)) #to double check that I got a ResultSet

print(len(posts)) #to double check I got 120 (elements/page)
#grab the first post

post_one = posts[0]
#grab the price of the first post

post_one_price = post_one.a.text

post_one_price.strip()
#grab the time of the post in datetime format to save on cleaning efforts

post_one_time = post_one.find('time', class_= 'result-date')

post_one_datetime = post_one_time['datetime']
#title is a and that class, link is grabbing the href attribute of that variable

post_one_title = post_one.find('a', class_='result-title hdrlnk')

post_one_link = post_one_title['href']



#easy to grab the post title by taking the text element of the title variable

post_one_title_text = post_one_title.text
#grabs the whole segment of housing details. We will need missing value handling in the loop as this kind of detail is not common in posts

#the text can be split, and we can use indexing to grab the elements we want. number of bedrooms is the first element.

#sqft is the third element



post_one_num_bedrooms = post_one.find('span', class_ = 'housing').text.split()[0]



post_one_sqft = post_one.find('span', class_ = 'housing').text.split()[2][:-3] #cleans the ft2 at the end
#the neighborhood is grabbed by finding the span class 'result-hood' and pulling the text element from that

post_one_hood = posts[0].find('span', class_='result-hood').text



import re

re.findall(r"[\w]+", testing)[0].title() #this takes only letters, thereby bypassing all the possible delimiters (,, /, -, etc.)





#post_hood = posts[100].find('span', class_= 'result-hood').text

#post_hood = re.findall(r"[\w]+", posts[100].find('span', class_= 'result-hood').text)[0].title()

#post_hood



hood = posts[28].find('span', class_= 'result-hood').text.strip().split()[0:2]



" ".join(hood).title()
#build out the loop

from time import sleep

import re

from random import randint #avoid throttling by not sending too many requests one after the other

from warnings import warn

from time import time

from IPython.core.display import clear_output

import numpy as np



#find the total number of posts to find the limit of the pagination

results_num = html_soup.find('div', class_= 'search-legend')

results_total = int(results_num.find('span', class_='totalcount').text) #pulled the total count of posts as the upper bound of the pages array



#each page has 119 posts so each new page is defined as follows: s=120, s=240, s=360, and so on. So we need to step in size 120 in the np.arange function

pages = np.arange(0, results_total+1, 120)



iterations = 0



post_timing = []

post_hoods = []

post_title_texts = []

bedroom_counts = []

sqfts = []

post_links = []

post_prices = []



for page in pages:

    

    #get request

    response = get("https://sfbay.craigslist.org/search/eby/apt?" 

                   + "s=" #the parameter for defining the page number 

                   + str(page) #the page number in the pages array from earlier

                   + "&hasPic=1"

                   + "&availabilityMode=0")



    sleep(randint(1,5))

     

    #throw warning for status codes that are not 200

    if response.status_code != 200:

        warn('Request: {}; Status code: {}'.format(requests, response.status_code))

        

    #define the html text

    page_html = BeautifulSoup(response.text, 'html.parser')

    

    #define the posts

    posts = html_soup.find_all('li', class_= 'result-row')

        

    #extract data item-wise

    for post in posts:



        if post.find('span', class_ = 'result-hood') is not None:



            #posting date

            #grab the datetime element 0 for date and 1 for time

            post_datetime = post.find('time', class_= 'result-date')['datetime']

            post_timing.append(post_datetime)



            #neighborhoods

            post_hood = post.find('span', class_= 'result-hood').text

            post_hoods.append(post_hood)



            #title text

            post_title = post.find('a', class_='result-title hdrlnk')

            post_title_text = post_title.text

            post_title_texts.append(post_title_text)



            #post link

            post_link = post_title['href']

            post_links.append(post_link)

            

            #removes the \n whitespace from each side, removes the currency symbol, and turns it into an int

            post_price = int(post.a.text.strip().replace("$", "")) 

            post_prices.append(post_price)

            

            if post.find('span', class_ = 'housing') is not None:

                

                #if the first element is accidentally square footage

                if 'ft2' in post.find('span', class_ = 'housing').text.split()[0]:

                    

                    #make bedroom nan

                    bedroom_count = np.nan

                    bedroom_counts.append(bedroom_count)

                    

                    #make sqft the first element

                    sqft = int(post.find('span', class_ = 'housing').text.split()[0][:-3])

                    sqfts.append(sqft)

                    

                #if the length of the housing details element is more than 2

                elif len(post.find('span', class_ = 'housing').text.split()) > 2:

                    

                    #therefore element 0 will be bedroom count

                    bedroom_count = post.find('span', class_ = 'housing').text.replace("br", "").split()[0]

                    bedroom_counts.append(bedroom_count)

                    

                    #and sqft will be number 3, so set these here and append

                    sqft = int(post.find('span', class_ = 'housing').text.split()[2][:-3])

                    sqfts.append(sqft)

                    

                #if there is num bedrooms but no sqft

                elif len(post.find('span', class_ = 'housing').text.split()) == 2:

                    

                    #therefore element 0 will be bedroom count

                    bedroom_count = post.find('span', class_ = 'housing').text.replace("br", "").split()[0]

                    bedroom_counts.append(bedroom_count)

                    

                    #and sqft will be number 3, so set these here and append

                    sqft = np.nan

                    sqfts.append(sqft)                    

                

                else:

                    bedroom_count = np.nan

                    bedroom_counts.append(bedroom_count)

                

                    sqft = np.nan

                    sqfts.append(sqft)

                

            #if none of those conditions catch, make bedroom nan, this won't be needed    

            else:

                bedroom_count = np.nan

                bedroom_counts.append(bedroom_count)

                

                sqft = np.nan

                sqfts.append(sqft)

            #    bedroom_counts.append(bedroom_count)

                

            #    sqft = np.nan

            #    sqfts.append(sqft)

                

    iterations += 1

    print("Page " + str(iterations) + " scraped successfully!")



print("\n")



print("Scrape complete!")
import pandas as pd



eb_apts = pd.DataFrame({'posted': post_timing,

                       'neighborhood': post_hoods,

                       'post title': post_title_texts,

                       'number bedrooms': bedroom_counts,

                        'sqft': sqfts,

                        'URL': post_links,

                       'price': post_prices})

print(eb_apts.info())

eb_apts.head(10)
#first things first, drop duplicate URLs because people are spammy on Craigslist. 

#Let's see how many uniqe posts we really have.

eb_apts = eb_apts.drop_duplicates(subset='URL')

len(eb_apts.drop_duplicates(subset='URL'))



#make the number bedrooms to a float (since np.nan is a float too)

eb_apts['number bedrooms'] = eb_apts['number bedrooms'].apply(lambda x: float(x))



#convert datetime string into datetime object to be able to work with it

from datetime import datetime



eb_apts['posted'] = pd.to_datetime(eb_apts['posted'])



#Looking at what neighborhoods there are with eb_apts['neighborhood'].unique() allowed me to see what

#I needed to deal with in terms of cleaning those.



#remove the parenthesis from the left and right of the neighborhoods

eb_apts['neighborhood'] = eb_apts['neighborhood'].map(lambda x: x.lstrip('(').rstrip(')'))



#titlecase them

eb_apts['neighborhood'] = eb_apts['neighborhood'].str.title()



#just take the first name of the neighborhood list, splitting on the '/' delimiter

eb_apts['neighborhood'] = eb_apts['neighborhood'].apply(lambda x: x.split('/')[0])



#fix one-offs that

eb_apts['neighborhood'].replace('Belmont, Ca', 'Belmont', inplace=True)

eb_apts['neighborhood'].replace('Hercules, Pinole, San Pablo, El Sob', 'Hercules', inplace=True)



#remove whitespaces

eb_apts['neighborhood'] = eb_apts['neighborhood'].apply(lambda x: x.strip())



#save the clean data

eb_apts.to_csv("eb_apts_1642_Jan_2_19_clean.csv", index=False)
eb_apts.info()
import pandas as pd, numpy as np

#I think Craigslist blocked me. Reload dataframe via file for now

eb_apts = pd.read_csv("eb_apts_1642_Jan_2_19_clean.csv")
#start to look at the distributions

from matplotlib import figure

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.figure(figsize=(10, 6))

plt.hist(eb_apts['price'], edgecolor='black');

plt.xlabel("Price")

plt.ylabel('Count')

plt.title("Distribution of Prices");
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',

          'figure.figsize': (15, 5),

         'axes.labelsize': 'x-large',

         'axes.titlesize':'x-large',

         'xtick.labelsize':'x-large',

         'ytick.labelsize':'x-large'}

pylab.rcParams.update(params)



plt.figure(figsize=(12, 8))

sns.scatterplot(x='price', y='sqft', hue='number bedrooms', palette='summer', x_jitter=True, y_jitter=True, s=125, data=eb_apts.dropna())

plt.legend(fontsize=12)

plt.xlabel("Price", fontsize=18)

plt.ylabel("Square Footage", fontsize=18);

plt.title("Price vs. Square Footage Colored by Number of Bedrooms", fontsize=18);
plt.figure(figsize=(12, 8))

sns.regplot(x='price', y='sqft', data=eb_apts.dropna());

plt.title('Price vs. Square Footage Regression Plot');

plt.xlabel("Price (USD)");

plt.ylabel("Square Feet");
eb_apts.corr()
#group by neighborhood

eb_apts.groupby('neighborhood').mean()
eb_apts['neighborhood'] == 'North Oakland'
#sort price to find cheapest

eb_apts.groupby('neighborhood').mean()['price'].sort_values()
plt.figure(figsize=(15,10))

params = {'legend.fontsize': 'x-large',

          'figure.figsize': (15, 5),

         'axes.labelsize': 'x-large',

         'axes.titlesize':'x-large',

         'xtick.labelsize':'x-large',

         'ytick.labelsize':'x-large'}

pylab.rcParams.update(params)



sns.boxplot(x='neighborhood', y='price', data=eb_apts)

plt.xlabel("Neighborhood");

plt.xticks(rotation=75)

plt.ylabel("Price USD");

plt.title("Prices by Neighborhood - Boxplots");