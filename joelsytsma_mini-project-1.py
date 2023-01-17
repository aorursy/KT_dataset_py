# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re #allowing regex to happen. I didn't end up using regex in the final version, but at one point I was trying it.

import matplotlib.pyplot as plt #importing matplotlib to create the final draft



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print("Please enter a keyword from the title of a book you want to learn about, try hobbit") #some instructional text

uTitle = input() #setting the variable that will inherit the user input
#I'm going to use the Goodreads CSV to make sure that I get a good book title to put in the API

data = pd.read_csv("../input/goodreadsbooks/books.csv", error_bad_lines=False) #setting a variable that will hold the CSV. Omitting bad lines of data.

results = {} #creating a blank dictionary that I will use to hold all the results in from the query I'm about to run

gdReviews= {}#creating a blank dictionary that will hold review numbers

x= 0

#After examining the goodreadsbooks file in excel, I saw that the CSV error lines had extra commas. 

#Since there are so few of these lines, I decided to omit those lines rather than fix them.



for titles in data['title'].str.lower(): #looping through the title column in the Goodreads CSV. Also putting the titles in lower case.

    x= x+1 #creating an iterator variable as a way to ID each result. 

    for words in titles.split(): #splitting the titles by word

        if words == uTitle: #searching to see if each title contains the user inputted keyword

            gdRating=data.loc[x,'average_rating'] #if it contains the keyword, then add the overall rating to this dictionary.

            results[x]=[titles] #adding the book title to a dictionary.

            gdReviews[x]=[gdRating]

print("Please select the number associated with book you want to know about:")

for bookID in results:

    titles2= results[bookID][0]

    print(bookID,":",titles2)

    

#plt.bar(*zip(*gdRating.items())) I've tried all sorts of things to plot the reviews that are associated with the books

#plt.show() I keep getting the error numpy.float64' object has no attribute 'items'. I don't know how fix it.



userChoice = input()



#need to strip symbols from the title for it to work in the api.



userChoice=int(userChoice)

z= results[userChoice][0]

whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')

answer = ''.join(filter(whitelist.__contains__, z))

print(answer)



from kaggle_secrets import UserSecretsClient #importing the Kaggle secrets keeper

user_secrets = UserSecretsClient() #setting up the function "UserSecretsClient" to be called when the user_secrets variable is called

IDBkey = user_secrets.get_secret("dreamBooks") #finally setting the variable that will call that secret

import requests #Allows kaggle to search for URL

import urllib.request, json #converting things to json

from pandas.io.json import json_normalize #

from pandas import DataFrame





url = ('https://idreambooks.com/api/books/reviews.json?q={%s}&key=2c0065017da38949651aa7bf12a2031ab3ba8492' %answer) #defining my URL

r = requests.get(url) #settting r to retrieve that URL



dictr = r.json() #putting the JSON I receive back from the URL inside this variable

#v4

import random #this allows me to select a random review snippet later on

starAvg = [] #creating an empty list to house the avg. star ratings I will get later

pos_review = {} #creating a dictionary that will hold positive review text snippets

neg_review = {} #doing the same for negative reviews

negative = 0 #setting a number that will change everytime a book gets a negative review

positive= 0 #doing the same for positive

numberReviews= 0 #counting the number of reviews a book has gotten

for values in dictr: #looping through the JSON and looking at each key in it.

    values1=dictr[values] #pulling out the value that is for each key and assigning it to it's own variable

    if isinstance(values1, int): #since I know that some of the keys are dictionaries and some are values, I'm using a conditional statement here to check for that.

        print('There is',values1,'book by this name')  #I know that if an integer comes up, it means that a book has come up with that name.

    else:  #if it's not an integer, than it's a dictionary. If I didn't know what the json looked like, I would write an elif statement for every datatype but since I know, I'll just check for dicts here.

        for values2 in values1: #looping through every key that is in the dictionary I found with this else statement

            values3= values1[values2] #using that key I just found to extract the values for each new key that comes up

            if isinstance(values3,list): #if it's a list then move on! Otherwise skip to the else

                numberReviews = len(values3) #the lenghth of the list is how many reviews there are. Assign it to the list we created earlier

                for cReviews in values3: #It starts looping through the list (i know there is a list in the data)

                    for cReviews1 in cReviews: #since each list item is a dict we need to start looping through each dict.

                        if cReviews1 == "star_rating": #if the key value that is found matches "star rating" then...

                            starAvg.append(cReviews[cReviews1]) #add the information about that star to a dictionary.

                        elif cReviews1 == "pos_or_neg": #if the key value is pos_or_neg" then...

                            if cReviews[cReviews1]== "Positive": #check if the value is positive...

                                positive = positive+1 #if it is positive, then iterate up the positive variable and...

                                pos_review[cReviews["source"]] = cReviews['snippet'] #pull the review snippet from the dictionary into a new dictionary that collects positive reviews

                            elif cReviews[cReviews1]== "Negative": #do the same thing as above. Only for negative reviews.

                                negative = negative+1

                                neg_review[cReviews["source"]] = cReviews['snippet']

            else: #going way back. If a value in a dictionary is not a list then print do this segment

                if values2 == "title": #check the value of the key and print the corresponding statement if true.

                    print("the",values2,"is",values3)

                elif values2 == "author":

                    print("the",values2,"is",values3)#check the value of the key and print the corresponding statement if true.

                elif values2 == "rating":

                    print("the average",values2,"out of 100 is",values3) #check the value of the key and print the corresponding statement if true.

if len(pos_review)== 0: #i need to account for an empty dictionary if a book gets no positive reviews. This is doing that.

    emptyPos= "There are no positive reviews" #pring this if the lenghth of my dictionary is zero

else:

    x=random.choice(list(pos_review)) #if it's not zero. print this. Since there can't be negative reviews I feel okay not accounting for that.

if len(neg_review)== 0:

    emptyNeg= "There are no negative reviews"

else:

    y=random.choice(list(neg_review))

#the following section of code just prints all the variables and lists I've been parsing up above.

print("It has",numberReviews,"reviews")

print("It received the following star ratings",starAvg)

print("There are",positive,"positive reviews and",negative,"negative reviews")

if len(pos_review) == 0:

    print("there are no positive reviews")

else:

    print("A randomly selected positive review from",x,"says","-",pos_review[x]) #need to account for empty states

if len(neg_review) == 0:

    print("there are no negative reviews")

else:

    print("A randomly selected negative review from",y,"says","-",neg_review[y]) #need to account for empty states
#a poor attempt to plot the star rating. I don't know why it's picking up zeroes in the first two spots of the list.

plt.plot(starAvg)

plt.ylabel('Star counts')

plt.show()
#one version of the code where I set dictr to be the JSON that came back from the API

dictr = {'total_results': 1, 'book': {'title': 'Wool Omnibus Edition', 'sub_title': '(Wool 1 - 5)', 'author': 'Hugh Howey', 'review_count': 5, 'rating': 72, 'to_read_or_not': 'https://idreambooks.com/images/api/rating-icons/positive-small.png', 'detail_link': 'https://idreambooks.com/Wool-Omnibus-Edition-Wool-1-5--by-Hugh-Howey/reviews/6143', 'genre': 'Fiction', 'pages': 530, 'release_date': '2012-01-25', 'critic_reviews': [{'snippet': "Some elements of Wool work brilliantly: the first two sections are frightening, intriguing and mysterious. Holston, the old mayor Jahns and Holston's deputy, Marnes, are unusual, fully realised characters...Other elements don't work so well. It's partly down, I think, to the way the novel developed.", 'source': 'Guardian', 'review_link': 'http://www.guardian.co.uk/books/2013/jan/09/wool-by-hugh-cowey-review', 'pos_or_neg': 'Negative', 'star_rating': 3.0, 'review_date': '2013-01-10', 'smiley_or_sad': 'http://idreambooks.com/images/api/review-rating-icons/above_average_cr_small.png', 'source_logo': 'https://idreambooks.com/images/libraries_landing_page/guardian.png'}, {'snippet': 'A marvelously chilling and beautifully written series, set in a post-apocalyptic world that lives and breathes. Howey does a magnificent job of bringing the Silo to life...', 'source': 'IndieReader', 'review_link': 'http://indiereader.com/2012/04/wool-omnibus-edition/', 'pos_or_neg': 'Positive', 'star_rating': 5.0, 'review_date': '2013-05-24', 'smiley_or_sad': 'http://idreambooks.com/images/api/review-rating-icons/excellent_cr_small.png'}, {'snippet': '...it was an interesting ride but in the end I didn’t believe the answers. It’s the sort of book where everything makes sense as you go along, but then as you think it out later you start noticing how improbable it all is.', 'source': 'Patheos', 'review_link': 'http://www.patheos.com/blogs/happycatholicbookshelf/2012/09/wool/', 'pos_or_neg': 'Negative', 'star_rating': 3.0, 'review_date': '2012-09-19', 'smiley_or_sad': 'http://idreambooks.com/images/api/review-rating-icons/above_average_cr_small.png'}, {'snippet': 'Here is a non-traditional author who can stand proudly in the company of traditionally published writers. Hugh Howey has arrived, and his arrival heralds a new day for indie authors.', 'source': 'Wired', 'review_link': 'http://www.wired.com/geekdad/2012/03/geekdad-book-review-the-wool-omnibus-by-hugh-howey/', 'pos_or_neg': 'Positive', 'star_rating': 5.0, 'review_date': '2012-03-29', 'smiley_or_sad': 'http://idreambooks.com/images/api/review-rating-icons/excellent_cr_small.png'}, {'snippet': "...Howey's immaturity as a writer, especially the bland characters and conflict reminiscent of B-movies, overshadows his intriguing world.", 'source': 'Publishers Weekly', 'review_link': 'http://www.publishersweekly.com/978-1-4767-3511-5', 'pos_or_neg': 'Negative', 'star_rating': 2.0, 'review_date': '2013-04-01', 'smiley_or_sad': 'http://idreambooks.com/images/api/review-rating-icons/poor_cr_small.png', 'source_logo': 'https://idreambooks.com/images/libraries_landing_page/publishersweekly.png'}]}}

starAvg = [ ]

pos_review = {}

neg_review = {}

negative = 0

positive= 0

numberReviews= 0

for values in dictr:

    values1=dictr[values]

    if isinstance(values1, int):

        print('There is',values1,'book by this name')

    else:

        for values2 in values1:

            values3= values1[values2]

            if isinstance(values3,list):

                numberReviews = len(values3)

                for cReviews in values3:

                    for cReviews1 in cReviews:

                        if cReviews1 == "star_rating":

                            starAvg.append(cReviews[cReviews1])

                        elif cReviews1 == "pos_or_neg":

                            if cReviews[cReviews1]== "Positive":

                                positive = positive+1

                                pos_review[cReviews["source"]] = cReviews['snippet']

                            elif cReviews[cReviews1]== "Negative":

                                negative = negative+1

                                neg_review[cReviews["source"]] = cReviews['snippet']

                        #print(cReviews[cReviews1])

            else:

                if values2 == "title":

                    print("the",values2,"is",values3)

                elif values2 == "author":

                    print("the",values2,"is",values3)

print("It has",numberReviews,"reviews")

print("There are",positive,"positive reviews and",negative,"negative reviews")

print("An example positive review, ")

#v3- another version of the code

dictr = {'total_results': 1, 'book': {'title': 'Wool Omnibus Edition', 'sub_title': '(Wool 1 - 5)', 'author': 'Hugh Howey', 'review_count': 5, 'rating': 72, 'to_read_or_not': 'https://idreambooks.com/images/api/rating-icons/positive-small.png', 'detail_link': 'https://idreambooks.com/Wool-Omnibus-Edition-Wool-1-5--by-Hugh-Howey/reviews/6143', 'genre': 'Fiction', 'pages': 530, 'release_date': '2012-01-25', 'critic_reviews': [{'snippet': "Some elements of Wool work brilliantly: the first two sections are frightening, intriguing and mysterious. Holston, the old mayor Jahns and Holston's deputy, Marnes, are unusual, fully realised characters...Other elements don't work so well. It's partly down, I think, to the way the novel developed.", 'source': 'Guardian', 'review_link': 'http://www.guardian.co.uk/books/2013/jan/09/wool-by-hugh-cowey-review', 'pos_or_neg': 'Negative', 'star_rating': 3.0, 'review_date': '2013-01-10', 'smiley_or_sad': 'http://idreambooks.com/images/api/review-rating-icons/above_average_cr_small.png', 'source_logo': 'https://idreambooks.com/images/libraries_landing_page/guardian.png'}, {'snippet': 'A marvelously chilling and beautifully written series, set in a post-apocalyptic world that lives and breathes. Howey does a magnificent job of bringing the Silo to life...', 'source': 'IndieReader', 'review_link': 'http://indiereader.com/2012/04/wool-omnibus-edition/', 'pos_or_neg': 'Positive', 'star_rating': 5.0, 'review_date': '2013-05-24', 'smiley_or_sad': 'http://idreambooks.com/images/api/review-rating-icons/excellent_cr_small.png'}, {'snippet': '...it was an interesting ride but in the end I didn’t believe the answers. It’s the sort of book where everything makes sense as you go along, but then as you think it out later you start noticing how improbable it all is.', 'source': 'Patheos', 'review_link': 'http://www.patheos.com/blogs/happycatholicbookshelf/2012/09/wool/', 'pos_or_neg': 'Negative', 'star_rating': 3.0, 'review_date': '2012-09-19', 'smiley_or_sad': 'http://idreambooks.com/images/api/review-rating-icons/above_average_cr_small.png'}, {'snippet': 'Here is a non-traditional author who can stand proudly in the company of traditionally published writers. Hugh Howey has arrived, and his arrival heralds a new day for indie authors.', 'source': 'Wired', 'review_link': 'http://www.wired.com/geekdad/2012/03/geekdad-book-review-the-wool-omnibus-by-hugh-howey/', 'pos_or_neg': 'Positive', 'star_rating': 5.0, 'review_date': '2012-03-29', 'smiley_or_sad': 'http://idreambooks.com/images/api/review-rating-icons/excellent_cr_small.png'}, {'snippet': "...Howey's immaturity as a writer, especially the bland characters and conflict reminiscent of B-movies, overshadows his intriguing world.", 'source': 'Publishers Weekly', 'review_link': 'http://www.publishersweekly.com/978-1-4767-3511-5', 'pos_or_neg': 'Negative', 'star_rating': 2.0, 'review_date': '2013-04-01', 'smiley_or_sad': 'http://idreambooks.com/images/api/review-rating-icons/poor_cr_small.png', 'source_logo': 'https://idreambooks.com/images/libraries_landing_page/publishersweekly.png'}]}}

starAvg = []

pos_review = {}

neg_review = {}

negative = 0

positive= 0

numberReviews= 0

for values in dictr:

    values1=dictr[values]

    if isinstance(values1, int):

        print('There is',values1,'book by this name')

    else:

        for values2 in values1:

            values3= values1[values2]

            if isinstance(values3,list):

                numberReviews = len(values3)

                for cReviews in values3:

                    for cReviews1 in cReviews:

                        if cReviews1 == "star_rating":

                            starAvg.append(cReviews[cReviews1])

                        elif cReviews1 == "pos_or_neg":

                            if cReviews[cReviews1]== "Positive":

                                positive = positive+1

                                pos_review[cReviews["source"]] = cReviews['snippet']

                            elif cReviews[cReviews1]== "Negative":

                                negative = negative+1

                                neg_review[cReviews["source"]] = cReviews['snippet']

                        #print(cReviews[cReviews1])

            else:

                if values2 == "title":

                    print("the",values2,"is",values3)

                elif values2 == "author":

                    print("the",values2,"is",values3)

x=pos_review.popitem()

y=neg_review.popitem()

print("It has",numberReviews,"reviews")

print("It received the following star ratings",starAvg)

print("There are",positive,"positive reviews and",negative,"negative reviews")

print("An example positive review from",x)

print("An example negative review from",y)