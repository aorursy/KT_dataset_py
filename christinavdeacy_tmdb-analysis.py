import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import matplotlib as mpl

import os

import ast

#print(os.listdir("../input"))

#----------------------------------------------

# READING IN THE .csv FILES

movie_data = pd.read_csv("../input/movies_metadata.csv", low_memory=False)

rate_data_small = pd.read_csv("../input/ratings_small.csv")

#keyword_data = pd.read_csv("../input/keywords.csv")

link_data_small = pd.read_csv("../input/links_small.csv")

#link_data = pd.read_csv("../input/links.csv")

rate_data = pd.read_csv("../input/ratings.csv", low_memory=False)

credit_data = pd.read_csv("../input/credits.csv")

#----------------------------------------------

movie_data.rename(columns={'id':'tmdbId'},inplace=True)

#pd.set_option("display.max_rows", 15)



#---FUNCTIONS---

#-------------------

#function that takes a movie title, finds out the tmdbId for it, then converts that to its movieId using the links 

#file 

def tmdb_to_movieID(movie_title):

    a = movie_data.loc[movie_data.title == movie_title].tmdbId#.values[0]

    if a.empty:

        #print("That movie does not exist in the database.")

        return 0

    elif a.shape[0] == 1:

        #print('!',a,'!')

        a = int(a) 

        b = link_data_small.loc[link_data_small.tmdbId == a].movieId

        if b.empty:

            return 0

        b = int(b)

        #print('#',b,'#')

        return b

    else:

        idnum = list()

        for index, row in a.iteritems():

            #print('£', row.title, '£')

            #print('$',row,'$')

            a = int(row) 

            b = link_data_small.loc[link_data_small.tmdbId == a].movieId

            if b.empty:

                continue

            b = int(b)

            #print('+',b,'+')

            idnum.append(b)

        return idnum

#-------------------    

#function that finds out the average rating for a movie, when given a movie id number

def ave_rating(idnum):

    c = rate_data_small.loc[rate_data_small.movieId == idnum]

    r = round(c.rating.mean(),2)

    #print("Average Rating is : ", r)

    return r

#-------------------

#def sep_genres(id_and_genres):

def sep_genres(movie_row):

    id_and_genres = movie_row.genres#.values[0]

    idg_list = ast.literal_eval(id_and_genres)

    gen_list = set()

    while idg_list:

        indiv_genre = idg_list[0]

        #print(indiv_genre)

        #print(indiv_genre['name'])

        gen_list.add(indiv_genre['name'])

        idg_list.remove(idg_list[0])

    return gen_list

#-------------------

print('Finished Loading')
#num_of_rows = movie_data.shape[0]

#print(num_of_rows)



#MIDNIGHT MAN

#movie_data.loc[movie_data.tmdbId == '1997-08-20']
class actor_rev: #actor is the key, revenue is the 'value'

    

    #initialises a dictionary from within the class, 'self' means that specific instance of the class, meaning

    #the specific dictionary being used... MEANING YOU NEED TO PUT SELF BEFORE ALL THE actdict BITS

    def __init__(self, actdict = None):

        if actdict is None:

            actdict = {}

        self.actdict = actdict

        

    #adds a new actor

    def add_actor(self, actor):

        if actor not in self.actdict:

            self.actdict[actor] = []

    

    #adds a new revenue to a pre-existing actor

    def add_revenue(self, revenue):

        revenue = set(revenue)

        #pairs the actor and the revenue together in a tuple

        (a, r) = tuple(revenue)

        #if the actor is already in the dictionary, add the revenue to it

        if a in self.actdict:

            self.actdict[a].append(r)      

        

        #return the list of the actors in the dictionary

    def get_actors(self):

        return list(self.actdict.keys())

        

        #iterates through the dictionary to get the revenue information

    def get_revs(self):

        revenue = []

        for a in self.actdict:

            for r in self.actdict[a]:

                if {a, r} not in revenue:

                    revenue.append({a, r})

        return revenue

###################################################################################################

print('Finished Loading')
actor_rev_dict = {}

ard = actor_rev(actor_rev_dict)



for index, row in credit_data.iterrows():

    cinfo = ast.literal_eval(row.cast)   

    a = row.id

    a = str(int(a))

    #print('+', a, '+')

    b = movie_data.loc[movie_data.tmdbId == a].revenue.values[0]

    #print(b)

    #b = int(b)

    #print(type(b))

    if np.float(b) == 0:

        #print('**************************')

        continue

    #print('@@@@@@@@@@@@@@@@@', b)

    #iterate through the cast list, and append the cast members name and the revenue the film earnt

    while cinfo:

        indiv_credit = cinfo[0]

        #print(indiv_credit)

        #print(indiv_credit['name'])

        #cred_list.append(indiv_credit['name'])

        n = indiv_credit['name']

        #print(n)

        ard.add_actor(n)

        ard.add_revenue({n, b})

        cinfo.remove(cinfo[0])



print(actor_rev_dict)    
#movie_data.loc[movie_data.tmdbId == '862'].revenue



data = credit_data.sample(frac=0.0005, replace=True, random_state=1)

print(data)

data = pd.DataFrame(data)





#data = credit_data.iloc[0]

#data = pd.DataFrame(data)

#print(data)



actor_rev_dict = {}

ard = actor_rev(actor_rev_dict)



for index, row in data.iterrows():

    cinfo = ast.literal_eval(row.cast) 

    

    a = row.id

    a = str(int(a))

    #print('+', a, '+')

    b = movie_data.loc[movie_data.tmdbId == a].revenue.values[0]

    #print(b)

    #b = int(b)

    #print(type(b))

    if np.float(b) == 0:

        #print('**************************')

        continue  

    print(row)  

    #iterate through the cast list, and append the cast members name and the revenue the film earnt

    while cinfo:

        indiv_credit = cinfo[0]

        #print(indiv_credit)

        #print(indiv_credit['name'])

        #cred_list.append(indiv_credit['name'])

        n = indiv_credit['name']

        ard.add_actor(n)

        ard.add_revenue({n, b})

        cinfo.remove(cinfo[0])



print(actor_rev_dict)    
from datetime import datetime

import re

num_to_writ_month = {1:'January', 2:'February', 3: 'March',

                    4:'April', 5:'May', 6:'June', 7:'July',

                     8:'August', 9:'September', 10:'October',

                     11:'November', 12:'December'}



#m = movie_data.loc[:, ('title', 'release_date', 'revenue')]





filter = movie_data.revenue != 0.0

mov_filtered = movie_data[filter]

mov_filtered = pd.DataFrame(mov_filtered)

months_ave_revenue = {1:0, 2:0, 3:0,

                    4:0, 5:0, 6:0, 7:0,

                     8:0, 9:0, 10:0,

                     11:0, 12:0}

mon = input("Enter a month: ")

mon = int(mon)

#num_mov = number of movies released in that month, rev_total = total revenue for movies

#released in that month

num_mov = 0

rev_total = 0

for index, row in mov_filtered.iterrows():

    #print(x)

    #print(type(x))

    #if not isinstance(x,str):

    rel_dat = row.release_date

    date_format = '....-..-..'

    if not isinstance(rel_dat, str) or not re.match(date_format, rel_dat):

        continue

    get_date = datetime.strptime(rel_dat, '%Y-%m-%d').month

    #print('~', get_date, '~')

    if get_date != mon:

        continue

    num_mov += 1

    rev_total += row.revenue

    

    #a = months_ave_revenue.get(mon)

    #print(a)

    #a = int(a)

    #a += row.revenue

    #months_ave_revenue[mon] = a

    #print(months_ave_revenue.get(mon))

    

    #print(num_mov, ',', row.revenue)

    #print('*', rev_total)



#print(rev_total)

a = num_to_writ_month.get(mon)

ave_rev =  round((rev_total/num_mov), 2)

print('Average revenue for movies released in', a, 'is £',ave_rev)

#print(months_ave_revenue)
credit_row = credit_data



cinfo = credit_row.cast[0]#.values[0]

idg_list = ast.literal_eval(cinfo)

cred_list = list()

while idg_list:

    indiv_credit = idg_list[0]

    #print(indiv_credit)

    #print(indiv_credit['name'])

    cred_list.append(indiv_credit['name'])

    idg_list.remove(idg_list[0])

    

print(cred_list)
#genre = input("Enter a movie genre: ")

#movie_data.loc[movie_data.genres.name == 'Action']

#a = pd.DataFrame(movie_data.genres)

#b = pd.Series(movie_data.genres[0])



movie = input("Enter a movie: ")

movie = movie.title()



idnum = tmdb_to_movieID(movie)

#print('^',idnum)

#print(type(idnum))



#if idnum == type(list()):

if isinstance(idnum, list):

        #print(':)')

    for i in idnum:

        b = ave_rating(i)

        print('ave rating = ', b)

else:# idnum != type(list()):

    if (idnum != 0):

        a = ave_rating(idnum)

        print('ave rating = ', a)

##Given a genre, prints out all the movies with a ave user rating of higher that 4.5



#movie = input("Enter a movie: ")

#g = 'Drama'

#idnum = tmdb_to_movieID(movie)

#movie_row = movie_data.loc[movie_data.title == movie]

#print(movie_row.genres)

#movie_gen = sep_genres(movie_row)

#if g in movie_gen:

#        print('Huzzah.')



#takes an input (a genre) - no error checking for a genre not in the database    

g = input("Enter a genre: ")

print("The top rated ",g," movies are: ")

for index, row in movie_data.iterrows():

    movie_gen = sep_genres(row)

    if g in movie_gen:

        #print('@', row.title, '@')

        idnum = tmdb_to_movieID(row.title)

        if isinstance(idnum, list):

            #print(':)', idnum)

            for i in idnum:

                r = ave_rating(i)

                #print(r)

                if r > 4.5:

                    print(row.title, '\nThe average rating is: ', r, '\n')

        else:# idnum != type(list()):

            if (idnum != 0):

                r = ave_rating(idnum)

                if r > 4.5:

                    print(row.title, '\nThe average rating is: ', r,'\n')

                #print(r)    



#movie = input("Enter a movie: ")

#idnum = tmdb_to_movieID(movie)

#if idnum != 0:

#    ave_rating(idnum)
y = movie_data.loc[movie_data.title == 'Sabrina']

num_of_rows = y.shape[0]

print(num_of_rows)

print(y)

for index, row in y.iterrows():

    print('£', row.title, '£')
#removes the movies with 0 as their budget and/or revenue field

#revenue_budget_info = movie_data.loc[(movie_data.budget != 0) & (movie_data.revenue != 0)]

#print(revenue_budget_info)



#37 rows = 0.005

#data = revenue_budget_info.sample(frac=0.0005, replace=True, random_state=1)

#print(data)





#N = 500

#x = data.budget

#y = data.revenue

#colors = (154,254,1)

#area = np.pi*3



#mpl.pyplot.scatter(x, y, alpha=0.5)

#mpl.pyplot.title('Scatter plot showing movie Budgets vs Revenue')

#mpl.pyplot.xlabel('x')

#mpl.pyplot.ylabel('y')

#mpl.axis([0, 100000000, 0, 100000000])

#mpl.pyplot.show()

#mpl.figure()
#No. of movies with each rating

#no_rat = rating_data.groupby('rating').rating.count()



rated_five = rate_data_small.loc[rate_data_small.rating == 5.0]

#five_point_movies = five_pts.loc[:,('movieId','rating')]

#link_movies = link_data_small.loc[:,('movieId','tmdbId')]

#id_movies = movie_data.loc[:,('id', 'title')]

#id_movies.rename(columns={'id':'tmdbId'},inplace=True)

movie_data.rename(columns={'id':'tmdbId'},inplace=True)



rf = pd.DataFrame(rated_five)



#used a set here instead of a list because if an element is already in the set, using .add() won't re-add it,

#leaving us with a unique list

#print(five_rated_movies)

five_rated_movies = set()



for index, row in rf.iterrows():

    a = row.movieId

    #print('!',a,'!')

    # .values[0] removes the index so just the id number is saved to the variable

    b = link_data_small.loc[link_data_small.movieId == a].tmdbId.values[0]

    #print('*',b,'*')

    #if the cell is blank, skip and move to the next one

    if np.isnan(b):

        continue

    #numpy float dynamically assigns the size of the string - so 4854.0 could be assigned to str as 

    #4854.0 or 4853.000000000 so easier to covert to int, then to str

    b = str(int(b))

    #print('&',b,'&')

    c = movie_data.loc[movie_data['tmdbId'] == b]

    if c.empty:

        continue

    #movieId:7502 tmdbId:191903 does not exist

    #print('@',c,'@')

    c = c.title.values[0]

    #print('?',c,'?')

    five_rated_movies.add(c)



print('The number of movies in this database which have been given a 5-point rating are: ', len(five_rated_movies), 

          '\nThey are as follows: \n', "\n".join(five_rated_movies))



#five_rated_movies.clear()