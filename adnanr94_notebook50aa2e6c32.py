# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
f = open('../input/movie_metadata.csv', 'r')

movie_data = f.read()

split_rows = movie_data.split('\n')

movie_data = []

for row in split_rows:

    split_row = row.split(',')

    movie_data.append(split_row)

print(movie_data)
# movie names

def first_elts(input_lst):

    elts = []

    for each in input_lst:

        elts.append(each[0])

    return elts



movie_names = first_elts(movie_data)

print(movie_names)
# movie made in given country

wonder_woman = ['Wonder Woman','Patty Jenkins','Color',141,'Gal Gadot','English','USA',2017]

def is_usa(inn):

    if inn[6]=='USA':

        return True

    else:

        return False

wonder_woman_usa = is_usa(wonder_woman)

print(wonder_woman_usa)
# functions with different arguements

wonder_woman = ['Wonder Woman','Patty Jenkins','Color',141,'Gal Gadot','English','USA',2017]



def is_usa(input_lst):

    if input_lst[6] == "USA":

        return True

    else:

        return False

def index_equals_str(lst,index,input_str):

    if lst[index] == input_str:

        return True

    else:

        return False

wonder_woman_in_color =  index_equals_str(wonder_woman,2,'Color')

print(wonder_woman_in_color)
# using optional arguement

def index_equals_str(input_lst,index,input_str):

    if input_lst[index] == input_str:

        return True

    else:

        return False

def counter(input_lst,header_row = False):

    num_elt = 0

    if header_row == True:

        input_lst = input_lst[1:len(input_lst)]

    for each in input_lst:

        num_elt = num_elt + 1

    return num_elt

def feature_counter(input_lst,index, input_str, header_row = False):

    num = 0

    if header_row == True:

        input_lst = input_lst[1:len(input_lst)]

    for each in input_lst:

        if each[index] == input_str:

            num = num + 1

    return num





num_of_us_movies = feature_counter(movie_data,6,"USA",True)



print(num_of_us_movies,len(movie_data))
def feature_counter(input_lst,index, input_str, header_row = False):

    num_elt = 0

    if header_row == True:

        input_lst = input_lst[1:len(input_lst)]

    for each in input_lst:

        if each[index] == input_str:

            num_elt = num_elt + 1

    return num_elt

def summary_statistics(input_lst):

    num_japan_films = feature_counter(input_lst,6,'Japan',True)

    num_color_films = feature_counter(input_lst,2,'Color',True)

    num_films_in_english = feature_counter(input_lst,5,'English',True)

    summary_dic = {"japan_films" : num_japan_films, "color_films" : num_color_films, "films_in_english" : num_films_in_english}

    return summary_dic

summary = summary_statistics(movie_data)
print(summary)