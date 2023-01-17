import csv

from collections import Counter

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

#%matplotlib inline

#this is a real world data from stackoverflow data.



with open ('../input/data.csv') as csv_file:

    csv_reader = csv.DictReader(csv_file) #using dictreader gets values by keys, not index

    

    language_counter = Counter() #new variable set to an empty counter

    

    #row = next(csv_reader) #to access the first row

    #print(row) 

    #print(row['LanguagesWorkedWith'].split(';')) #prints only the  values under the specified key.

    

    for row in csv_reader:

        language_counter.update (row['LanguagesWorkedWith'].split(';'))

        

#print(language_counter) #prints a sorted list of all the 28 languages as a TUPLE

#print(language_counter.most_common(15)) #prints the top 15 most common languages as a tuple



languages = []

popularity = []



for item in language_counter.most_common(15):

    languages.append(item[0]) #grabs the first item in the tuple, ie the language

    popularity.append(item[1]) #grabs the second item in the tuple, ie the number of people

    

#print(languages) #prints a list of all the 15 languages

#print(popularity) #prints a list of the number of people using the language



languages.reverse() # to reverse the list so that the most popular lang appears at the top.

popularity.reverse() # to reverse the list so that the most popular lang appears at the top.



plt.barh(languages, popularity)

plt.title('Most popular programming languages')



#plt.xlabel('Languages Worked with') for vertical bar chart

#plt.ylabel('Number of people working with the language') for vertical bar chart



#plt.ylabel('Languages Worked with')

plt.xlabel('Number of people working with the language')



plt.tight_layout()



plt.show()



#the vertical bar is messed up with so many data available, hence we use the horizontal bar, barh.
import csv

import pandas as pd

from collections import Counter

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



#this is a real world data from stackoverflow data.



#url = 'https://raw.githubusercontent.com/CoreyMSchafer/code_snippets/master/Python/Matplotlib/02-BarCharts/data.csv'

data = pd.read_csv('../input/data.csv') 

ids = data['Responder_id']

lang_responses = data['LanguagesWorkedWith']

#data.columns



language_counter = Counter()



for response in lang_responses:

    language_counter.update (response.split(';'))

    

languages = []

popularity = []



for item in language_counter.most_common(15):

    languages.append(item[0]) 

    popularity.append(item[1])

    

languages.reverse() # to reverse the list so that the most popular lang appears at the top.

popularity.reverse() # to reverse the list so that the most popular lang appears at the top.



plt.barh(languages, popularity)

plt.title('Most popular programming languages')



#plt.xlabel('Languages Worked with') for vertical bar chart

#plt.ylabel('Number of people working with the language') for vertical bar chart



#plt.ylabel('Languages Worked with')

plt.xlabel('Number of people working with the language')



plt.tight_layout()



plt.show()