#import relevant libraries
import wordcloud
import matplotlib
import bs4
import csv
import pandas
import numpy as np
from collections import Counter
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from PIL import Image
#open connection, grab page
#my_url = 'https://en.wikipedia.org/wiki/List_of_Marvel_Comics_superhero_debuts'
#uClient = uReq(my_url)
#page_html = uClient.read()
#uClient.close()

#open csv file for writing
#file_name = "marvel.csv"
#f = open(file_name,"w")

#write headers to csv
#headers = "Hero, Debut, Author, Comic\n"
#f.write(headers)

# html parser
#page_soup = soup(page_html,"html.parser")

# scrape table rows for data
#table_row = page_soup.findAll("tr") 
#for row in table_row[2:len(table_row)-13]:
#	try:
#		row_info = row.findAll("td")
#		hero = row_info[0].text.strip()
#		date = row_info[1].text
#		author = row_info[2].text
#		comic = row_info[3].text
#		#clean commas from data because csv are comma delimeted
#		f.write(hero.replace(",", "|").replace("\n","") + "," + date.replace(",", "|") + "," + author.replace(",", "|") + "," + comic.replace(",", "|").replace("\n","") + "\n")
#		#skip rows 
#	except IndexError:
#		pass

#close csv file		
#f.close()

#create arrays to hold data from csv columns
dates = []
authors = []

#read data from csv columns to arrays
with open('../input/marvel.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter = ',')
	for row in readCSV:
		dates.append(row[1])
		authors.append(row[2])
#seperate entries with multip authors
new_authors = []
for author in authors:
	new_authors.extend(author.split('|'))
#clean white space
new_authors = [x.strip() for x in new_authors[2:]]
#create dictionary 
author_counts = Counter(new_authors)

#remove objects from dictionary count less than 2
for k in list(author_counts):
	if author_counts[k] < 3:
		del author_counts[k]
		
#use pandas library to create bar graph
df = pandas.DataFrame.from_dict(author_counts, orient='index')
df.plot(kind = 'bar', color = 'rygbck')
plt.title("Creators of Marvel Heroes")
plt.xlabel("Authors")
plt.ylabel("Number of Heroes Created")
plt.show()
#Create Word cloud
wc = WordCloud(scale = 3)
wc.generate(str(new_authors))
plt.imshow(wc)
plt.axis('off')
plt.show()