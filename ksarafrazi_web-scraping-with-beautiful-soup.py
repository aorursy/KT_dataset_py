from bs4 import BeautifulSoup
import re
import urllib.request as ur

# The BeautifulSoup object
html_doc = '''
<html><head><title>Best Books</title></head>
<body>
<p class='title'><b>DATA SCIENCE FOR DUMMIES</b></p>

<p class='description'>Jobs in data science abound, but few people have the data science skills needed to fill these increasingly important roles in organizations. Data Science For Dummies is the pe
<br><br>
Edition 1 of this book:
        <br>
 <ul>
  <li>Provides a background in data science fundamentals before moving on to working with relational databases and unstructured data and preparing your data for analysis</li>
  <li>Details different data visualization techniques that can be used to showcase and summarize your data</li>
  <li>Explains both supervised and unsupervised machine learning, including regression, model validation, and clustering techniques</li>
  <li>Includes coverage of big data processing tools like MapReduce, Hadoop, Storm, and Spark</li>   
  </ul>
<br><br>
What to do next:
<br>
<a href='http://www.data-mania.com/blog/books-by-lillian-pierson/' class = 'preview' id='link 1'>See a preview of the book</a>,
<a href='http://www.data-mania.com/blog/data-science-for-dummies-answers-what-is-data-science/' class = 'preview' id='link 2'>get the free pdf download,</a> and then
<a href='http://bit.ly/Data-Science-For-Dummies' class = 'preview' id='link 3'>buy the book!</a> 
</p>

<p class='description'>...</p>
'''
soup = BeautifulSoup(html_doc, 'html.parser')
print(soup.prettify()[0:350])
# Tag objects
# Working with names
soup2 = BeautifulSoup('<b body="description"">Product Description</b>', 'html')

tag=soup2.b
type(tag)
print(tag.name)

tag.name = 'bestbooks'

print(tag)


#Working with attributes
tag['body']
#Adding an attribute
tag['id'] = 3
print(tag.attrs)

#Removing attributes
del tag['body']
del tag['id']
print(tag.attrs)


# Using tags to navigate a tree
#Head of the html document
soup.head
#Title
soup.title
#Name of the document
soup.body.b
#Unordered list within the document
soup.ul
#The first weblink in the document
soup.a
# NavigableString objects
soup2 = BeautifulSoup('<b body="description">Product description</b>')
tag= soup2.b
type(tag)
print(tag.string)
print(type(tag.string))
nav_string = tag.string
nav_string

nav_string.replace_with('Null')
nav_string
#Extracting strings from tree

for string in soup.stripped_strings: print(repr(string))
#Accesing parent tags
title_tag = soup.title
print(title_tag)

print(title_tag.parent)

print(title_tag.string)
print(title_tag.string.parent)
# Data parsing
# Getting data from a parse tree
text_only = soup.get_text()
print(text_only)
# Searching and retrieving data from a parse tree

# Retrieving tags by filtering with name arguments
soup.find_all("li")
#Retrieving tags by filtering with keyword arguments
soup.find_all(id="link 3")
# Retrieving tags by filtering with string arguments
soup.find_all('ul')
# Retrieving tags by filtering with list objects
soup.find_all(['ul', 'b'])
#Retrieving tags by filtering with regular expressions
l = re.compile('l')
for tag in soup.find_all(l): print(tag.name)
#Retrieving tags by filtering with a Boolean value
for tag in soup.find_all(True): print(tag.name)
#Retrieving weblinks by filtering with string objects
for link in soup.find_all('a'): print(link.get('href'))
#Retrieving strings by filtering with regular expressions¶
soup.find_all(string=re.compile("data"))
#Web scraping
r = ur.urlopen('https://analytics.usa.gov').read()
soup = BeautifulSoup(r, "lxml")
type(soup)

print (soup.prettify()[:100])
#Finding all a tags ans retrieving href valus
for link in soup.find_all('a'): print(link.get('href'))
#Finding all a tags that have an attribute of href and printing out https
for link in soup.findAll('a', attrs={'href': re.compile("^http")}): print(link)
#Saving results to a text file
file = open('parsed_data.txt', 'wt')
for link in soup.findAll('a', attrs={'href': re.compile("^http")}):
    soup_link = str(link)
    print(soup_link)
    file.write(soup_link)
file.flush()
file.close()
