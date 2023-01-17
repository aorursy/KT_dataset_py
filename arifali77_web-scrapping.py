# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# From Scratch
from bs4 import BeautifulSoup
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

print(soup)
print(soup.prettify()[0:600])
soup=BeautifulSoup('<b body="description"">Product Description</b>', 'html')

tag=soup.b

type(tag)
print(tag)
tag.name
tag.name='bestbooks'

tag
tag.name
tag['body']
tag.attrs
tag['id'] = 3

tag.attrs
tag
del tag['body']

del tag['id']

tag
tag.attrs
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
soup.head
soup.title
print(soup.body)
print(soup.body.b)
print(soup.ul)
print(soup.a)
soup=BeautifulSoup('<b body="description"">Product Description</b>')
tag=soup.b

type(tag)
tag.name
tag.string
type(tag.string)
nav_string=tag.string

nav_string
nav_string.replace_with('Null')

tag.string
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
for string in soup.stripped_strings: print(repr(string))
title_tag=soup.title

title_tag
print(title_tag.head)
title_tag.string
title_tag.string.parent
import re # regular expression
r = '''

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
soup=BeautifulSoup(r, 'lxml')

type(soup)
print(soup.prettify()[0:100])
text_only=soup.get_text()

print(text_only)
soup.find_all("li")
soup.find_all(id = "link 3")
soup.find_all("ul")
soup.find_all(["ul", "b"])
l = re.compile('l')

for tag in soup.find_all(l): 

    print(tag.name)
for tag in soup.find_all(True): 

        print(tag.name)
for link in soup.find_all('a'):

    print(link.get('href'))
soup.find_all(string=re.compile('data'))