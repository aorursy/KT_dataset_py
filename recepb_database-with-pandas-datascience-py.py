import sqlite3

#SQLite3 is a very easy to use database engine. It is self-contained, serverless, zero-configuration and transactional. 

#It is very fast and lightweight, and the entire database is stored in a single disk file.

#It is used in a lot of applications as internal data storage.The Py.Library includes a module called "sqlite3" intended for working with this database. 



import numpy as np

#NumPy is a python library used for working with arrays.

#It also has functions for working in domain of linear algebra, fourier transform, and matrices.

#We have lists that serve the purpose of arrays, but they are slow.NumPy aims to provide an array object that is up to 50x faster that traditional Python lists.



import pandas as pd 

#Why pandas: you want to explore a dataset stored in a CSV on your computer. Pandas will extract the data from that CSV into a DataFrame — 

#a table, basically — then let you do things like:

#Calculate statistics and answer questions about the data, like: What's the average, median, max, or min of each column?

#Does column A correlate with column B?

#What does the distribution of data in column C look like?

#Clean the data by doing things like removing missing values and filtering rows or columns by some criteria

#Visualize the data with help from Matplotlib. Plot bars, lines, histograms, bubbles, and more.

#Store the cleaned, transformed data back into a CSV, other file or database



import os

#The OS module in python provides functions for interacting with the operating system.

#This module provides a portable way of using operating system dependent functionality.

#The *os* and *os.path* modules include many functions to interact with the file system.



import matplotlib.pyplot as plt

#Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.



import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#UTF-8 is a variable-width character encoding standard 

#that uses between one and four eight-bit bytes to represent all valid Unicode code points.



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.       
con = sqlite3.connect("/kaggle/input/soccer/database.sqlite")

data = pd.read_sql_query('SELECT * from League, Country,Team', con)

data.head()