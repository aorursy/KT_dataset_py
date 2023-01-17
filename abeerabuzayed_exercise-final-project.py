import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex7 import *

print("Setup Complete")
# Check for a dataset with a CSV file

step_1.check()
# Fill in the line below: Specify the path of the CSV file to read

my_filepath = "../input/goodreadsbooks/books.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath, error_bad_lines=False, index_col="bookID") #This data was entirely scraped via the Goodreads API





# Check that a dataset has been uploaded into my_data

step_3.check()
my_data.shape
my_data.describe()
# Print the first five rows of the data

my_data.head()
language_code_count= my_data.groupby('language_code').count()

#language_code_count= my_data.groupby('language_code')['title'].count()
language_code_count
# Create a plot

# Your code here

# Set the width and height of the figure

plt.figure(figsize=(25,10))

# Add title

plt.title("Distribution of books over all languages")

# Bar chart 

sns.barplot(x= language_code_count.index, y= language_code_count['title'] )

# Add label for vertical axis

plt.ylabel("Books number")

# Check that a figure appears below

step_4.check()
plt.figure(figsize=(25,10))

plt.title("Average rating of books over all languages")

sns.barplot(x= my_data.language_code, y= my_data.average_rating)



#P.S. there is only one book is written in wel (Welsh language) that got 5/5!

plt.title("Books average rating over pages numbers")

sns.scatterplot(x="average_rating", y="# num_pages", data = my_data)
#Find top 5 popular books:

my_data.sort_values('text_reviews_count', ascending=False)[:5]

top10_books= my_data.sort_values('text_reviews_count', ascending=False)[:10]

plt.figure(figsize=(13,12))

plt.title("The top 10 popular books")

sns.barplot(x=top10_books['text_reviews_count'], y=top10_books['title'])
