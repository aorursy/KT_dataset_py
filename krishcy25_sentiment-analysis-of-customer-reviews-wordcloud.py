# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Installation of WordCloud Package
!pip install WordCloud
# Importing the necessery modules 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import csv 
  
# File is read and object is created
file_ob = open(r"/kaggle/input/consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv") 
#Below are the stop words that are automatically imported from the WordCloud
stopwords = set(STOPWORDS) 

stopwords1=list(stopwords)
print(stopwords1)
#Any stop words other than the created above can be added here if you feel you don't need them based on the business problem

stopwords2=["for", "account", "not","on","FOR", "ACCOUNT","NOT","ON","is","This","For","The","the","hi","It","it","and"
           , "And","this","to","It is"]

print(type(stopwords2))
#Printing the combined list of stopwords created above that will be used in the final algorithm
stopwords_final=stopwords1+stopwords2

print(stopwords_final)
# Reader object is created 
reader_ob = csv.reader(file_ob) 
  
# Contents of reader object is stored . 
# Data is stored in list of list format. 
reader_contents = list(reader_ob) 
  
# Empty string is declare the text value to parse through
text = "" 
  
# Iterating through list of rows 
for row in reader_contents : 
      
    # iterating through words in the row.
    for word in row : 
  
        # concatenate the words 
        text = text + " " + word 
        
wordcloud = WordCloud(width=1000, height=1000, 
            stopwords=stopwords_final, background_color='white').generate(text) 
  


# Plot the WordCloud image.You can also give the size of the WordCloud image using the figsize paramter                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()
#The above code produces the below WordCloud

#Insights from the above generated WordCloud

#1)Most customers had a postive impact on all the products purchase as more positive keywords formed the WordCloud (like great,loves)
#2)Customer feel that the products are easy to use, They came at the great price, and they love them