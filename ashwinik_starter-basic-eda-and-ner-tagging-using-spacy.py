# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
%matplotlib inline
from matplotlib import pyplot as plt
import nltk 
import spacy ### For NER tagging
import seaborn as sns
import pickle
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#### Read the compalints data csv
complaint_data = pd.read_csv("../input/consumer-complaints-financial-products/Consumer_Complaints.csv",low_memory = False)
### Convert the columns names so that they don't have space and are more readable
complaint_data.columns = [i.lower().replace(" ","_").replace("-","_") for i in complaint_data.columns]
complaint_data.columns
### Let us do basic description of the data
print ("The shape of data is ",complaint_data.shape)
print ("The data types for our data are as follows ")
print (complaint_data.info())

print (complaint_data.describe(include= 'object'))
### All the varables are text - which may correspond to categories and other variables
print (" The number of unique values in each column is as follows")
### Lets do a describe with including objects
complaint_data.describe(include = 'object').T.reset_index()

#### Keep only the consumer complaints is not null
complaint_data = complaint_data[~complaint_data['consumer_complaint_narrative'].isna()]
#### Create a distirbution of length of customers complaints. We have very left skew in length of complaints
### Which is expected as most compalints can be written in less than 500 words
complaint_data['consumer_complaint_narrative'].apply(len).plot(kind = 'hist',title = 'Histogram by length of compalints text')
plt.xlabel("Number of complaints")
### Keep the length columns as a new column
complaint_data['comp_length'] = complaint_data['consumer_complaint_narrative'].apply(len)
### Lets look at complaints distribution by product type
fig,ax = plt.subplots(figsize=(24,6))
complaint_data['product'].value_counts().plot(kind = 'bar',title = 'Complaints By Product')
plt.xlabel("Product")
plt.ylabel("Number of complaints")
### Lets look at the distribution of every product by distputed or not
pd.crosstab(complaint_data['product'],complaint_data['consumer_disputed?']).reset_index().set_index('product').sort_values('No',ascending = False).plot(kind='bar',title = 'Distribution of compalints by labels')
print (pd.crosstab(complaint_data['product'],complaint_data['consumer_disputed?']).reset_index().set_index('product').sort_values('No',ascending = False))
### Company Distribution by number of complaints
complaint_data['company'].value_counts()[0:25].plot(kind= 'bar',title = ' Top 25 companies by number of complaints')
plt.xlabel("Company Name")
plt.ylabel("Number of Complaints")
### Create a unqiue list fo each products
product_list = complaint_data['product'].unique()
### Iterate through each products category 
for i in product_list:
    ### Convert the text to lower case and subset only text for product of interest
    text = " ".join(review.lower() for review in complaint_data[complaint_data['product'] == i]['consumer_complaint_narrative'])
    ### Import the redefine stopwords list
    stopwords = set(STOPWORDS)
    ### Extend the predefine stop words list 
    stopwords.update(["xxxx", "xx", "xxxx", "xxxxx",'said','told','phone','trying','ask','asked',"call","called"])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

    # Display the generated image:
    # the matplotlib way:
    print ("Producing Word Cloud for :", i)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show() 

### Lets do some cleaning on the data. Mainly we will remove the stopwords and XXXX marks from our data
### In this data some word are masked due to sensitivity of the data

complaint_data['consumer_complaint_narrative'] =complaint_data['consumer_complaint_narrative'].str.replace(r'[^\w\s]',"")
complaint_data['consumer_complaint_narrative'] = complaint_data['consumer_complaint_narrative'].str.replace(r"XX+\s","")


#### Defined extract entities names and store it in a list 
def extract_org_list(str1):
    ''' This will take a str1 and extract the list of organization. This will be stored as a list of organisations'''
    ### We are using the predefined ner parser tagging
    docs = nlp(str1)
    
    ### We will return a list 
    return ( [str(i) for i in docs.ents if i.label_ == 'ORG'])
cmp = complaint_data.iloc[0:10]
nlp = spacy.load('en_core_web_sm')
cmp['organisation_list'] =cmp['consumer_complaint_narrative'].apply(extract_org_list)
print (cmp[['consumer_complaint_narrative','organisation_list']].head(1))
### apply it and store its as org_list_spacy
import time
start = time.process_time()
complaint_data['org_list_spacy'] = complaint_data['consumer_complaint_narrative'].apply(extract_org_list)
### Store the output in text file so that we don't have to run the model again
print(" Time taken to extract org list from data is ",time.process_time() - start)
complaint_data.to_csv("text.csv")