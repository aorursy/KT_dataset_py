import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os
# read_csv without arguments

nf = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

nf.head()
# Parse dates with "parse_dates" argument

nf = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv', parse_dates=['date_added'])

nf.head()
# Read selected columns using "usecols" argument

list_cols = ['show_id', 'type', 'title', 'country', 'date_added', 'rating', 'duration']

nf = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv', parse_dates=['date_added'], usecols = list_cols)

nf.head()
# Set a column as index by "index_col" argument - useful for making time series

nf = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv', parse_dates=['date_added'], 

                 usecols = list_cols, index_col = 'show_id')

nf.head()
data_dir = '/kaggle/input/flowers-recognition/flowers/flowers'



# Dictionary of all images paths

flowers = {'sunflower':[], 'tulip':[], 'daisy':[], 'rose': [], 'dandelion': []}
# Populate the dictionary with image paths

for flower in flowers.keys():

    for dirname, _, filenames in os.walk(os.path.join(data_dir, flower)):

        for filename in filenames:

            flowers[flower].append((os.path.join(

                os.path.join(data_dir, flower), filename)))
# Showing 2 random images from each of the categories

from tensorflow.keras.preprocessing import image



for flower in list(flowers.keys()):

    plt.figure(figsize=(8, 5))

    flower_choice = np.random.choice(len(flowers[flower]),2) # Choose two images by random

    plt.subplot(1, 2, 1)

    img_path1 = flowers[flower][flower_choice[0]]

    img = image.load_img(img_path1)

    plt.imshow(img)

    plt.title(flower)

    plt.subplot(1, 2, 2)

    img_path2 = flowers[flower][flower_choice[1]]

    img = image.load_img(img_path2)

    plt.imshow(img)

    plt.title(flower)



plt.show()
# The dataset contains a metadata table



path = '../input/dickens/dickens'

meta = pd.read_csv(os.path.join(path, 'metadata.tsv'), delimiter='\t')

meta.head()
# Read the first part of one txt file

file1 = open(os.path.join(path, '924-0.txt'),'r')

_ = file1.read(500)

print(_)
files = {} # Create a dictionary of file names and contents

for f in os.listdir('/kaggle/input/dickens/dickens'):

    if f.endswith('.txt'): 

        with open(os.path.join(path, f), "r") as file:

            files[f] = file.read()
# Convert to dataframe

_ = pd.DataFrame.from_dict(files, orient='index', columns=['content']).reset_index()

dickens = meta.merge(_, left_on='Path', right_on='index')

dickens.head()
from google.cloud import bigquery



client = bigquery.Client()



# Get dataset

hacker_ref = client.dataset('hacker_news', project='bigquery-public-data')

hacker = client.get_dataset(hacker_ref)



# List all table names

tables = list(client.list_tables(hacker))

for table in tables:  

    print(table.table_id)
# Get a table

table_ref = hacker.table('full')

table = client.get_table(table_ref)
# Before loading the table, we can get some attributes

print ('No. of rows: ' + str(table.num_rows))

print ('Size (MB): ' + str(int(table.num_bytes / 1024768)))

print ('Columns:')

print (list(c.name for c in table.schema))
# Get first 100 rows

table_df = client.list_rows(table, max_results=100).to_dataframe()

table_df.head()
# Write SQL query to get data

# Get top 10 authors by number of articles

query = """

SELECT author, count(id) as stories

FROM `bigquery-public-data.hacker_news.stories`

GROUP BY author

ORDER BY count(id) DESC

"""



query_job = client.query(query)

iterator = query_job.result()

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

top_authors = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



# Look at the first 10 headlines

top_authors.head(10)
img = pd.read_parquet('../input/bengaliai-cv19/train_image_data_0.parquet')

img.shape
img.head()
img2 = img.iloc[:,1:].values.reshape((-1,137,236,1))



row=3; col=4;

plt.figure(figsize=(20,(row/col)*12))

for x in range(row*col):

    plt.subplot(row,col,x+1)

    plt.imshow(img2[x,:,:,0])

plt.show()
import cv2



DIM = 64



img3 = np.zeros((img2.shape[0],DIM,DIM,1),dtype='float32')

for j in range(img2.shape[0]):

    img3[j,:,:,0] = cv2.resize(img2[j,],(DIM,DIM),interpolation = cv2.INTER_AREA)



row=3; col=4;

plt.figure(figsize=(20,(row/col)*12))

for x in range(row*col):

    plt.subplot(row,col,x+1)

    plt.imshow(img3[x,:,:,0])

plt.show()
# Free up memory

del img
import json



data  = []

with open("/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json", 'r') as f:

    for line in f: 

        data.append(json.loads(line))



print("No. of records: {}".format(len(data)))
# See first item

data[0]
# Convert to dataframe - due to memory issue we only load the first 1000 items

df = pd.DataFrame(data[:1000])

df.head()
# Free up some memory

del data
from xml.etree import ElementTree



path = '../input/covid19-clinical-trials-dataset/COVID-19 CLinical trials studies/'



files = os.listdir(path)

print('Total Researches going on: ',len(files))
# Inspect the first element

file_path = os.path.join(path, files[0])

tree = ElementTree.parse(file_path)

root = tree.getroot()

print (root.tag, root.attrib)
# Look at all the children nodes

for child in root:

    print (child.tag, child.attrib)
df_temp = pd.DataFrame()

df = pd.DataFrame()

i = 0



for file in files[:10]: # Get the first 10 studies

    file_path = os.path.join(path, file)

    tree = ElementTree.parse(file_path)

    root = tree.getroot()

    trial = {} # Initialize dictionary

    

    # read tags using root.find() method

    trial['nct_id'] = root.find('id_info').find('nct_id').text

    trial['brief_title'] = root.find('brief_title').text

    trial['overall_status'] = root.find('overall_status').text

    

    df_temp  = pd.DataFrame(trial,index=[i])

    i=i+1

    

    df = pd.concat([df, df_temp])    
df