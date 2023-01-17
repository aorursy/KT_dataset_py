# Imports

from PIL import Image

import requests

from io import BytesIO

import pandas as pd

import numpy as np

from urllib.parse import urlsplit

import matplotlib.pyplot as plt

import re



%matplotlib inline
# Data source: '500 Companies' --> 'http://www.opendata500.com/us/list/' Uncomment below to grab the data if needed.



#!wget http://www.opendata500.com/us/download/us_companies.csv



data = pd.read_csv('../input/us_companies.csv')
data.head()
# This code will be working exclusively with the company URLs, to extract/scrape logo images only.

urls_list = data['url']
print(urls_list[0:10])
# Ensure all of the urls are in the proper form (with scheme) for using the urlsplit module.

i=0

for url in urls_list:

    if not re.match(r'http(s?)\:', url):

            data['url'][i] = 'http://' + url

            i+=1
# Re-copy the fixed urls

urls_list = data['url']
# Parse the domains only from each url, save over the existing df values.

i=0

for url in urls_list:

    parsed = urlsplit(url)

    host = parsed.netloc 

    if host.startswith('www.'):

        host = host[4:]

        urls_list[i] = host

        i+=1
''' The below code utilizes the following API --> 'https://logo.clearbit.com/SomeDomain.com'

For each domain, fetches just the logo image from a company's url. Logos successfully fetched are appended to the

logos list, since some errors are to be expected here for those exceptions a log is kept of skipped urls/logos in

the skipped_urls list. This script will run for est. 3-4 minutes. '''



logos = []

skipped_urls = []

for url in urls_list:

    try:

        url_text = 'https://logo.clearbit.com/' + str(url)

        response = requests.get(url_text)

        img = Image.open(BytesIO(response.content))

        logos.append(img)

    except OSError:

        skipped_urls.append(url)

        continue
# Number of logos successfully scraped.

len(logos)
# Print the first 20 scraped logos for review.

# ** Note: using this method via matplotlib to display multiple images in one cell reduces quality. **

i=0

while i < 20:

    plt.figure()

    plt.imshow(logos[i])

    i+=1
import pickle



''' Uncomment the appropriate option below: load or save scraped logos. '''



# Save the scraped logos for future use via Pickle.

pickle.dump( logos, open( "logos.p", "wb" ) )



# Load a previous dumped logos file from Pickle. 

#logos = pickle.load( open( "logos.p", "rb" ) )