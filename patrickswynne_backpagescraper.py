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
#imports
from bs4 import BeautifulSoup
import urllib.request 
import pandas as pd
#######
### Configurable items

domain = 'https://dallas.2backpage.com'

#target
target_url = domain + '/post/escorts'

#limits how many sub urls will be scanned
limiter = 5

#list of search terms: replace this with phone numbers or any other search term
search_terms = ['asian', 'Older Women 💛🌷💙 L', 'bitcoin']
#######
#variables 
# the page to be scraped
target_url_str = str(target_url)
# open the target page
r = urllib.request.urlopen(target_url)
#create the soup object
soup = BeautifulSoup(r, "lxml")
# create an empty set
comp_set = [] # a placeholder for the urls found within the primary domain
url_collection = []
def scan_for_urls(url_to_scan):
    #compare url_collection size start to end size to determine if more were found
    
    for link in soup.find_all('a'):
        if type(link.get('href')) == str and len(link.get('href')) > 1 and link.get('href')[0] != '#':
            clean_url = (
                link.get('href')[0: -1] 
            if link.get('href')[-1] == '/' 
            else link.get('href'))
        
        focus_url = clean_url if clean_url[0:8] == 'https://' and domain in clean_url else domain + clean_url
        if focus_url not in url_collection:
            url_collection.append(focus_url)
            
    print(len(url_collection))
    
scan_for_urls('asd')
         

"""
We find web links by iterating over each anchor tag in the html code
"""
for link in soup.find_all('a'):
    
    """
    We then extract the href attribute, ensure it isn't simply '/' meaning 
    'this domain' which would diplicate of the domain. We also do not want to 
    collect hashtags, which are present for page navigation and organization.
    An additional check for type of str is present to avoid None types.
    """
    if type(link.get('href')) == str and len(link.get('href')) > 1 and link.get('href')[0] != '#':
        
        """
        Before we add any urls to the python set we should remove trailing / 
        to ensure that links are not duplicated. home.html/ and home.html would
        otherwise be viewed as two urls when they point to the same web page.
        """
        clean_url = (
            link.get('href')[0: -1] 
            if link.get('href')[-1] == '/' 
            else link.get('href'))
        
        
        """
        Add the clean url to the set, concat the domain where it is not present
        """
        
        #check that the url contains 'https://dallas.2backpage.com/view/escorts' these are the targets to open
        if '/view/escorts' in clean_url:
            comp_set.append(clean_url if clean_url[0:8] == 'https://' else domain + clean_url)
    
        
# Can display the extracted links, ex) https://dallas.2backpage.com/view/escorts/2519703.html
#for rec in comp_set:
    #print(rec)
"""
1. Open each link in the list
2. Check if any item in our phone list is found on the open page
3. If a phone number is found then add that tuple {phone: url} to a suspect list
4. move to the next item
5. export the suspect list to csv locally
"""
final_set =[]

df = pd.DataFrame(final_set, columns=['term', 'url'])
for sub_url in comp_set[:limiter]:
    r = urllib.request.urlopen(sub_url)
    #create the soup object
    soup = BeautifulSoup(r, "lxml")
    for item in search_terms:
        if item in soup.body.text:
            new_row = {'term':item, 'url':sub_url}
            #append row to the dataframe
            df = df.append(new_row, ignore_index=True)

print(df)
df.to_csv('web_findings.csv', sep=',', encoding='utf-8')