import pandas as pd

from bs4 import BeautifulSoup

from tqdm import tqdm, tqdm_notebook

import time

import requests





SLEEP_TIME_S = 0.1
def extract_URL_and_Name(page):

    """ From the page name in the input file extract the Name and the URL """

    return (['_'.join(page.split('_')[:-3])]

            + ['http://' + page.split("_")[-3:-2][0] +

               '/wiki/' + '_'.join(page.split('_')[:-3])])
# Load the dataset

train = pd.read_csv('../input/train_1.csv')



# We will just take a sample of the data, 

# remove this line to run on all the data

train = train.sample(2)



# Extract the Page name and URL:

page_data = pd.DataFrame(

    list(train['Page'].apply(extract_URL_and_Name)),

    columns=['Name', 'URL'])

page_data.head()
# Since Kaggle kernels don't have internet access this method will always return 

# an empty string

def fetch_wikipedia_text_content(row):

    """Fetch the all text data of a given page"""

    try:

        r = requests.get(row['URL'])

        # Sleep for 100 ms so that we don't use too many Wikipedia resources 

        time.sleep(SLEEP_TIME_S)

        to_return = [x.get_text() for x in 

                     BeautifulSoup(

                         r.content, "html.parser"

                     ).find(id="mw-content-text").find_all('p')]

    except:

        to_return = [""]

    return to_return
# This will fail due to lack of Internet

tqdm.pandas(tqdm_notebook)

page_data['TextData'] = page_data.progress_apply(fetch_wikipedia_text_content, axis=1)



page_data.head()