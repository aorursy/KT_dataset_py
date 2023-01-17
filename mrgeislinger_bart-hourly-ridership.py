# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Typing ftw

from typing import List



import requests # For scraping from original source

import re # To pull out specific URLs in scraping

import time # Timing functions calls

from zipfile import ZipFile # Creating compressed files
# Flag to run with debugging information (printouts)

DEBUG = False
def get_origin_destination_urls(url_origin_destination: str = 'http://64.111.127.166/origin-destination', 

               ext: str ='csv.gz', use_full_path: bool = False, 

               printout: bool = False) -> List:

    '''Return list of URLs of files (from given extension).

        

        Args: 

            url_origin_destination: Source to scrape file URLs from.

            ext: Extension of files to search for (defaults "csv.gz").

            use_full_path: Flag to return full path of files or relative path.

            printout: Prints out information for basic debugging.

    '''

    

    resp = requests.get(url_origin_destination)

    # DEBUG

    if printout:

        print(resp.text)

        

    # Pattern to pull from HTML links for files

    href_pattern = r'<a\s+(?:[^>]*?\s+)?href=(["\'])(\S*\.{})\1+>'.format(ext)



    # More efficient to compile (https://docs.python.org/3/library/re.html#re.compile)

    prog = re.compile(href_pattern)

    file_group_list = prog.findall(resp.text)



    # Decide to return full path in list of files

    file_list = [group[-1] for group in file_group_list]

    if use_full_path:

        return [f'{url_origin_destination}/{fname}' for fname in file_list]

    else:

        return file_list

    
# Example usage

if DEBUG:

    csv_list = get_origin_destination_urls(use_full_path=True)

    df = pd.read_csv(csv_list[0], 

                     names=['Date','Hour','Origin Station','Destination Station','Trip Count']

                    )
# Retrieve the README (also will grab all other `txt`) files

txt_list = get_origin_destination_urls(use_full_path=True, ext='txt')

for txt_url in txt_list:

    resp_txt_urls = requests.get(txt_url)

    # Remove HTML escaped spaces

    export_fname = txt_url.split('/')[-1].replace('%20','')

    with open(export_fname,'wb') as f:

        f.write(resp_txt_urls.content)
# Directory for all data

try:

    os.mkdir('ridership')

except:

    print('New directory not made')
# Using the definition of columns from "READ ME.txt"

col_names = ['Date','Hour','Origin Station','Destination Station','Trip Count']



# Retrieve all CSVs (`csv.gz`)

csv_list = get_origin_destination_urls(use_full_path=True)
def time_eval(func):

    '''Simple timer decorator to printout the time a function runs.

    '''

    def timer_wrapper(*args, **kw):

        start = time.time()

        print(f'Starting {func.__name__} run...')

        result = func(*args, **kw)

        elapsed_time = time.time() - start

        print(f'Time to run {func.__name__}: {elapsed_time:.1f} seconds')

        return result

    return timer_wrapper
@time_eval

def create_df(csv_fname, col_names):

    '''Creates a DataFrame of using only keeping certain columns.

    '''

    df = pd.read_csv(csv_fname, names=col_names)

    export_fname = csv_fname.split('/')[-1]

    # Ignore the default index (useless for this dataset)

    csv_fname = f'ridership/{export_fname}'

    return df





@time_eval

def create_data_files(list_of_csvs, col_names):

    '''Create one zip file of all CSVs.

    '''

    with ZipFile('ridership.zip', 'w') as myzip:

        # Add all compressed CSVs to one zip

        for csv_fname in list_of_csvs:

            print(f'\t Adding {csv_fname} to zip archive')

            df = create_df(csv_fname, col_names)

            myzip.writestr(csv_fname, df.to_csv(index=False))
# Iterated over each URL to create a separate file (for each year) in archive

create_data_files(csv_list, col_names)