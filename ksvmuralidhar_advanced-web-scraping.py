import os

print(os.popen("curl https://www.cars-data.com/robots.txt").read())
import requests

from bs4 import BeautifulSoup

import pandas as pd

import numpy as np

import time

import os

from IPython.display import clear_output

pd.set_option('display.max_colwidth', None)
init_urls_df = pd.DataFrame() # Initializing DataFrame to store the scraped URLs



#last_page = 91 # Original number of pages

last_page = 1 # Limiting the number of pages to 1. Change it to 91 to scrape the whole website



for i in range(1,last_page+1): # Loop to iterating through the pages

    

    #'requests' module has a method 'get' which fetches the data from a webpage

    r = requests.get("https://www.cars-data.com/en/all-cars/page" + str(i) + ".html")

    

    # BeautifulSoup enables to find the elements/tags in a webpage

    bs = BeautifulSoup(r.text) # passing the source code of the webpage to 'BeautifulSoup'

    

    # Selecting all the 'a' tags (URLs) present in the webpage and extracting their 'href' attribute

    init_urls = pd.Series([a.get("href") for a in bs.find_all("a")])

    

    ''' Among all the URLs we could find a pattern in the URLs of the cars.

        All the URLs of cars have 5 forward slashes (/) and end with a digit. Hence we select URLs containing 5 '/'

        and ending with a digit

    '''

    init_urls = init_urls[(init_urls.str.count("/")==5) & (init_urls.str.contains("\d$")==True)].unique()

    

    # Adding the URL we need to a DataFrame 'df'

    df = pd.DataFrame({"initial_urls":init_urls})

    

    # Appending 'df' to a main DataFrame 'init_urls_df'

    init_urls_df = init_urls_df.append(df).copy()

    

    # Printing the status

    print("Processed " + str(i) + "/" + str(last_page) + " URLs")

    

    '''

    Pausing the process by 5 to 10 seconds to ensure the server isn't overloaded with requests.

    '''

    time.sleep(np.random.randint(5,10,1))

    

    # clearing the output printed

    clear_output(wait=True)

    

    # Writing the scraped URLs to a csv file with a tab separator

    init_urls_df = init_urls_df.reset_index(drop=True)

    init_urls_df.to_csv("cars-data-init-urls.csv",sep="\t",index=False)
init_urls_df
# Reading the URLs scraped in Phase 1

init_urls = pd.read_csv("cars-data-init-urls.csv",sep="\t")["initial_urls"].unique()



# Limiting to only first 10 URLs. Comment/remove the statement below to scrape all the URLs

init_urls = init_urls[:10]



# Initializing DataFrame to store URLs scraped in this phase

stage2_url_df = pd.DataFrame()





for i in range(len(init_urls)): # Iterating through each URL scraped in Phase 1

    r = requests.get(init_urls[i])

    bs = BeautifulSoup(r.text)

    

    # Selecting all the URLs in the page

    stage2_url = pd.Series([a.get("href") for a in bs.find_all("a")])

    

    '''

    Among all the URLs, the URLs having the car details have a '-specs/' string in them which 

    distinguishes them from others.

    '''

    stage2_url = stage2_url[stage2_url.str.contains("-specs/",regex=False)]

    

    # Adding the scraped URLs to a DataFrame as we did in Phase 1

    df = pd.DataFrame({"stage2_urls":stage2_url})

    

    stage2_url_df = stage2_url_df.append(df).copy()

    print("Processed "+str(i+1) + "/"+str(len(init_urls))+" URLs")

    clear_output(wait=True)

    time.sleep(np.random.randint(5,10,1))

    

    # Writing the scraped URLs to a csv file with a tab separator

    stage2_url_df = stage2_url_df.reset_index(drop=True)

    stage2_url_df.to_csv("cars-data-stage-2-urls.csv",sep="\t",index=False)
stage2_url_df
# Function to scrape the cars data

def final_scrape(url,key):

    r = requests.get(url)

    bs = BeautifulSoup(r.text)

    

    '''Getting the full name of a car from the breadcrumb on the top of the page.

    example: https://www.cars-data.com/en/audi-tt-coupe-40-tfsi-specs/80385/tech

    The last part of the breadcrumb is the full name of the car, which contains "-specs/" on it's "href"

    '''

    namu = pd.Series([a.get("href") for a in bs.find_all("a")])

    namu = namu.str.contains("-specs/",regex=False)

    nam = pd.Series([a.text for a in bs.find_all("a")])

    nam = nam[namu]

    nam = nam.str.replace("\n","") # removing \n (if any) in the car' name

    nam = nam.reset_index(drop=True)[0]



    # 'dt' tags represent the property of the car

    dt = pd.Series([a.text for a in bs.find_all("dt")])

    

    # removing the : at the end and also \n (if any)

    dt = dt.str.replace("\n|(:$)","")

    

    # 'dd' tags represent the values of the car

    dd = pd.Series([a.text for a in bs.find_all("dd")])



    dt = dt.reset_index(drop=True)

    dd = dd.reset_index(drop=True)



    # Creating DataFrame with full car name, scraped 'dt' and 'dd' tags, key, and source URL

    # 'key' odentifies the page (tech / sizes / options) from which data is scraped

    df = pd.DataFrame({"Car Name_Full":nam,"Spec":dt,"Info":dd,"Page":key,"Source URL":url})

    

    # Replacing N.A. and '-' with NaN

    df["Info"] = df["Info"].replace("N.A.",np.nan,regex=False)

    df["Info"] = df["Info"].replace("-",np.nan,regex=False)

    time.sleep(np.random.randint(5,10,1))

    return df
# Reading URLs scraped in Phase 2

stage2_urls = pd.read_csv("cars-data-stage-2-urls.csv",sep="\t")["stage2_urls"].unique()





# Initializing DataFrame to store final data

final_df = pd.DataFrame()



fcnt=0 # File counter which is explained later

for i in range(len(stage2_urls)):

    

    

    # Scraping the /tech page (example: https://www.cars-data.com/en/audi-tt-coupe-40-tfsi-specs/80385/tech)

    final_df = final_df.append(final_scrape(stage2_urls[i] + "/tech","tech"))

    print("Step "+str(i+1)+str(".1"))

    

    # Scraping the /sizes page (example: https://www.cars-data.com/en/audi-tt-coupe-40-tfsi-specs/80385/sizes)

    final_df = final_df.append(final_scrape(stage2_urls[i] + "/sizes","sizes"))

    print("Step "+str(i+1)+str(".2"))

    

    # Scraping the /options page (example: https://www.cars-data.com/en/audi-tt-coupe-40-tfsi-specs/80385/options)

    final_df = final_df.append(final_scrape(stage2_urls[i] + "/options","options")).copy()

    print("Step "+str(i+1)+str(".3"))

    

    # Printing the progress

    print("Processed "+str(i+1)+"/"+str(len(stage2_urls))+" URL")

    

    clear_output(wait=True)

    

    # The scraped data is written to an Excel file after every 100 iterations to prevent data loss

    if i%100==0:

        final_df.to_excel("cars-data-final-data"+str(fcnt)+".xlsx",index=False)

        

    ''' Since the data is being written to a file after every 100 iterations, 

    it is better to reinitialize the 'final_df' after every 10,000 iterations which removes the data already written to Excel

    and reduces the time taken to write the dataframe to Excel. 

    '''

    if (i%10000==0) & (i>0):

        final_df=pd.DataFrame()

        fcnt = fcnt+1

        

final_df.to_excel("cars-data-final-data"+str(fcnt+1)+".xlsx",index=False)
# Combining all the data written to different files

files = pd.Series(os.listdir(os.getcwd()))



# Selecting files which contain the phrase 'cars-data-final-data'

files = files[files.str.contains("cars-data-final-data")].reset_index(drop=True)

files
# Combining/appending all the files and storing them in FINAL_DF dataframe

FINAL_DF = pd.DataFrame()

for i,j in enumerate(files):

    FINAL_DF = FINAL_DF.append(pd.read_excel(j)).copy()

    print("Added "+str(i+1)+"/"+str(len(files))+" Files")
FINAL_DF.drop_duplicates(inplace=True)

FINAL_DF = FINAL_DF.reset_index(drop=True)

FINAL_DF.to_excel("Cars_data_Final.xlsx",index=False)
FINAL_DF