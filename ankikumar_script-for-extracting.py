'''

This is the program to extract 144 pages from Next Space Flight website



As a new coder,looking forward for valuable suggestion and advice to make it better

'''

!pip install selenium

!apt-get update # to update ubuntu to correctly run apt install

!apt install -y chromium-chromedriver

!cp /usr/lib/chromium-browser/chromedriver /usr/bin





import sys

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bs4 import BeautifulSoup, Tag, NavigableString

import re



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))







from selenium import webdriver

chrome_options = webdriver.ChromeOptions()

chrome_options.add_argument('--headless')

chrome_options.add_argument('--no-sandbox')

chrome_options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)









details = []    #details of rocket name on first page 

url_list=[]     #for fetching details

r_lst = []      #list of rocket name

f_lst = []      #final list for dataframe





def url_format(url2, count=None):

   if "page=" in str(url2):

        new_url = "https://nextspaceflight.com/launches/past/"+url2

   elif "details" in str(url2):

        new_url = "https://nextspaceflight.com"+url2

   else:

       if count == 1:

            new_url = "https://nextspaceflight.com/launches/past/?page=1"

       else:

           return ("{} does not exist".format(url2))

    

   return new_url



def geturl(url):

    #opening a webpage

    driver.get(url)

    #Selenium stores the source HTML in the driver's page_source

    html = driver.page_source

    clean = BeautifulSoup(html, "html.parser")

    return clean





#--------------------------------------------------------------------------------

# FIRST PAGE SCRAPPING

# 1. extracting all the name of rockets on first page and putting in details list

def first_page(soup_f):



    soup_rname  = soup_f.find_all("h5")



    for rocket in soup_rname:

        item2 = str(rocket.text) #extracting text name and converting it into string

        r_lst.append(item2.strip()) #striping whitespace

    



    # 2. finding all the hrefs and open it

    soup_href = soup_f.find_all("button")

    for item in soup_href:

        if "location" not in str(item):

            continue

        url2 = item.get("onclick", None)

        clean_url = url2.split("'")

        new_url = url_format(clean_url[1])

        url_list.append(new_url)

    return url_list, r_lst



#----------------------------------------------------------------------------------





#----------------------------------------------------------------------------------

#Second Page Operation



def r_func(url2):



# 3. finding company name, price and status

#parsing in beautiful soup

    r_details = []

    soup_stat = url2.find(text= re.compile(r"^Status: "))

    tag_s = soup_stat.parent



#company name

    for comp in tag_s.previous_elements:

        if isinstance(comp, NavigableString):

            if comp.string == "\n":

                continue

            

            r_details.append(comp.string.strip())

            break



    count_s=0

    for sp in tag_s.next_elements:

        if isinstance(sp, NavigableString):          #check for navigable string

            if sp.string != "\n" and count_s<2:      #found, ignore new line and check count

                y = sp.string.split(":")             #convert it into list to extract text

                if y[0] == 'Status':

                    

                    r_details.append(y[1].strip())

                elif y[0] == 'Price':

                    if "," in y[1]:

                        x = "".join(y[1].split(","))

                        r_details.append(float(''.join(re.findall('[0-9.]\S*',x))))

                    else:

                        r_details.append(float(''.join(re.findall('[0-9.]\S*',y[1]))))

                else:

                    r_details.append("NaN")

                    

                count_s = count_s+1



    # 4. finding mission status

    soup_ms = url2.find_all("h6")

    for item in soup_ms:

        if "status" not in str(item):

            continue    

        r_details.append(item.text.strip())

 

    # 5. getting location status

    soup_address = url2.find(text="Location")

    tag_a = soup_address.parent

    for loc in tag_a.next_elements:

       if loc.name == "h4":

           

           r_details.append(loc.text)

           break



    # 6. launch date

    soup_ltime = ''.join(url2.find('br').next_siblings) 

    

    r_details.append(soup_ltime.strip())



    return r_details

#----------------------------------------------------------------------------------

#loop for extracting data



    

url = "https://nextspaceflight.com/"

page = 1

func_count = 0



total_page = 144



while(page<=total_page):

    print(page,"start extracting")

    details = []

    if page == 1:

        clean_url = url_format(url, count=1)    #it will create the url -> return new url

        soup_first = geturl(clean_url)          #beautiful soup parsing -> return parsed 

        first_page(soup_first)                  #first page operation -> return clean url_list and r_lst 

    

    #adding rocket name (details) from first page operations

    

    

    

    for lst in url_list[:-1]:                   #it will run 31 time, including next page url

        

        if "details" in lst:

            soup_d = geturl(lst)                #beautiful soup parsing -> return parsed

            details = r_func(soup_d)                      #second page operations -> return r_details      

            details.append(r_lst[func_count])

            func_count = func_count+1

            f_lst.append(details)

            

        elif ("page" in lst) and (url_list.index(lst)==len(url_list)-2):                 #it will open next page

            soup_sec = geturl(lst)           #beautiful soup parsing -> return parsed 

            

            url_list = []

            r_lst = []

            func_count = 0

            first_page(soup_sec) 

            

        

        

        details=[]

      

    print(page, "turn end")

    columns = ["Company","Status","Price","Mission Status","Location","Datum","Details"]

    df= pd.DataFrame(f_lst, columns=columns)

    df1_count = len(df)

    if page == 1:

        df.to_csv(r'RocketData.csv')

        

            

    else:

                

        df.index = np.arange(df1_count*(page-1), df1_count*page)

        df.to_csv(r'RocketData.csv', header=False, mode='a')

        print("CSV Created Successfully")

            

            

    f_lst=[]  

    page = page+1