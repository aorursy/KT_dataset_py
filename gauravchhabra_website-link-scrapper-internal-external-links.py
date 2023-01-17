print("""I Have created a function for the same, this function will fetch all the links with 'a' & 'href' tag.

After fetching the links, it will segregate them into Internal links, External Links & Rem Links (Remaining)

Bifurcation will be done on the basis of 'RootLink' Paramenter""")



##  Website's  Internal & External Link Scrapper

def Get_Website_Links(WebsiteLink, RootLink):

    ## Importing Necessary Libraries

    import numpy as np

    import pandas as pd

    import requests

    import bs4

    

    # Mapping Variables

    link = WebsiteLink

    RootLink = RootLink

    

    ## Creating Object - Connecting to web

    data = requests.get(link)

    

    # Using Html Parser - Importing Webpage Data

    

    soup = bs4.BeautifulSoup(data.text, 'html.parser')

    

    # Creating Necessary Array's - For First Loop

    Rem_Links = [] # (Remaining Links)

    Internal_Links = []

    External_Links = []

    

    # Creating Necessary Array's - For Second Loop

    Rem_Links_2 = []

    Internal_Links_2 = []

    External_Links_2 = []

    

    # Creating loop - For segregatin links into Internal & External & Rem (Remaining)

    for links in soup.find_all('a'):

        url = links.get('href')

    

    # If condition for segregating URL

        if url is not None and url !="":

            if url[0:4] == 'http' and url.find(RootLink) != -1 :

                Internal_Links.append(url)

            elif url[0] == '/':

                Internal_Links.append(link + url)

            elif url[0:4] == 'http' and url.find(RootLink) == -1 :

                External_Links.append(url)

            else:

                Rem_Links.append(url)

    

    # Getting Unique Values | Removing Duplicates

    Internal_Links = np.unique(np.asarray(Internal_Links))

    Rem_Links = np.unique(np.asarray(Rem_Links))

    External_Links = np.unique(np.asarray(External_Links))

    

    # Creating DataFrame

    Internal_Links = pd.DataFrame({'Internal_Links':Internal_Links})

    Rem_Links = pd.DataFrame({'Rem_Links':Rem_Links})

    External_Links = pd.DataFrame({'External_Links':External_Links})

    

    # Concat DataFrames

    All_links = pd.concat([Internal_Links,External_Links,Rem_Links ], axis = 1)



    ## Second Loop for all above links - Fetch from all the links added in above array

    for link_2 in All_links['Internal_Links']:

        data_2 = requests.get(link_2)

        

        # Using Html Parser

        soup_2 = bs4.BeautifulSoup(data_2.text, 'html.parser')

        

        # Creating loop - For segregatin links into Internal & External & Rem (Remaining)

        for link_3 in soup_2.find_all('a'):

            url_2 = link_3.get('href')

            

            # If condition for segregating URL

            if url_2 is not None and url_2 != "":

                if url_2[0:4] == 'http' and url_2.find(RootLink) != -1 :

                    Internal_Links_2.append(url_2)

                elif url_2[0] == '/':

                    Internal_Links_2.append(link + url_2)

                elif url_2[0:4] == 'http' and url_2.find(RootLink) == -1 :

                    External_Links_2.append(url_2)

                else:

                    Rem_Links_2.append(url_2)

    

    # Getting Unique Values | Removing Duplicates

    Internal_Links_2 = np.unique(np.asarray(Internal_Links_2))

    Rem_Links_2 = np.unique(np.asarray(Rem_Links_2))

    External_Links_2 = np.unique(np.asarray(External_Links_2))

    

    # Creating DataFrame

    Internal_Links_2 = pd.DataFrame({'Internal_Links_2':Internal_Links_2})

    Rem_Links_2 = pd.DataFrame({'Rem_Links_2':Rem_Links_2})

    External_Links_2 = pd.DataFrame({'External_Links_2':External_Links_2})

    

    # Concat DataFrames

    All_links_2 = pd.concat([Internal_Links_2,External_Links_2,Rem_Links_2 ], axis = 1)

    

    ## Returning Final DataFrame - All Links

    return All_links_2





# Sample Website - Test 1

Get_Website_Links("https://www.google.co.in", 'google.co')
# Sample Website - Test 2

Get_Website_Links("https://www.facebook.com", 'facebook.com')