from bs4 import BeautifulSoup

import requests

import pycountry



def headline_reader(country = "US",term = "Coronavirus", language = "en", when = "", exclude = ""):

    #Check for proper country ISO code

    if len(country) > 2:

        country = (pycountry.countries.search_fuzzy('Cote'))[0].alpha_2

    #Format the actual URL

    if (when == "") & (exclude == "") :

        url = "https://news.google.com/search?q={search_term}&hl={language}-{country}&gl={country}&ceid={country}:{language}".format(search_term = term, country = country, language = language)

    elif (when != "") & (exclude == "") :

        url = "https://news.google.com/search?q={search_term}%20when%3A{when}&hl={language}-{country}&gl={country}&ceid={country}:{language}".format(search_term = term, country = country, language = language, when = when )

    elif (when == "") & (exclude != "") :

        url = "https://news.google.com/search?q={search_term}%20-{exclude}&hl={language}-{country}&gl={country}&ceid={country}:{language}".format(search_term = term, country = country, language = language, exclude = exclude )

    else:

        url = "https://news.google.com/search?q={search_term}%20when%3A{when}%20-{exclude}&hl={language}-{country}&gl={country}&ceid={country}:{language}".format(search_term = term, country = country, language = language, when = when, exclude = exclude )

    #Fetch raw data

    response = requests.get(url)

    #Parse the txt part as html structured 

    soup = BeautifulSoup(response.text, 'html.parser')

    #Find all headlines

    headers = soup.find_all("h3")

    #Store them clean in a list

    headlines = [str(headers[i].find("a").contents).strip('[]').strip("'").strip("\"") for i in range(len(headers))]

    return(headlines)

    
headline_reader("America","Coronavirus", exclude = "live")[0:4]
headline_reader("England","Coronavirus", when = "1w")[0:4]
headline_reader("India","Coronavirus", when = "1w", exclude = "lockdown")[0:4]