import requests # send web-requests (to fetch the html files from websites)

from bs4 import BeautifulSoup # beautiful-soup: navigate through the html

import numpy as np

import pandas as pd

import os, re, pprint
## Configuration

url = "http://www.covidmaroc.ma/pages/Accueil.aspx"



headers = {

  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'

}



Data = {

  "Date"      : "",

  "Confirmed" : "",

  "Deaths"    : "",

  "Recovered" : "",

  "Excluded"  : "",

  

  "Beni Mellal-Khenifra"     : "",

  "Casablanca-Settat"        : "",

  "Draa-Tafilalet"           : "",

  "Dakhla-Oued Ed-Dahab"     : "",

  "Fes-Meknes"               : "", 

  "Guelmim-Oued Noun"        : "",

  "Laayoune-Sakia El Hamra"  : "",

  "Marrakesh-Safi"           : "",

  "Oriental"                 : "",

  "Rabat-Sale-Kenitra"       : "",

  "Souss-Massa"              : "",

  "Tanger-Tetouan-Al Hoceima": "",

}
## Helper functions

def getDate(text):

    tag = text.find_all('p')[0]

    

    # Remove chars other than numbers and -

    tag = re.sub(r'[^0-9\-]', '', str(tag))

    return re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", tag).group(0)



def getRecovered_Deaths(text):

    tag = text.find_all('p')[1]

    tag_Recov = tag.find_all('span')[0].get_text()

    tag_death = tag.find_all('span')[1].get_text()

    

    # Remove unicode characters

    tag_death = tag_death.encode('ascii', 'ignore').decode("utf-8")

    tag_Recov = tag_Recov.encode('ascii', 'ignore').decode("utf-8")

    return (int(tag_Recov), int(tag_death))



def getConfirmed_Excluded(text):

    tagConf = text.find_all('p')[3].get_text()

    tagExcl = text.find_all('p')[4].get_text()

    

    # Remove unicode characters

    tagConf = tagConf.encode('ascii', 'ignore').decode("utf-8")

    tagExcl = tagExcl.encode('ascii', 'ignore').decode("utf-8")

    return (int(tagConf), int(tagExcl))



def percent_to_actuale_number(percentage):

    return np.ceil(Data['Confirmed']*percentage/100)



def get_region_cases(text):

    headers = text.find_all('h2')

    

    vals = []

    for i in range(1, len(headers)):

        try:

            val = headers[i].get_text()

            # Remove unicode characters

            val = val.encode('ascii', 'ignore').decode("utf-8")

            val = val.split(' ')[0]

            

            # Select values that start with numbers

            if not val[0].isdigit():

                continue

            

            val = float(val.replace('%','').replace(',','.'))

            vals.append(percent_to_actuale_number(val))

        except:

            pass

    return vals
## Main function

def get_covid_Data(url):



    try:

        result = requests.get(url, headers=headers)

    except HTTPError as e:

        print("error in opening url")

        return None

  

    try:

        bsObj = BeautifulSoup(result.content.decode("utf-8"), "html.parser")



        table1 =  bsObj.find_all('table')[0] # Grab the first table

#         table2 =  bsObj.find_all('table')[1] # Grab the second table



        Data["Date"] = getDate(table1)

        Data["Recovered"], Data["Deaths"] = getRecovered_Deaths(table1)

        Data["Confirmed"], Data["Excluded"] = getConfirmed_Excluded(table1)



#         CasesByRegion = get_region_cases(table2)

#         Data["Beni Mellal-Khenifra"]      = CasesByRegion[0]

#         Data["Casablanca-Settat"]         = CasesByRegion[1]

#         Data["Draa-Tafilalet" ]           = CasesByRegion[2]

#         Data["Dakhla-Oued Ed-Dahab"]      = CasesByRegion[3]

#         Data["Fes-Meknes"]                = CasesByRegion[4]

#         Data["Guelmim-Oued Noun"]         = CasesByRegion[5]

#         Data["Laayoune-Sakia El Hamra"]   = CasesByRegion[6]

#         Data["Marrakesh-Safi"]            = CasesByRegion[7]

#         Data["Oriental"]                  = CasesByRegion[8]

#         Data["Rabat-Sale-Kenitra"]        = CasesByRegion[9]

#         Data["Souss-Massa"]               = CasesByRegion[10]

#         Data["Tanger-Tetouan-Al Hoceima"] = CasesByRegion[11]



        pprint.pprint(Data)



    except AttributeError as e:

        print("error in reading html")

        return None





get_covid_Data(url)
df = pd.DataFrame(Data.items(), columns=['__', '__']).T

# Replace Header with first row

df.columns = df.iloc[0]

df = df[1:]

df
## Next step is to run this code every day and add add records to the Full Dataset

## Now detailed data is shared every day in PDF File. So Next step is to Create another script to scrape data from pdf files.