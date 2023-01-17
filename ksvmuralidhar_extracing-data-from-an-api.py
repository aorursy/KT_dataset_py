import requests

import pandas as pd

from IPython.display import display

pd.set_option("display.max_rows",None)
def api_extract(country,n=-1):

    '''

    Function to extract university information from an API "http://universities.hipolabs.com/search?country="

    param country: Country for which one needs the university data

    paran n: Number of records to return. -1 returns all the data

    '''

    err=0 # initializing an error variable to 0

    req = requests.get(f"http://universities.hipolabs.com/search?country={country.lower().replace(' ','%20')}") # API request

    if req.status_code!= 200: # if status code != 200 error variable is assigned to 1

        err = 1



    js = req.json() # Getting the request output



    try:    

        if err==1:

            raise Exception()

        df = pd.DataFrame()

        for i in js[0].keys(): # Structuring the JSON output into a data frame

            df[i] = [j[i] for j in js]

    except:

        return "No Data Found" # Returning no data upon an error

    else:

        if n!=-1:

            return df.head(n)

        else:

            return df
# API request for university data in the US

api_extract("United States",10)
req = requests.get("https://api.covid19api.com/summary")

# req.status_code #Need not check the status code since there is no scope for error unless the API in not functional

js = req.json()





country_data = pd.DataFrame()

for i in range(len(js["Countries"])):

    js["Countries"][i].pop("Premium")

    country_data = country_data.append(pd.DataFrame(js["Countries"][i],index=[i])).copy()



country_data_main = country_data.copy()

display(country_data)