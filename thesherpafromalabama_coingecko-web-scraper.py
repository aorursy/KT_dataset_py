# Webscraper script to get trust score data for exchanges from coingecko

# cd into the folder using terminal/anaconda and run >python CG_Webscraper.py



print("firing up scraper")



#%% Imports and Initializations



import requests

from bs4 import BeautifulSoup

import re

import pandas as pd

import datetime

from urlpath import URL

from tqdm import tqdm





url = 'https://www.coingecko.com/en/exchanges/gdax#trust_score'

base_url = URL('https://www.coingecko.com')
base_url
url = 'https://www.coingecko.com/en/exchanges/huobi_korea#trust_score'

r = requests.get(url)

c = r.content

soup = BeautifulSoup(c,'html.parser')



soup.find_all("div", {"class":"py-5 col-12"})[0].find_all('div', {'class':'progress-bar'})[0]['style']

# soup
# Need to get a list of all exchanges on CG

list_exchanges = []

for i in range(1,5):

    # Get URL for exchanges (Spot)

    url = 'https://www.coingecko.com/en/exchanges?page={}'.format(i)

    r = requests.get(url)

    c = r.content

    soup = BeautifulSoup(c,'html.parser')

    # Get length of exchanges

    exchanges_len = len(soup.find_all('div', {'class':'gecko-table-container'})[0].find_all('span', {'class':'pt-2 flex-column'}))

    # Now iterate through each page 

    for j in range(exchanges_len):

        end_url = soup.find_all('div', {'class':'gecko-table-container'})[0].find_all('span', {'class':'pt-2 flex-column'})[j].a['href'] + '#trust_score'

        list_exchanges.append(end_url)

# Initialize a pandas df

CG_df = pd.DataFrame()



# Get exchange URL

for link in tqdm(list_exchanges):

    url = base_url /link

    r = requests.get(url)

    c = r.content

    soup = BeautifulSoup(c,'html.parser')





    # Now take url and get soups, and parse out all the data into dictionaries





    # Header Information (Name and CG score)

    header_dict = {}

    # Get name of exchange XDDD LOL

    exchange_name = soup.find('h1').text

    # Get overall ranking (scale of 1-10)

    exchange_trust_score = soup.find_all("div", {"class":"pb-5 row mt-4 card-body"})[0].find_all('div', {'style':'font-size: 48px;'})[0].text.replace("\n","")

    if exchange_trust_score == '':

        exchange_trust_score = 'NA'

    # Store into dict

    header_dict['Exchange_Name'] = exchange_name

    header_dict['Trust_Score'] = exchange_trust_score

    header_dict['Type'] = soup.find_all('div', {'class':'col-lg-4 exchange-details d-flex justify-content-lg-start justify-content-center'})[0].find('small').text



    # Get Table of Trust score estimates

    trust_score_dict = {}

    trust_score_dict['Liquidity'] = soup.find_all("table", {"class":"table col-9 d-none d-lg-block"})[0].find_all("td")[0].text

    trust_score_dict['Scale'] = soup.find_all("table", {"class":"table col-9 d-none d-lg-block"})[0].find_all("td")[1].text

    trust_score_dict['API_Coverage'] = soup.find_all("table", {"class":"table col-9 d-none d-lg-block"})[0].find_all("td")[2].text

    trust_score_dict['Regulatory_Compliance'] = soup.find_all("table", {"class":"table col-9 d-none d-lg-block"})[0].find_all("td")[3].text

    trust_score_dict['Estimated_Reserves'] = soup.find_all("table", {"class":"table col-9 d-none d-lg-block"})[0].find_all("td")[4].text



    # Get specific Liquidty data

    liq_dict = {}

    liq_dict['Reported Trading Volume'] = re.search("(\d*\.\d*)", str(soup.find_all("div", {"class":"py-5 col-12"})[0].find_all("div",{"data-target":"price.price"})[0])).group(0)

    liq_dict['Normalized Trading Volume'] = re.search("(\d*\.\d*)", str(soup.find_all("div", {"class":"py-5 col-12"})[0].find_all("div",{"data-target":"price.price"})[1])).group(0)

    liq_dict['Reported-Normalized Trading Volume'] = soup.find_all("div", {"class":"py-5 col-12"})[0].find_all("td")[2].text

    liq_dict['Average Bid-Ask Spread'] = soup.find_all("div", {"class":"py-5 col-12"})[0].find_all("td")[3].text.replace("\n","")

    try:

        green = int(re.search('(\d*)(\%)', soup.find_all("div", {"class":"py-5 col-12"})[0].find_all('div', {'class':'progress-bar'})[0]['style']).group(1))

        yellow = int(re.search('(\d*)(\%)', soup.find_all("div", {"class":"py-5 col-12"})[0].find_all('div', {'class':'progress-bar'})[1]['style']).group(1))

        red = int(re.search('(\d*)(\%)', soup.find_all("div", {"class":"py-5 col-12"})[0].find_all('div', {'class':'progress-bar'})[2]['style']).group(1))

        liq_dict['Green_Pairs'] = green

        liq_dict['Yellow_Pairs'] = yellow

        liq_dict['Red_Pairs'] = red

        liq_dict['Unknown_Pairs'] = 100 - green - yellow - red

    except:

        liq_dict['Green_Pairs'] = 'NA'

        liq_dict['Yellow_Pairs'] = 'NA'

        liq_dict['Red_Pairs'] = 'NA'

        liq_dict['Unknown_Pairs'] = 'NA'     



    # Get specific exchange Scale data

    scale_dict = {}

    scale_dict['Normalized Volume Percent'] = soup.find_all("div", {"class":"py-5 col-12"})[1].find_all("td")[0].text.replace("\n","")

    scale_dict['Combined Orderbook Percentile'] = soup.find_all("div", {"class":"py-5 col-12"})[1].find_all("td")[1].text.replace("\n","")



    # get specific API Coverage data

    API_dict = {}

    try:

        API_dict['Grade_Score'] = soup.find_all("div", {"class":"api_grading_text"})[0].text.replace("\n","")

        API_dict['Tickers'] = soup.find_all("table", {"class":"table text-center"})[0].find_all("td")[0]['alt']

        API_dict['Historical_Data'] = soup.find_all("table", {"class":"table text-center"})[0].find_all("td")[1]['alt']

        API_dict['Orderbook'] = soup.find_all("table", {"class":"table text-center"})[0].find_all("td")[2]['alt']

        API_dict['Trading_via_API'] = soup.find_all("table", {"class":"table text-center"})[0].find_all("td")[3]['alt']

        API_dict['Candlestick'] = soup.find_all("table", {"class":"table text-center"})[0].find_all("td")[4]['alt']

        API_dict['Websocket'] =soup.find_all("table", {"class":"table text-center"})[0].find_all("td")[5]['alt']

        API_dict['Public_Documentation'] = soup.find_all("table", {"class":"table text-center"})[0].find_all("td")[6]['alt']

        API_dict['API_Last_Updated'] = re.search('\d*\-\d*\-\d*', soup.find_all('div', {'class':"text-right"})[0].text).group(0)

    except:

        API_dict['Grade_Score'] = 'NA'

        API_dict['Tickers'] = 'NA'

        API_dict['Historical_Data'] = 'NA'

        API_dict['Orderbook'] = 'NA'

        API_dict['Trading_via_API'] = 'NA'

        API_dict['Candlestick'] = 'NA'

        API_dict['Websocket'] = 'NA'

        API_dict['Public_Documentation'] = 'NA'

        API_dict['API_Last_Updated'] = 'NA'



    # get specific Regulatory Compliance data

    Reg_Comp_dict = {}

    try:

        Reg_Comp_dict['License_And_Authorization'] = soup.find_all("div", {"class":"py-5 col-12"})[3].find_all("table", {"class":"table"})[1].find_all("td")[0].find("span").text

        Reg_Comp_dict['Sanctions'] = soup.find_all("div", {"class":"py-5 col-12"})[3].find_all("table", {"class":"table"})[1].find_all("td")[1].find("span").text

        Reg_Comp_dict['Senior_Public_Figure'] = soup.find_all("div", {"class":"py-5 col-12"})[3].find_all("table", {"class":"table"})[1].find_all("td")[2].find("span").text

        Reg_Comp_dict['Jurisdiction_Risk'] = soup.find_all("div", {"class":"py-5 col-12"})[3].find_all("table", {"class":"table"})[1].find_all("td")[3].find("span").text

        Reg_Comp_dict['KYC_Procedures'] = soup.find_all("div", {"class":"py-5 col-12"})[3].find_all("table", {"class":"table"})[1].find_all("td")[4].find("span").text

        Reg_Comp_dict['Negative_News'] = soup.find_all("div", {"class":"py-5 col-12"})[3].find_all("table", {"class":"table"})[1].find_all("td")[5].find("span").text

        Reg_Comp_dict['AML'] = soup.find_all("div", {"class":"py-5 col-12"})[3].find_all("table", {"class":"table"})[1].find_all("td")[6].find("span").text

        Reg_Comp_dict['Regulatory_Last_Updated'] = re.search('\d*\-\d*\-\d*', soup.find_all("div", {"class":"py-5 col-12"})[3].find("small").text).group(0)

    except:

        Reg_Comp_dict['License_And_Authorization'] = 'NA'

        Reg_Comp_dict['Sanctions'] = 'NA'

        Reg_Comp_dict['Senior_Public_Figure'] = 'NA'

        Reg_Comp_dict['Jurisdiction_Risk'] = 'NA'

        Reg_Comp_dict['KYC_Procedures'] = 'NA'

        Reg_Comp_dict['Negative_News'] = 'NA'

        Reg_Comp_dict['AML'] = 'NA'

        Reg_Comp_dict['Regulatory_Last_Updated'] = 'NA'



    # Merge existing dictionaries, then append to the df

    CG_dict = {**header_dict, **trust_score_dict, **scale_dict, **API_dict, **Reg_Comp_dict}

    CG_df = CG_df.append(CG_dict, ignore_index = True)
list_exchanges[185]
# Post processing of df (reorder columns so name and score are first two columns)

cols = CG_df.columns.tolist()

cols.insert(0, cols.pop(cols.index('Exchange_Name')))

cols.insert(1, cols.pop(cols.index('Trust_Score')))

cols.insert(2, cols.pop(cols.index('Type')))

CG_df = CG_df.reindex(columns= cols)
# Save this data

CG_df.to_csv('CG_data05102020.csv', index = False)
# Now, want to combine data from CMC with CG, import it here (previously scraped)

CMC_data = pd.read_csv('liquidity_data_2020-05-10_13.csv')



# left merge entries and save as a csv

CMC_CG_Combined = pd.merge(CG_df, CMC_data, on="Exchange_Name")

CMC_CG_Combined.to_csv('May_CMC_CG_Combo.csv', index = False)