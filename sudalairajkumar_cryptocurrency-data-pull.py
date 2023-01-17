# -*- coding: utf-8 -*-

import re

import sys

import csv

import time

import random

import requests

from datetime import date

from bs4 import BeautifulSoup



end_date = str(date.today()).replace("-","")

base_url = "https://coinmarketcap.com/currencies/{0}/historical-data/?start=20130428&end="+end_date



currency_name_list = ["bitcoin", "ethereum", "ripple", "bitcoin-cash", "nem", "litecoin", "dash", "ethereum-classic", "iota", "neo", "stratis", "monero", "waves", "bitconnect", "omisego", "qtum", "numeraire"]





def get_data(currency_name):

    print("Currency : ", currency_name)

    url = base_url.format(currency_name)

    html_response = requests.get(url).text.encode('utf-8')

    soup = BeautifulSoup(html_response, 'html.parser')

    table = soup.find_all('table')[0]

    elements = table.find_all("tr")

    with open("./{0}_price.csv".format(currency_name.replace("-","_")),"w") as ofile:

        writer = csv.writer(ofile)

        for element in elements:

            writer.writerow( element.get_text().strip().split("\n") )

    time.sleep(1)



if __name__ == "__main__":

    for currency_name in currency_name_list:

        #get_data(currency_name)

        pass
import time

import requests

import pandas as pd



urls = [

'https://blockchain.info/charts/market-price',

'https://blockchain.info/charts/total-bitcoins',

'https://blockchain.info/charts/market-cap',

'https://blockchain.info/charts/trade-volume',

'https://blockchain.info/charts/blocks-size',

'https://blockchain.info/charts/avg-block-size',

'https://blockchain.info/charts/n-orphaned-blocks',

'https://blockchain.info/charts/n-transactions-per-block',

'https://blockchain.info/charts/median-confirmation-time',

'https://blockchain.info/charts/hash-rate',

'https://blockchain.info/charts/difficulty',

'https://blockchain.info/charts/miners-revenue',

'https://blockchain.info/charts/transaction-fees',

'https://blockchain.info/charts/cost-per-transaction-percent',

'https://blockchain.info/charts/cost-per-transaction',

'https://blockchain.info/charts/n-unique-addresses',

'https://blockchain.info/charts/n-transactions',

'https://blockchain.info/charts/n-transactions-total',

'https://blockchain.info/charts/n-transactions-excluding-popular',

'https://blockchain.info/charts/n-transactions-excluding-chains-longer-than-100',

'https://blockchain.info/charts/output-volume',

'https://blockchain.info/charts/estimated-transaction-volume',

'https://blockchain.info/charts/estimated-transaction-volume-usd'

]



suffix_to_add = '?timespan=8years&format=csv'



def get_btc_data():

    counter = 0

    for url in urls:

        header = ['Date', "btc_" + url.split("/")[-1].replace("-","_")]

        print(header[-1])

        temp_df = pd.read_csv(url+suffix_to_add, header=None, names=header)

        if counter == 0:

            df = temp_df.copy()

        else:

            df = pd.merge(df, temp_df, on="Date", how="left")

        print(temp_df.shape, df.shape)

        counter += 1

        time.sleep(1)

    df.to_csv("../input_v9/bitcoin_dataset.csv", index=False)

    

#get_btc_data()
import time

import requests

import pandas as pd



urls = [

'https://etherscan.io/chart/etherprice',

'https://etherscan.io/chart/tx',

'https://etherscan.io/chart/address',

'https://etherscan.io/chart/marketcap',

'https://etherscan.io/chart/hashrate',

'https://etherscan.io/chart/difficulty',

'https://etherscan.io/chart/blocks',

'https://etherscan.io/chart/uncles',

'https://etherscan.io/chart/blocksize',

'https://etherscan.io/chart/blocktime',

'https://etherscan.io/chart/gasprice',

'https://etherscan.io/chart/gaslimit',

'https://etherscan.io/chart/gasused',

'https://etherscan.io/chart/ethersupply',

'https://etherscan.io/chart/ens-register'

]



suffix_to_add = '?output=csv'



def get_ether_data():

    counter = 0

    for url in urls:

        header = ['Date', 'TimeStamp', "eth_" + url.split("/")[-1].replace("-","_")]

        print(header[-1])

        

        with open("temp.csv", "w") as ofile:

            response = requests.get(url+suffix_to_add).text.encode('utf-8')

            ofile.write(response)

        temp_df = pd.read_csv("temp.csv")

        

        col_names = temp_df.columns.tolist()

        if col_names[-1] == "Value":

            col_names = col_names[:2] + [header[-1]]

            temp_df.columns = col_names

        else:

            temp_df = temp_df[["Date(UTC)","UnixTimeStamp", "Supply", "MarketCap"]]

            temp_df.columns = ["Date(UTC)","UnixTimeStamp", "eth_supply", "eth_marketcap"]

            

        if counter == 0:

            df = temp_df.copy()

        else:

            df = pd.merge(df, temp_df, on=["Date(UTC)","UnixTimeStamp"], how="left")

        print(temp_df.shape, df.shape)

        counter += 1

        time.sleep(1) 

    df.to_csv("../input_v9/ethereum_dataset.csv", index=False)

    

#get_ether_data()