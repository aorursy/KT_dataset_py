# import libraries

from datetime import datetime

import lxml

from lxml import html

import requests

import numpy as np

import pandas as pd
# input a stock symbol

symbol = 'GOOGL'
# Yahoo Finance links

url_bs = 'https://finance.yahoo.com/quote/' + symbol + '/balance-sheet?p=' + symbol

url_is = 'https://finance.yahoo.com/quote/' + symbol + '/financials?p=' + symbol

url_cf = 'https://finance.yahoo.com/quote/' + symbol + '/cash-flow?p='+ symbol
# Set up the request headers that we're going to use, to simulate a request by the Chrome browser. 

# Simulating a request from a browser is generally good practice when building a scraper

headers = {

    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',

    'Accept-Encoding': 'gzip, deflate, br',

    'Accept-Language': 'en-US,en;q=0.9',

    'Cache-Control': 'max-age=0',

    'Pragma': 'no-cache',

    'Referrer': 'https://google.com',

    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36'

}
def get_table(url):

    # Fetch the page that we're going to parse, using the request headers defined above

    page = requests.get(url, headers)



    # Parse the page with LXML, so that we can start doing some XPATH queries

    # to extract the data that we want

    tree = html.fromstring(page.content)



    # Smoke test that we fetched the page by fetching and displaying the H1 element

    tree.xpath("//h1/text()")

    table_rows = tree.xpath("//div[contains(@class, 'D(tbr)')]")



    # Ensure that some table rows are found; if none are found, then it's possible

    # that Yahoo Finance has changed their page layout, or have detected

    # that you're scraping the page.

    assert len(table_rows) > 0



    parsed_rows = []



    for table_row in table_rows:

        parsed_row = []

        el = table_row.xpath("./div")



        none_count = 0



        for rs in el:

            try:

                (text,) = rs.xpath('.//span/text()[1]')

                parsed_row.append(text)

            except ValueError:

                parsed_row.append(np.NaN)

                none_count += 1



        if (none_count < 4):

            parsed_rows.append(parsed_row)



    df = pd.DataFrame(parsed_rows)

    df_org = df

    

    df = pd.DataFrame(parsed_rows)

    df = df.set_index(0) # Set the index to the first column: 'Period Ending'.

    df = df.transpose() # Transpose the DataFrame, so that our header contains the account names



    # Rename the "Breakdown" column to "Date"

    cols = list(df.columns)

    cols[0] = 'Date'

    df = df.set_axis(cols, axis='columns', inplace=False)

    df_rot = df

    return df_org, df_rot
# get Balance Sheet 

BS_orginal, BS_transpose = get_table(url_bs)



# get Income Statement 

IS_orginal, IS_transpose = get_table(url_is)



# get Cash Flow

CF_orginal, CF_transpose = get_table(url_cf)
# Balance Sheet original table

BS_orginal
# Balance Sheet transpose table

BS_transpose
BS_analysis = pd.DataFrame(BS_transpose['Date']) # copy columns of dataframe

BS_analysis
current_assets = BS_transpose['Total Current Assets'].str.replace(',', '').astype(int)

current_liabilities = BS_transpose['Total Current Liabilities'].str.replace(',', '').astype(int)

working_capital = current_assets - current_liabilities

BS_analysis['Working Capital'] = working_capital # copy columns of dataframe

BS_analysis
current_assets = BS_transpose['Total Current Assets'].str.replace(',', '').astype(int)

current_liabilities = BS_transpose['Total Current Liabilities'].str.replace(',', '').astype(int)

working_capital = current_assets - current_liabilities

total_sales = IS_transpose['Total Revenue'].str.replace(',', '').astype(int)

working_capital_per_dollar_of_sales = working_capital / total_sales

BS_analysis['Working Capital per Dollar of Sales'] = working_capital # copy columns of dataframe

BS_analysis
current_ratio = current_assets / current_liabilities

BS_analysis['Current Ratio'] = current_ratio

BS_analysis
inventory = BS_transpose['Inventory'].str.replace(',', '').astype(int)

quick_current_ratio = (current_assets - inventory) / current_liabilities

BS_analysis['Quick Current Ratio'] = quick_current_ratio

BS_analysis
total_liabilities = BS_transpose['Total Liabilities'].str.replace(',', '').astype(int)

shareholders_equity = BS_transpose['Total stockholders\' equity'].str.replace(',', '').astype(int)

debt2equity_ratio = total_liabilities / shareholders_equity

BS_analysis['Debt to Equity Ratio'] = debt2equity_ratio

BS_analysis
net_credit_sales = IS_transpose['Net Income'].str.replace(',', '').astype(int)

average_net_receivables_for_the_period = BS_transpose['Net Receivables'].str.replace(',', '').astype(int)

receivable_turnover = net_credit_sales / average_net_receivables_for_the_period

BS_analysis['Receivable Turnover'] = receivable_turnover

BS_analysis
number_of_days_in_period = 365

average_age_of_receivables = number_of_days_in_period / receivable_turnover

BS_analysis['Average Age of Receivables'] = average_age_of_receivables

BS_analysis
cost_of_goods_sold = IS_transpose['Cost of Revenue'].str.replace(',', '').astype(int)

average_inventory_for_the_period = inventory

inventory_turnover = cost_of_goods_sold / average_inventory_for_the_period

BS_analysis['Inventory Turnover'] = inventory_turnover

BS_analysis
number_of_days_for_inventory_to_turn = number_of_days_in_period / inventory_turnover

BS_analysis['Number of Days for Inventory to Turn'] = number_of_days_for_inventory_to_turn

BS_analysis
# Income Statement original table

IS_orginal
# Income Statement transpose table

IS_transpose
IS_analysis = pd.DataFrame(IS_transpose['Date']) # copy columns of dataframe

IS_analysis
revenue = IS_transpose['Total Revenue'].str.replace(',', '').astype(int)

cost_of_goods_sold = IS_transpose['Cost of Revenue'].str.replace(',', '').astype(int)

gross_profit_margin = (revenue - cost_of_goods_sold) / revenue

IS_analysis['Gross Profit Margin'] = gross_profit_margin * 100

IS_analysis
RD_expense = IS_transpose['Research Development'].str.replace(',', '').astype(int)

RD_to_sales = RD_expense / revenue

IS_analysis['RD to sales'] = RD_to_sales

IS_analysis
operating_income = IS_transpose['Operating Income or Loss'].str.replace(',', '').astype(int)

operating_profit_margin = operating_income / revenue

IS_analysis['Operating Profit Margin'] = RD_to_sales

IS_analysis
interest_expense = IS_transpose['Interest Expense'].str.replace(',', '').astype(int)

earnings_before_interest_and_taxes = IS_transpose['Income Before Tax'].str.replace(',', '').astype(int)

interest_coverage_ratio = earnings_before_interest_and_taxes / interest_expense

IS_analysis['Interest Coverage Ratio'] = interest_coverage_ratio

IS_analysis
net_income = IS_transpose['Net Income'].str.replace(',', '').astype(int)

net_profit_margin = net_income / revenue

IS_analysis['Net Profit Margin'] = net_profit_margin

IS_analysis
net_profit = IS_transpose['Net Income available to common shareholders'].str.replace(',', '').astype(int)

average_shareholde_equity_for_the_period = BS_transpose['Total stockholders\' equity'].str.replace(',', '').astype(int)

return_on_equity = net_profit / average_shareholde_equity_for_the_period

IS_analysis['Return on Equity'] = return_on_equity

IS_analysis
average_assets_for_the_period = BS_transpose['Total Assets'].str.replace(',', '').astype(int)

assets_turnover = revenue / average_assets_for_the_period

IS_analysis['Asset Turnover'] = assets_turnover

IS_analysis
return_on_assets = net_profit_margin / assets_turnover

IS_analysis['Return on Assets'] = return_on_assets

IS_analysis
# Cash Flow original table

CF_orginal
# Cash Flow transpose table

CF_transpose
CF_analysis = pd.DataFrame(CF_transpose['Date']) # copy columns of dataframe

CF_analysis
net_cash_provided_from_operating_activites = CF_transpose['Net cash provided by operating activites'].str.replace(',', '').astype(int)

average_current_liabilities = BS_transpose['Total Current Liabilities'].str.replace(',', '').astype(int)

current_liability_coverage_ratio = net_cash_provided_from_operating_activites / average_current_liabilities

CF_analysis['Current Liability Coverage Ratio'] = current_liability_coverage_ratio

CF_analysis
share_price = 1428.96

operating_cash_flow = CF_transpose['Operating Cash Flow'].str.replace(',', '').astype(int)

common_stock = BS_transpose['Common Stock'].str.replace(',', '').astype(int)

operating_cash_flow_per_share = operating_cash_flow / common_stock

price_to_cash_flow_ratio = share_price /  operating_cash_flow_per_share

CF_analysis['Price to Cash Flow Ratio'] = price_to_cash_flow_ratio

CF_analysis
cash_flow_margin_ratio = operating_cash_flow / net_sales

CF_analysis['Cash Flow Margin Ratio'] = cash_flow_margin_ratio

CF_analysis
cash_flow_coverage_ratio = cash_flow_from_operations / total_debt

CF_analysis['Cash Flow Margin Ratio'] = cash_flow_coverage_ratio

CF_analysis