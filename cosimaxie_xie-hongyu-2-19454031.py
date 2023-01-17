!pip install simfin
import simfin as sf



# Import names used for easy access to SimFin's data-columns.

from simfin.names import *
sf.__version__
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

%matplotlib inline
# set API key

sf.set_data_dir('~/simfin_data/')

sf.set_api_key(api_key='free')

sf.load_api_key(path='~/simfin_api_key.txt', default_key='free')
df_companies = sf.load_companies(index=TICKER, market='us')

df_industries = sf.load_industries()

df_companies.loc['JNJ']
# the sector and industry of johnson

df_industries.loc[106005]
df_income = sf.load_income(variant='annual', market='us', index=[TICKER,FISCAL_YEAR])

df_balance = sf.load_balance(variant='annual', market='us', index=[TICKER,FISCAL_YEAR])

df_cashflow = sf.load_cashflow(variant='annual', market='us', index=[TICKER,FISCAL_YEAR])
PL = df_income.loc['JNJ']

BS = df_balance.loc['JNJ']

CF = df_cashflow.loc['JNJ']
firm_BS = BS.copy ()

firm_BS[TOTAL_ASSETS] /= 1e6

firm_BS[TOTAL_LIAB] /= 1e6

firm_BS[SHARES_BASIC] /= 1e6
# Intangible assets of JNJ in million USD, net intangible asset and goodwill from 2008 to 2018 from 2008,2009,2010,2011,2012,2013,2014,2015,2016,2017 and 2018 annual report of Johnson & Johnson

INTANGIBLES = pd.DataFrame(np.array([[13976,13719],[6323,14862],[16716,15294],[18138,16138],[28752,22424],

                                    [27947,22798], [27222,21832],[25764,21629],[26876,22805],[53228,31906],[47611,30453]]), 

                           columns=['IA',"Goodwill"],

                           index = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018])
# 1.Book value (figure 1) under "Book Value"

firm_BS["book value"] = firm_BS[TOTAL_ASSETS] - firm_BS[TOTAL_LIAB] - INTANGIBLES["IA"]-INTANGIBLES["Goodwill"]

firm_BS["book value"].plot()

plt.title("Book Value")
df_prices = sf.load_shareprices(variant='daily', market='us')
# obtain the share price in the year end from 2008 to 2018

share_prices = [df_prices.loc['JNJ'][CLOSE]['2008-12-31'],df_prices.loc['JNJ'][CLOSE]['2009-12-31'],

                df_prices.loc['JNJ'][CLOSE]['2010-12-31'],df_prices.loc['JNJ'][CLOSE]['2011-12-30'],

                df_prices.loc['JNJ'][CLOSE]['2012-12-31'],df_prices.loc['JNJ'][CLOSE]['2013-12-31'],

                df_prices.loc['JNJ'][CLOSE]['2014-12-31'],df_prices.loc['JNJ'][CLOSE]['2015-12-31'],

                df_prices.loc['JNJ'][CLOSE]['2016-12-30'], df_prices.loc['JNJ'][CLOSE]['2018-01-02'],

                df_prices.loc['JNJ'][CLOSE]['2018-12-31']]
# 2. Price-to-book value (PB) Ratio from 2008 to 2018 (table 1) under "Book Value"

BV_pershare = firm_BS["book value"]/firm_BS[SHARES_BASIC]

price_to_book = share_prices/BV_pershare 

price_to_book
# 3. Revenue growth ratio (table 2) under " Sales potential "

year18 = PL[REVENUE].loc[2018]/PL[REVENUE].loc[2017]-1

year17 = PL[REVENUE].loc[2017]/PL[REVENUE].loc[2016]-1

year16 = PL[REVENUE].loc[2016]/PL[REVENUE].loc[2015]-1

year15 = PL[REVENUE].loc[2015]/PL[REVENUE].loc[2014]-1

year14 = PL[REVENUE].loc[2014]/PL[REVENUE].loc[2013]-1

year13 = PL[REVENUE].loc[2013]/PL[REVENUE].loc[2012]-1

year12 = PL[REVENUE].loc[2012]/PL[REVENUE].loc[2011]-1

year11 = PL[REVENUE].loc[2011]/PL[REVENUE].loc[2010]-1

year10 = PL[REVENUE].loc[2010]/PL[REVENUE].loc[2009]-1

year09 = PL[REVENUE].loc[2009]/PL[REVENUE].loc[2008]-1
# 4. calculate gross margin, operating margin, and net profit margin (table 3) under "Margin pressure"

gross_margin = PL[GROSS_PROFIT]/PL[REVENUE]

operation_margin= PL[OP_INCOME]/PL[REVENUE]

profit_margin = PL[NET_INCOME_COMMON]/PL[REVENUE]

print(gross_margin)

print(operation_margin)

print(profit_margin)
# 5. free cash flow (FCF) (table 6) under "Investment plans"

FCF = CF[NET_CASH_OPS] + CF[CHG_FIX_ASSETS_INT]

FCF

Dividend_to_FCF = -CF[DIVIDENDS_PAID]/FCF
# 6. Interest Coverage Ratio (figure 2) under "Interest Coverage Ratio"

# Interest Coverage Ratio= EBIT / Interest Expense

Interest_Coverage = -PL[OP_INCOME]/ PL[INTEREST_EXP_NET]

Interest_Coverage

plt.title("Interest Coverage Ratio")

Interest_Coverage.plot()

Interest_Coverage.round(3)
# 7. Target Buy Price under "Buy recommendation"



# 1) EPS = Net Income Available to Common Shareholders / Total weighted average shares outstanding, basic

EPS = PL[NET_INCOME_COMMON]/BS[SHARES_BASIC]

EPS
# 2) Earnings Per Share Annual Compounded Growth Rate (EPS CAGR)

EPS_in_2009 = EPS[2009]

EPS_in_2018 = EPS[2018]

year = 9

EPS_CAGR = (EPS_in_2018/EPS_in_2009)**(1/year)-1

print("Earnings Per Share Annual Compounded Growth Rate is %.6f" % EPS_CAGR)
# 3) Estimate EPS 10 years from now

estimated_EPS = EPS[2018]* (1+EPS_CAGR)**10

print("Estimate EPS 10 years from now is %0.3f" % (estimated_EPS))
# 4) Determine Current Target Buy Price

# a. Estimate Stock Price 10 Years from now: Estimated future EPS X five-year Average PE ratio (74.8) (2016-2019) from https://finbox.com/NYSE:JNJ/explorer/pe_ltm

Estimated_SP = estimated_EPS * 74.8

Discount_rate = 0.09 #from 2019 annual report



# b. Determine Target Buy Price Today based on Desired Returns

Target_Buy_Price = Estimated_SP/(1+Discount_rate)**10

print("Target Buy Price based on Desired Returns is %0.3f" % (Target_Buy_Price))
# 5) Margin of Safety (15% off the target buy price)

margin_of_safety = 0.15 * Target_Buy_Price



# 6) final results-Current Target Buy Price

current_Target_Buy_Price = Target_Buy_Price - margin_of_safety

print("Current Target Buy Price is %0.3f" % (current_Target_Buy_Price))
# 8.ROA and ROE uner "Competitor comparisons"

# relatively high ROA (>7%) and ROE (>15%) show the efficency of company management 

ROA = PL[NET_INCOME_COMMON]/BS[TOTAL_ASSETS]

ROE = PL[NET_INCOME_COMMON]/BS[TOTAL_EQUITY]

ROA.round(3)

ROA.round(3)