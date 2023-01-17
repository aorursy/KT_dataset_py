# !pip install plotly

# !pip install matplotlib

# !pip install pandas-datareader
import pandas as pd

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from datetime import datetime

import csv

import os

import pandas_datareader as web

import matplotlib.pyplot as plt

import numpy as np
# set constant variables

TICKER = "Ticker"

REPORT_DATE = "Report Date"

PUBLISH_DATE = "Publish Date"

FISCAL_YEAR = "Fiscal Year"

DATE = "Date"

DATEPARSER = lambda x: datetime.strptime(x, "%d/%m/%Y")
# get financial statements

def get_fs(ticker, varient="annual", market="us", method="offline"):

    ticker = ticker.lower()

    if method == "online":

        us_pl = sf.load(dataset='income', variant=varient, market=market,

                        index=[TICKER, FISCAL_YEAR],

                        parse_dates=[REPORT_DATE, PUBLISH_DATE], refresh_days=1)



        us_bs = sf.load(dataset='balance', variant=varient, market=market,

                        index=[TICKER, FISCAL_YEAR],

                        parse_dates=[REPORT_DATE, PUBLISH_DATE], refresh_days=1)



        us_cf = sf.load(dataset='cashflow', variant=varient, market=market,

                        index=[TICKER, FISCAL_YEAR],

                        parse_dates=[REPORT_DATE, PUBLISH_DATE], refresh_days=1)

        df_pl = us_pl.loc[ticker]

        df_bs = us_bs.loc[ticker]

        df_cf = us_cf.loc[ticker]

    # show the properties

    elif method == "offline":

        df_pl = pd.read_csv("/kaggle/input/{0}-financial-statement/income_{1}.csv".format(ticker, varient), sep=";").set_index("Fiscal Year")

        df_bs = pd.read_csv("/kaggle/input/{0}-financial-statement/balance_{1}.csv".format(ticker, varient), sep=";").set_index("Fiscal Year")

        df_cf = pd.read_csv("/kaggle/input/{0}-financial-statement/cashflow_{1}.csv".format(ticker, varient), sep=";").set_index("Fiscal Year")

    # print("income statement", df_pl.columns)

    # print("balance sheet", df_bs.columns)

    # print("cash flow", df_cf.columns)



    return df_pl, df_bs, df_cf
def get_current_price():

    curr_price = sf.load_shareprices(variant="latest", market='us',refresh_days=1)

    return curr_price

# assess to stock price

def get_stock_price(ticker,start_date="2018-1-1"):

    start_date = datetime.strptime(start_date, "%Y-%m-%d")

    stock_price = web.DataReader(ticker, "yahoo", start_date)

    return stock_price
def cal_CAGR(start_year, end_year, df):

    """

    calculate the Compounded Annual Growth Rate of df between start_year and end_year

    """

    cagr = (df.loc[end_year] / df.loc[start_year]) ** (1 / (end_year - start_year)) - 1

    return cagr





def cal_GR(start_year, end_year, df):

    """

    calculate the year on year growth rate of df between start_year and end_year

    """

    gr = df.loc[int(end_year)] / df.loc[int(start_year)] - 1

    return gr
def CAMP():

    # Goldman's estimation

    sp_return = 0.12

    # from yahoo finance 

    beta = 1.17

    # https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield

    rf = 0.0156

    # excepted return = risk-free rate + beta *(expected market return - risk-free rate)

    expected_return = rf + beta * (sp_return - rf)

    

    return expected_return
def get_key_data(df_pl, df_bs, df_cf, curr_price, ticker):

    """

    get some necessary data that need to calcualted, 

    e.g. BV,FCF, EPS, BVPS, FCFPS, D/E, current ratio etc

    """

    df_ks = pd.DataFrame(index=df_pl.index)

    df_ks["Book Value"] = df_bs["Total Assets"] - df_bs["Total Liabilities"]

    df_ks["Free Cash Flow"] = df_pl["Operating Income (Loss)"] + df_cf["Depreciation & Amortization"] + df_cf[

        "Change in Fixed Assets & Intangibles"] + df_cf["Change in Working Capital"] - df_pl[

                                  "Income Tax (Expense) Benefit, Net"]

    

    # the data of Free Cash Flow in 2019

    if ticker == "BIIB":

        df_ks["Free Cash Flow"].iloc[-1] = 6564000000 # from simfin

    elif ticker == "AMGN":

        df_ks["Free Cash Flow"].iloc[-1] = 8532000000 # from simfin

    elif ticker == "GILD":

        df_ks["Free Cash Flow"].iloc[-1] = 2525000000 # from simfin

    elif ticker == "PFE":

        df_ks["Free Cash Flow"].iloc[-1] = 12669000000 # from simfin

        

    df_ks["Shares Outstanding"] = df_bs["Shares (Diluted)"]

    df_ks["Earning Per Share"] = df_pl["Net Income"] / df_ks["Shares Outstanding"]

    df_ks["Book Value Per Share"] = df_ks["Book Value"] / df_ks["Shares Outstanding"]

    df_ks["Free Cash Flow Per Share"] = df_ks["Free Cash Flow"] / df_ks["Shares Outstanding"]

    df_ks["Sales Per Share"] = df_pl["Revenue"] / df_ks["Shares Outstanding"]

    df_ks["Gross Profit Margin "] = df_pl["Gross Profit"] / df_pl["Revenue"]

    df_ks["Net Income Margin"] = df_pl["Net Income"] / df_pl["Revenue"]

    df_ks["Debt to Equity"] = df_bs["Total Liabilities"]/df_bs["Total Equity"]

    df_ks["Current Ratio"] = df_bs["Total Current Assets"] / df_bs["Total Current Liabilities"]

    df_ratio = get_price_multiples(curr_price, df_ks)

    

    return df_ks, df_ratio
def get_price_multiples(curr_price, df_ks):

    """

    Calculating "P/E", "Forward P/E", "PEG", "P/B", "P/S", "P/FCF" 

    Retrun DataFrame

    """

    #DataFrame to hold those price multiples

    df_ratio = pd.DataFrame(

        index=["P/E", "Forward P/E", "PEG", "P/B", "P/S", "P/FCF"])

    start_year = df_ks.index.values[0]

    end_year = df_ks.index.values[-1]

    

    # P/E ratio, Formula: Current stock price/ Earning Per Share(TTM)

    pe = curr_price / (df_ks["Earning Per Share"].values[-1])

    # Foward P/E ratio, Formula: Current stock price/ Expected Earning Per Share

    f_pe = curr_price/ df_ks["Earning Per Share"].values[-1] / (1 + cal_CAGR(start_year, end_year,

                                                                                         df_ks["Earning Per Share"]))

    # P/BV ratio, Fomula: Current stock price/ Book Value Per Share(TTM)

    pbv = curr_price / df_ks["Book Value Per Share"].values[-1]

    # P/S ratio, Fomula: Current stock price/ Sales Per Share(TTM)

    ps = curr_price / df_ks["Sales Per Share"].values[-1]

    # P/FCF ratio, Fomula: Current stock price/ Free Cash Flow(TTM)

    pfcf = curr_price / df_ks["Free Cash Flow Per Share"].values[-1]

    # P/E growth Fomula: P/E / Expected Growth Earning /100

    peg = curr_price / df_ks["Earning Per Share"].values[-1] / (

            cal_CAGR(start_year, end_year, df_ks["Earning Per Share"]) * 100)

    

    df_ratio["Value"] = [pe, f_pe, peg, pbv, ps, pfcf]

    

    return df_ratio
def get_info_with_competitors(ticker, ticker_ks, ticker_ratio, competitors=["GILD", "AMGN", "PFE"]):

    """

    return the price mutilples, net income margin, current ratio, D/E ratio of BIIB and its three main rivals "GILD", "AMGN" and "PFE".

    """

    

    # hold price mutilples

    sector_pm = []

    # holde net income margin

    sector_nim = []

    # hold current ratio

    sector_curr_ratio = []

    # hold D/E ratio

    sector_de_ratio = []

    

    # get competitors' data

    for com_ticker in competitors:

        if com_ticker == "AMGN":

            cp = 222.14

        elif com_ticker == "GILD":

            cp = 67.00

        elif com_ticker == "PFE":

            cp = 35.85

        pl, bs, cf = get_fs(com_ticker)

        ks, pm = get_key_data(pl, bs, cf, cp, ticker=com_ticker)

        sector_pm.append(pm["Value"])

        sector_nim.append(ks["Net Income Margin"])

        sector_curr_ratio.append(ks["Current Ratio"])

        sector_de_ratio.append(ks["Debt to Equity"])

        

    sector_pm = pd.concat(sector_pm, axis=1).set_index(pm.index)

    sector_pm.columns = competitors

    sector_pm[ticker] = ticker_ratio["Value"]

    # 2 decimal places

    sector_pm = sector_pm.round(2)



    # set the average price multipiles from https://www.gurufocus.com/industry_overview.php?industry=Drug-Manufacturers

    sector_pm["Sector Average"] = [51.43, 30.21, 3.08, 7.14, 0.97, 43.49]

    

    # add BIIB as the name of last column

    competitors.append(ticker)

    

    sector_nim.append(ticker_ks["Net Income Margin"])

    sector_nim = pd.concat(sector_nim, axis=1).set_index(ticker_ks.index)

    sector_nim.columns = competitors



    sector_curr_ratio.append(ticker_ks["Current Ratio"])

    sector_curr_ratio = pd.concat(sector_curr_ratio, axis=1).set_index(ticker_ks.index)

    sector_curr_ratio.columns = competitors



    sector_de_ratio.append(ticker_ks["Debt to Equity"])

    sector_de_ratio = pd.concat(sector_de_ratio, axis=1).set_index(ticker_ks.index)

    sector_de_ratio.columns = competitors

    

    return sector_pm, sector_nim, sector_curr_ratio, sector_de_ratio
def draw_candlestick(df_price, events):

    """

    draw the candlestick with events showing on

    """

    fig = go.Figure()

    fig.add_trace(go.Candlestick(

        name="BIIB",

        x=df_price.index,

        open=df_price['Open'],

        high=df_price['High'],

        low=df_price['Low'],

        close=df_price['Close'],

    ))

    """

    plot the events

    """

    for event in events.values:

        date = event[0]

        content = event[1]

        fig.add_annotation(

            x=date,

            y=df_price.iloc[df_price.index == date]["Open"].values[0],

            text=content, arrowhead=3, font_size=16)

    fig.update_layout(height=700, showlegend=False)

    fig.show()
def draw_ni_eps(df_pl, df_ks):

    """

    draw Net Income and EPS 

    """

    fig = make_subplots(2, 1)

    fig.add_trace(go.Bar(x=df_pl.index, y=df_pl["Net Income"].values, name="Net Income"), row=1, col=1)

    fig.add_trace(go.Bar(x=df_ks.index, y=df_ks["Earning Per Share"], name="Earning Per Share"), row=2, col=1)

    # save the YOY growth Rate

    yoygr = [0.]

    for i in range(1, len(df_ks.index)):

        yoygr.append(df_ks["Earning Per Share"].iloc[i] / df_ks["Earning Per Share"].iloc[i - 1] - 1)

    yoygr

    fig.show()
def draw_product_sales():

    """

    draw the sales of drugs in 2018 and 2019

    """

    ps = pd.read_csv("/kaggle/input/biib-financial-statement/product_sales.csv", sep=";", index_col="Drug Name")

    fig = go.Figure()

    fig.add_trace(go.Bar(x=ps.index, y=ps["2018"], name="2018"))

    fig.add_trace(go.Bar(x=ps.index, y=ps["2019"], name="2019"))

    fig.show()
def draw_peers_nim(df_sector_em):

    """

    showing Net Income Magrin of Biogen and its competitors.

    """

    traces = []

    for c_name in df_sector_em.columns:

        traces.append(go.Scatter(x=df_sector_em.index, y=df_sector_em[c_name], name=c_name))

    fig = go.Figure(data=traces)

    fig.show()
def draw_peers_pm(df_sector_ratio):

    """

    showing Price multiples of Biogen and its competitors.

    """

    fig = go.Figure(data=[go.Table(

        columnorder=[1, 2, 3, 4, 5, 6],

        columnwidth=[10, 10, 10, 10, 10, 10],

        header=dict(values=list(df_sector_ratio.columns.insert(0, " ")),

                    fill_color='royalblue',

                    line_color='darkslategray',

                    align=['left', 'center'],

                    font=dict(color='white', size=12)),

        cells=dict(

            values=[df_sector_ratio.index, df_sector_ratio["GILD"], df_sector_ratio["AMGN"], df_sector_ratio["PFE"],

                    df_sector_ratio["BIIB"], df_sector_ratio["Sector Average"]],

            fill=dict(color=['paleturquoise', 'white', 'white', 'white', 'white', 'white']),

            align=['left', 'center'], font_size=12, line_color='darkslategray', ))

    ])

    fig.show()
ticker = "BIIB"

# get BIIB's financial statements 

df_pl, df_bs, df_cf = get_fs(ticker)
# show candlestick and events

stock_price = get_stock_price(ticker, start_date="2019-01-01")

events = pd.read_csv("/kaggle/input/biib-financial-statement/events.csv", sep=";", date_parser=DATEPARSER,parse_dates=["Date"])

draw_candlestick(stock_price, events)
curr_price = get_stock_price(ticker)

df_ks, df_pm = get_key_data(df_pl, df_bs, df_cf, curr_price["Close"].iloc[-1], ticker)
# show Net Income and EPS

draw_ni_eps(df_pl, df_ks)
# show the numbers of Shares Outstanding 

df_ks["Shares Outstanding"].plot()
# get key data of BIIB and its rivals

df_sector_pm, df_sector_nim, df_sector_cr, df_sector_de = get_info_with_competitors(ticker, df_ks, df_pm)
# show Net Income Margin

draw_peers_nim(df_sector_nim)
# show price mutilples 

draw_peers_pm(df_sector_pm)
# show BIIB's sales of drugs

draw_product_sales()
# show BIIB's D/E

df_sector_de.plot()
# calculate target price

# set Compound Annual Growth Rate

cagr = cal_CAGR(2015, 2019, df_ks["Earning Per Share"])

print("Compounded Annual Growth Rate", cagr)

# set current EPS

current_eps = df_ks["Earning Per Share"].iloc[-1]

print("Current EPS", current_eps)

# cal future EPS 10 year from now

# formula: current EPS * (1+Compounded Annual Growth Rate) **10

future_eps = current_eps * (1 + cagr) ** 10

print("Future EPS", future_eps)

# set Average P/E of sector

average_pe = np.mean(df_sector_pm.loc["P/E"][:-1])

print("Average PE", average_pe)

# cal future Stock Price 10 year from now

# formula: future EPS * Average P/E

future_price = future_eps * average_pe

print("Future Stock Price", future_price)

# set discount rate

discount_rate = CAMP()

print("discount rate", discount_rate)

# cal present stock price by discount future stock price

# formula: future stock price / ((1+discount rate)**10)

present_price = future_price / ((1 + discount_rate) ** 10)

print("Present Stock Price", present_price)

# set magin safety

safe_margin = 0.4

# get target buy price

# formula: present stock price * (1-margin safety)

print("target buy price {:.2f}".format(present_price*(1-safe_margin)))