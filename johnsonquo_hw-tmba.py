import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import datetime

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def FindMDD(Nav):

    maxNav = Nav[0]

    mdd = 0

    listLen = len(Nav)

    for i in range(1,listLen):

        if Nav[i] > maxNav:

            maxNav = Nav[i]

        dd = maxNav - Nav[i]

        if dd > mdd:

            mdd = dd

    return mdd

    
stock = pd.read_csv("../input/stcok.csv")
stock.head()
codeList = stock["Code"].unique()

retTable = pd.DataFrame()

for codeName in codeList:

    stockRet = stock[stock["Code"] == codeName][["Date", "Ret"]]

    if(retTable.empty):

        retTable = stockRet

    else:

        retTable = pd.merge(retTable,stockRet, on="Date")

retTable.columns = np.append("Date",codeList)
retTable.corr()
handlingFee = 0.001425

handlingDiscount = 0.5

taxRate = 0.003

contract = 1

n = contract*1000

initailAmount = 1000000

code1 = 9924

code2 = 9918
##暫存變數設定/宣告

hedgeSignalTable = pd.DataFrame()

dateList1 = stock[stock["Code"] == code1]["Date"]

dateList2 = stock[stock["Code"] == code2]["Date"]

dateList = list(set(dateList1).intersection(set(dateList2)))

dateList = sorted(dateList, key=lambda date: datetime.datetime.strptime(date, "%Y/%m/%d"))

signal = 0

tradeCounter = 0

cash = initailAmount

currentAmount = 0

lastSignal = 0

priceRatio = 0

currentTradePrice1 = 0

currentTradePrice2 = 0
for date in dateList:

    stock1 = stock[(stock["Date"] == date) & (stock["Code"] == code1)]

    stock2 = stock[(stock["Date"] == date) & (stock["Code"] == code2)]

    tradeCost = 0

    

    ##目前持有獲利

    recordProfit = 0

    profit = (stock1["Open"].values[0] - currentTradePrice1) * n * lastSignal - (stock2["Open"].values[0]-currentTradePrice2) * n * lastSignal * priceRatio

    priceRatio = (currentTradePrice1/currentTradePrice2) if (currentTradePrice2 != 0) else 0 

    ##

    

    ##紀錄交易編號及成本與獲利紀錄

    if ((lastSignal != signal) & (signal !=0)):

        currentTradePrice1 = stock1["Open"].values[0]

        currentTradePrice2 = stock2["Open"].values[0]##紀錄買進金額

        tradeCounter+=1

        tradeCost = 2*(2*n*stock1["Open"].values[0]*handlingFee*handlingDiscount + n*stock1["Open"].values[0]*taxRate) 

        cash = cash+profit-tradeCost

        recordProfit = profit

        profit = 0

        

        

    elif((lastSignal != signal) & (signal ==0)):

        currentTradePrice1 = 0

        currentTradePrice2 = 0

        tradeCost = 2*(n*stock1["Open"].values[0]*handlingFee*handlingDiscount + n*stock1["Open"].values[0]*taxRate)

        cash = cash+profit-tradeCost

        recordProfit = profit

        profit = 0

        priceRatio = 0

        

    tradeID = int(np.where(signal !=0 , tradeCounter, 0))

    currentAmount = cash + profit

    profit += recordProfit

    ##

    

    hedgeSignalTable = hedgeSignalTable.append({"Date":date,"Code1Price":stock1["Open"].values[0],"Code2Price":stock2["Open"].values[0],"Signal": signal, "tradeID": tradeID, "tradeCost": tradeCost, "profit": profit,"currentAmount":currentAmount, "StartPrice1": currentTradePrice1, "StartPrice2":currentTradePrice2,"cash":cash},ignore_index=True)

     

        ##交易訊號

    lastSignal = signal

    if(stock1["Close"].values[0]>stock1["MA10"].values[0]):

        signal = 1

    elif(stock1["Close"].values[0]<stock1["MA10"].values[0]):

        signal = -1

    else:

        signal = 0     

    ##  
hedgeSignalTable.head(500)
plt.plot(hedgeSignalTable['Date'],hedgeSignalTable['currentAmount'])
tradeDetailTable = hedgeSignalTable[hedgeSignalTable["tradeID"]!= 0].groupby("tradeID").agg({"StartPrice1": "last","Code1Price":"last","StartPrice2": "last","Code2Price":"last", "profit":["min","max","first"],"Signal":"last"})

tradeDetailTable.columns = ["Stock1_StartPrice", "Stock1_EndPrice","Stock2_StartPrice", "Stock2_EndPrice","MinProfit","MaxProfit","Profit","Stock1_BuySell"]

tradeDetailTable["Stock2_BuySell"] = np.where(tradeDetailTable["Stock1_BuySell"]*-1>0, "Buy","Sell")

tradeDetailTable["Stock1_BuySell"] = np.where(tradeDetailTable["Stock1_BuySell"]>0 ,"Buy","Sell")

tradeDetailTable["Profit"] = tradeDetailTable["Profit"].shift(-1)

tradeDetailTable = tradeDetailTable[["Stock1_BuySell","Stock1_StartPrice", "Stock1_EndPrice", "Stock2_BuySell", "Stock2_StartPrice", "Stock2_EndPrice", "Profit", "MinProfit","MaxProfit"]]

tradeDetailTable.head()
totalNetProfit = hedgeSignalTable["currentAmount"].iloc[-1]-initailAmount

tradeTimes = len(tradeDetailTable)

winRate = len(tradeDetailTable[tradeDetailTable["Profit"]>0]) / len(tradeDetailTable)

ave_Profit = tradeDetailTable[tradeDetailTable["Profit"]>0]["Profit"].mean()

ave_Loss = tradeDetailTable[tradeDetailTable["Profit"]<0]["Profit"].mean()

max_Profit = tradeDetailTable["MaxProfit"].max()

max_Loss = tradeDetailTable["MinProfit"].min()

totalTradeCost = hedgeSignalTable["tradeCost"].sum()

profitFactor = tradeDetailTable[tradeDetailTable["Profit"]>0]["Profit"].sum() / tradeDetailTable[tradeDetailTable["Profit"]<0]["Profit"].sum()*(-1)

mdd = FindMDD(hedgeSignalTable["currentAmount"])

return_risk_Ratio = totalNetProfit/mdd
print("總獲利 : ",totalNetProfit)

print("總交易次數", tradeTimes)

print("總交易成本 : ",totalTradeCost)

print("勝率 : ",winRate)

print("平均獲利 : ", ave_Profit)

print("平均虧損 : ",ave_Loss)

print("獲利因子 : ",profitFactor)

print("mdd : ", mdd)

print("風報比 : ",return_risk_Ratio)