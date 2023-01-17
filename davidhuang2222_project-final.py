import requests
from bs4 import BeautifulSoup
tickers=[]
market_capital = []
market_capital_adjust = []
market_capital_adjust_new = []
prices=[]
fixprices=[]
YTDpercentchange=[]
holder = []
changed = []
changedholder = []
fixchangeholder = []
new = []
news = []
newx = []
newy = []
holdthis = []
final = []

for x in range(1,6):
    url = "https://money.cnn.com/data/sectors/tech/communications/?sector=4900&industry=4910&page={}".format(x)
    pages = requests.get(url)
    soup2 = BeautifulSoup(pages.content)
    #Find Ticker Symbol
    rows = soup2.find_all('td')
    for row in rows:
        cells = row.find('a')
        if cells:
            for cell in cells:
                title = cell
                tickers.append(title)
    #Find Market Capital
    tr = soup2.find_all('tr')
    for marketc in tr:
            marketcap = marketc.find('td',"wsod_aRight")
            if marketcap:
                for money in marketcap:
                    moneybag = money
                    market_capital.append(moneybag)
    #Find Price
    for p in rows:
        price = p.find('span')
        if price:
            for x in price:
                blank = x
                prices.append(blank)
    #Find the change in price
    for change in rows:
        look = change.find('span',"posData")
        if look:
            for looks in look:
                variable = looks
                changed.append(variable)
    #negative changes were in diff location
    for negative in rows:
        neg = negative.find('span',"negData")
        if neg:
            for negs in neg:
                l = negs
                changed.append(l)

#set a holder so that data won't interrupt other code
find = '%'
changedholder = changed
for remove in changedholder:
    str = find in remove
    if str == True:
        changedholder.remove(remove)

#had to clean data
for fix in prices:
     if len(fix) > 1:
        fixprices.append(fix)
#had to move the % away from the actual price
find = '%'
for loop in fixprices:
    res = [loop for loop in fixprices if find in loop]
    
#input YTD
for insert in res:
    YTDpercentchange.append(insert)
#include the --
YTDpercentchange.insert(0,'0.00%')
YTDpercentchange.insert(1,'0.00%')
YTDpercentchange.insert(2,'0.00%')
YTDpercentchange.insert(4,'0.00%')
YTDpercentchange.insert(8,'0.00%')
YTDpercentchange.insert(9,'0.00%')
YTDpercentchange.insert(10,'0.00%')
YTDpercentchange.insert(14,'0.00%')
YTDpercentchange.insert(15,'0.00%')
YTDpercentchange.insert(17,'0.00%')
YTDpercentchange.insert(23,'0.00%')
YTDpercentchange.insert(27,'0.00%')
YTDpercentchange.insert(32,'0.00%')
YTDpercentchange.insert(42,'0.00%')
YTDpercentchange.insert(48,'0.00%')
YTDpercentchange.insert(54,'0.00%')
YTDpercentchange.insert(58,'0.00%')
YTDpercentchange.insert(63,'0.00%')
YTDpercentchange.insert(66,'0.00%')
YTDpercentchange.insert(68,'0.00%')
YTDpercentchange.insert(72,'0.00%')
YTDpercentchange.insert(77,'0.00%')
YTDpercentchange.insert(78,'0.00%')

#there were strings in the return so to clean it I removed the strings with len
for adjust in market_capital:
    if len(adjust)>1:
        market_capital_adjust.append(adjust)
#some strings were in prices so i removed strings that were greater than 6 len because no number
#was greater than 5 in len
for edit in fixprices:
    if len(edit) > 6:
        fixprices.remove(edit)
holder = fixprices
#removed strings in my list
for remove in holder:
    str = find in remove
    if str == True:
        fixprices.remove(remove)
#Remove % from price change
fixedchangeholder = changedholder
for excuse in fixedchangeholder:
    str = find in excuse
    if str == True:
        fixedchangeholder.remove(excuse)
        
#mannually added to reach 81, missing data
for manually in range(1,41):
    fixedchangeholder.append("0.00")
#create a dictionary to remove duplicates
tickers = list(dict.fromkeys(tickers))

#if the stock for top movers change I must change the values in the equals
for topmover1 in fixprices:
    if topmover1 == '5.32':
        fixprices.remove(topmover1)
for topmover2 in fixprices:
    if topmover2 == '11.11':
        fixprices.remove(topmover2)
for topmover3 in fixprices:
    if topmover3 == '8.99':
        fixprices.remove(topmover3)
for bottommover1 in fixprices:
    if bottommover1 == '7.15':
        fixprices.remove(bottommover1)
#for bottommover2 in fixprices:
#    if bottommover2 == '23.13':
#        fixprices.remove(bottommover2)
#for bottommover3 in fixprices:
#    if bottommover3 == '17.33':
#        fixprices.remove(bottommover3)
#Have to remember to edit this when it changes for top mover
for last in fixprices:
        if last == 'WideOpenWest Inc':
            fixprices.remove(last)
fixprices.remove(fixprices[2])

#remove the top and bottom movers
tickers.remove(tickers[0])
tickers.remove(tickers[1])
tickers.remove(tickers[2])
tickers.remove(tickers[4])
tickers.remove(tickers[0])
tickers.remove(tickers[0])
#I had to add N/A to make the list 81 but due to the top movers repeating I had to remove the actual values and couldn't add them back
tickers.insert(0,"SVNJ")
tickers.insert(2,"APWL")
tickers.insert(3,"AFFN")

#have to remove the signs so that they can be used for visualization
moos = '$'
moof = 'M'
moofs = 'K'
moofies = 'B'
for empty in market_capital_adjust:
    for loops in empty:
        moo = [loops for loops in empty if moos in loops]
    if moo:
        pop = empty.replace('$','')
        new.append(pop)
        
for empties in new:
    for loopity in empties:
        mow = [loopity for loopity  in empties if moof in loopity]
    if mow:
        pops = empties.replace('M','')
        news.append(pops)

for empty2 in new:
    for loopitys in empty2:
        mows = [loopitys for loopitys in empty2 if moofs in loopitys]
    if mows:
        popsy = empty2.replace('K','')
        newx.append(popsy)
for empties2 in new:
    for loopityx in empties2:
        mowz = [loopityx for loopityx  in empties2 if moofies in loopityx]
    if mowz:
        popsz = empties2.replace('B','')
        newy.append(popsz)
shabloosh = []        
#couldn't append these with the code above
for bleh in new:
    if bleh == '23.00':
        shabloosh.append(bleh)
final = shabloosh + news + newx + newy
#the values that didn't have $ sign that weren't caught in the find
final.insert(3,'5.00')
final.insert(54,'183.00')
final.insert(9,'0.00')
final.insert(26,'0.00')
final.insert(42,'0.00')

#in order to visualize data no strings allowed.
fixprices[41] = '0.00'
fixprices[42] = '0.00'
fixprices[43] = '0.00'
fixprices[44] = '0.00'
tickers.insert(36,'GTT')
tickers.insert(79,'WOW')
fixprices.insert(79,'5.32')
        
if len(tickers) > 81:
    tickers.remove(tickers[0])
print("Market Capital Numeric:")
print(final)
print(len(final))
print("Market Capital:")
print(market_capital_adjust)
print(len(market_capital_adjust))
print("Stock Tickers:")
print(tickers)
print(len(tickers))
print("Prices:")
print(fixprices)
print(len(fixprices))
print("YTD Percent Change")
print(YTDpercentchange)
print(len(YTDpercentchange))
print("Price Change:")
print(fixedchangeholder)
print(len(fixedchangeholder))
#sorting since after I removed the marketcapital k,m,b it doesn't put it into order of each stock
sortinglist = final
sortmarketcapital = []
for add in range(1,83):
    sortmarketcapital.append(add)
sortmarketcapital[0] = sortinglist[0]
sortmarketcapital[1] = sortinglist[35]
sortmarketcapital[2] = sortinglist[36]
sortmarketcapital[3] = sortinglist[3]
sortmarketcapital[4] = sortinglist[37]
sortmarketcapital[5] = sortinglist[2]
sortmarketcapital[6] = sortinglist[39]
sortmarketcapital[7] = sortinglist[40]
sortmarketcapital[8] = sortinglist[4]
sortmarketcapital[9] = sortinglist[41]
sortmarketcapital[10] = sortinglist[9]
sortmarketcapital[11] = sortinglist[43]
sortmarketcapital[12] = sortinglist[44]
sortmarketcapital[13] = sortinglist[4]
sortmarketcapital[14] = sortinglist[76]
sortmarketcapital[15] = sortinglist[45]
sortmarketcapital[16] = sortinglist[5]
sortmarketcapital[17] = sortinglist[6]
sortmarketcapital[18] = sortinglist[46]
sortmarketcapital[19] = sortinglist[47]
sortmarketcapital[20] = sortinglist[77]
sortmarketcapital[21] = sortinglist[7]
sortmarketcapital[22] = sortinglist[8]
sortmarketcapital[23] = sortinglist[10]
sortmarketcapital[24] = sortinglist[48]
sortmarketcapital[25] = sortinglist[9]
sortmarketcapital[26] = sortinglist[49]
sortmarketcapital[27] = sortinglist[11]
sortmarketcapital[28] = sortinglist[12]
sortmarketcapital[29] = sortinglist[50]
sortmarketcapital[30] = sortinglist[13]
sortmarketcapital[31] = sortinglist[14]
sortmarketcapital[32] = sortinglist[78]
sortmarketcapital[33] = sortinglist[15]
sortmarketcapital[34] = sortinglist[16]
sortmarketcapital[35] = sortinglist[51]
sortmarketcapital[36] = sortinglist[52]
sortmarketcapital[37] = sortinglist[17]
sortmarketcapital[38] = sortinglist[18]
sortmarketcapital[39] = sortinglist[19]
sortmarketcapital[40] = sortinglist[53]
sortmarketcapital[41] = sortinglist[9]
sortmarketcapital[42] = sortinglist[20]
sortmarketcapital[43] = sortinglist[21]
sortmarketcapital[44] = sortinglist[22]
sortmarketcapital[45] = sortinglist[54]
sortmarketcapital[46] = sortinglist[23]
sortmarketcapital[47] = sortinglist[55]
sortmarketcapital[48] = sortinglist[24]
sortmarketcapital[49] = sortinglist[25]
sortmarketcapital[50] = sortinglist[56]
sortmarketcapital[51] = sortinglist[27]
sortmarketcapital[52] = sortinglist[58]
sortmarketcapital[53] = sortinglist[57]
sortmarketcapital[54] = sortinglist[28]
sortmarketcapital[55] = sortinglist[29]
sortmarketcapital[56] = sortinglist[59]
sortmarketcapital[57] = sortinglist[60]
sortmarketcapital[58] = sortinglist[30]
sortmarketcapital[59] = sortinglist[31]
sortmarketcapital[60] = sortinglist[61]
sortmarketcapital[61] = sortinglist[79]
sortmarketcapital[62] = sortinglist[62]
sortmarketcapital[63] = sortinglist[63]
sortmarketcapital[64] = sortinglist[64]
sortmarketcapital[65] = sortinglist[65]
sortmarketcapital[66] = sortinglist[66]
sortmarketcapital[67] = sortinglist[67]
sortmarketcapital[68] = sortinglist[68]
sortmarketcapital[69] = sortinglist[69]
sortmarketcapital[70] = sortinglist[70]
sortmarketcapital[71] = sortinglist[71]
sortmarketcapital[72] = sortinglist[32]
sortmarketcapital[73] = sortinglist[80]
sortmarketcapital[74] = sortinglist[72]
sortmarketcapital[75] = sortinglist[73]
sortmarketcapital[76] = sortinglist[34]
sortmarketcapital[77] = sortinglist[74]
sortmarketcapital[78] = sortinglist[33]
sortmarketcapital[79] = sortinglist[34]
sortmarketcapital[80] = sortinglist[75]
sortmarketcapital[81] = sortinglist[76]
newfinal = sortmarketcapital
#tried to do percent change but it is located in different parts of website so i cannot format it to match the other variables
#for loop in rows:
    #changed = loop.find('span',"posChangePct")
    #print(changed)
    #if changed:
        #for maybe in changed:
            #f = maybe
            #percentchange.append(f)
import pandas as pd
import numpy as np
#Use pandas/numpy to clean and preprocess the dataset
column_names = ['Ticker', 'Market Capital Numeric','Market Capital', 'Price','YTD Percent Change','Price Change']
stocks = np.column_stack([tickers,newfinal,market_capital_adjust, fixprices,YTDpercentchange,fixedchangeholder])
stocksx = pd.DataFrame(stocks,columns=column_names)
stocksx

market_capital=stocksx['Market Capital Numeric'].astype(float)

market_capital.plot.bar(figsize=(15,15))
#most interesting point with highest market capital & ticker name
print(market_capital[26])
print(stocksx['Ticker'][26])
print(stocksx['Price'][26])
print(market_capital[35])
print(stocksx['Ticker'][35])
print(stocksx['Price'][35])
import matplotlib as plt
#histogram of price shows this sector of tech stocks have mainly low prices
#so investors with less money can buy in but this also shows risk as price wouldn't likely go up
fig = stocksx['Price'].hist(bins=50, figsize=(10,8), xrot = 90)
from pylab import rcParams
#scatter plot, does price relate to market capital? looks like strong and positive
stocky = stocksx.plot.scatter(x = 'Market Capital Numeric', y = 'Price', rot=90)
rcParams['figure.figsize'] = 10, 10
#changed dataframe column Price to float to visualize
price = stocksx['Price'].astype(float)
price.plot.bar(figsize=(15,15))
#most interesting point and stock ticker
print(price[21])
print(stocksx['Ticker'][21])
print(price[50])
print(stocksx['Ticker'][50])
print(price[51])
print(stocksx['Ticker'][51])
print(price[75])
print(stocksx['Ticker'][75])