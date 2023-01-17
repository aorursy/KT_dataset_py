#Data 

revenue = [14574.49, 7606.46, 8611.41, 9175.41, 8058.65, 8105.44, 11496.28, 9766.09, 10305.32, 14379.96, 10713.97, 15433.50]

expenses = [12051.82, 5695.07, 12319.20, 12089.72, 8658.57, 840.20, 3285.73, 5821.12, 6976.93, 16618.61, 10054.37, 3803.96]



#importing libraries required

import numpy as np



#Solution

profit = []

netProfit = []

profitMargin = []

goodMonths = []

badMonths = []



for i in range(len(revenue)):

    profit.append(round(revenue[i] - expenses[i],2))

    netProfit.append(round((profit[i] - profit[i]*0.30),2))

    profitMargin.append(round((netProfit[i]/revenue[i])*100))

    

meanOfYear = np.mean(netProfit)

maxOfYear = np.max(netProfit)

minOfYear = np.min(netProfit)

bestMonth = []

worstMonth = []



for i in range(len(netProfit)):

    if netProfit[i] > meanOfYear:

        goodMonths.append(i+1)

    elif netProfit[i] < meanOfYear:

        badMonths.append(i+1)

    if netProfit[i] == np.max(netProfit) :

        bestMonth = i+1

    elif netProfit[i] == np.min(netProfit) :

        worstMonth = i+1



print("profits for each month:\n", profit, "\n")

print("profits after tax for each month:\n", netProfit, "\n")

print("profit margin for each month:\n", profitMargin, "\n")

print("good months:\n", goodMonths,"\n")

print("bad months:\n", badMonths,"\n")

print("best month:\n", bestMonth,"\n")

print("worst month:\n", worstMonth,"\n")