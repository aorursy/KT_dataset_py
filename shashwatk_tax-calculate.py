# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def main():
    monthlySales = inputSales()
    countyTax = calcCounty(monthlySales)
    stateTax = calcState(monthlySales)
    totalTax = calcTotal(countyTax, stateTax)
    printInfo (countyTax, stateTax, monthlySales, totalTax)
def inputSales():
    monthlySales = float(input ("Enter sales for the month: "))
    return monthlySales
def calcCounty(monthlySales):
    countyTax = monthlySales * .025
    return countyTax
def calcState (monthlySales):
    stateTax = monthlySales * .05
    return stateTax
def calcTotal (countyTax, stateTax):
    totalTax = countyTax + stateTax
    return totalTax
def printInfo (countyTax, stateTax, monthlySales, totalTax):
    print ("There were $", monthlySales, "in sales")
    print ("The county tax is $", countyTax)
    print ("The state tax is $", stateTax)
    print ("The total tax is $", totalTax)
main()
