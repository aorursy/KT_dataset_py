# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import csv



LABACSVFileName = "LABA_dataset3000.csv"



class Keys:

    InvoiceNo='InvoiceNo'

    StockCode='StockCode'

    Description='Description'

    Quantity='Quantity'

    InvoiceDate='InvoiceDate'

    UnitPrice='UnitPrice'

    CustomerID='CustomerID'

    Country='Country'



keys = Keys()



class LABASolver:

    pathToLABACSVDataset = ""

    dataDict = []

    def readCSV(self):

        csvfile = open(self.pathToLABACSVDataset, newline='')

        self.dataDict = list( csv.DictReader(csvfile) )

        csvfile.close()

    def setupPath(self, fileName):

        for dirname, _, filenames in os.walk('/kaggle/input'):

            for filename in filenames:

                if( filename == fileName ):

                    self.pathToLABACSVDataset = os.path.join(dirname, filename)

                    self.readCSV()

                    for line in self.dataDict:

                        print( "line:" + str( line ) )

                        break

                    break

    

    

    def calcUniqueInvoice(self):

        invoiceSet = set()

        for line in self.dataDict:

            invoiceSet.add( line['InvoiceNo'] )

        return len( invoiceSet )

        

        

    def calcBestProduct(self):

        productWithCount = {}

        for line in self.dataDict:

            productWithCount[line[keys.StockCode]] = productWithCount.setdefault( line[keys.StockCode], 0 ) + 1

        sortProducts = sorted(productWithCount.items(), key=lambda x: x[1], reverse=True)

        return sortProducts[0][0]

        

        

    def calcTotalPriceByProduct(self, productID):

        total = 0;

        for line in self.dataDict:

            if line[keys.StockCode] == productID:

                total += float( line[keys.UnitPrice] ) * float( line[keys.Quantity] )

        return total

        

    def calcTotalCountOfProductsByUser(self, userID):

        countOfItems = 0

        for line in self.dataDict:

            if line[keys.CustomerID] == userID:

                countOfItems += int( line[keys.Quantity] )

        return countOfItems

    

    def calcTotalCountOfProducts(self):

        countOfItems = 0

        for line in self.dataDict:

            countOfItems += int( line[keys.Quantity] )

        return countOfItems

    

    def calcPriceOfProductByUser(self, productID, userID):

        price=[]

        for line in self.dataDict:

            if line[keys.StockCode] == productID and line[keys.CustomerID] == userID:

                price.append( line[keys.UnitPrice] )

        return price

    

    def findClientWithMostInvoice(self):

        clientsWithInvoiceHash = {}

        for line in self.dataDict:

            client = line[keys.CustomerID]

            if client == "":

                continue

            if not client in clientsWithInvoiceHash:

                clientsWithInvoiceHash[client] = set()

            clientsWithInvoiceHash[client].add( line[keys.InvoiceNo] )

        clientsWithUniqueIncoiceCount = {}

        for pair in clientsWithInvoiceHash:

            clientsWithUniqueIncoiceCount[pair] = len( clientsWithInvoiceHash[pair] )

        sortedValue = sorted(clientsWithUniqueIncoiceCount.items(), key=lambda x: x[1], reverse=True)

        return sortedValue[0][0]

        

        print( "not defined" )

    

    def calcTotalCountOfInvoice(self):

        totalCountInvoice = set()

        for line in self.dataDict:

            totalCountInvoice.add( line[keys.InvoiceNo] )

        return len( totalCountInvoice )



    def calcTotalPriceOfProducts(self):

        totalPrice = 0

        for line in self.dataDict:

            price = float( line[keys.UnitPrice] )

            quantity = float( line[keys.Quantity] )

            if quantity > 0:

                totalPrice += price * quantity

        return totalPrice

        

    def calcTotalPriceOfProductsWithRejects(self):

        

        totalPrice = 0

        for line in self.dataDict:

            price = float( line[keys.UnitPrice] )

            quantity = float( line[keys.Quantity] )

            totalPrice += price * quantity

        return totalPrice

    

    def calcPerventOfSellsWithRegistration(self):

        totalCountOfProducts = solver.calcTotalCountOfProducts()

        countOfUnregisteredProductsSell = solver.calcTotalCountOfProductsByUser( '' )

        percentOfProductByRegisteredUsers = None

        if totalCountOfProducts and totalCountOfProducts != 0:

            percentOfProductByRegisteredUsers = ( totalCountOfProducts - countOfUnregisteredProductsSell ) / totalCountOfProducts * 100

        return percentOfProductByRegisteredUsers











# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
solver = LABASolver()
solver.setupPath( LABACSVFileName )
print( "UniqueInvoice:" + str( solver.calcUniqueInvoice() ) )
bestProduct = solver.calcBestProduct()

print( "BestProduct:" + str( bestProduct ) + " and total price: " + str( solver.calcTotalPriceByProduct(bestProduct) ) )
print( "Count of product by unregistered user:" + str( solver.calcTotalCountOfProductsByUser( '' ) ) )
print( "Percent of product by registered user: " + str( solver.calcPerventOfSellsWithRegistration() ) + "%")
print( "Price of 85123A by unregistered user: " + str( solver.calcPriceOfProductByUser( '85123A', '' ) ) )
print( "Client with most invoice count: " + str( solver.findClientWithMostInvoice() ) )
print( "Total count of invoices: " + str( solver.calcTotalCountOfInvoice() ) )

print( "Total price of products sell: " + str( solver.calcTotalPriceOfProducts() ) )

print( "Total price of products sell with rejects: " + str( solver.calcTotalPriceOfProductsWithRejects() ) )