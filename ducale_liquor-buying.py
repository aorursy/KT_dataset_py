# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
print('numpy version\t:',np.__version__)
import pandas as pd
print('pandas version\t:',pd.__version__) # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" drirectory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import matplotlib.pyplot as plt
%matplotlib inline
from scipy import stats

# Regular expressions
import re

# seaborn : advanced visualization
import seaborn as sns
print('seaborn version\t:',sns.__version__)

# CPT modules

# import module we'll need to import our custom module
from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/liquorbuying/CPTp.py", dst = "../working/CPTp.py")
copyfile(src = "../input/liquorbuying/PredictionTree.py", dst = "../working/PredictionTree.py")

# import all our functions
from CPTp import *

#Machine learning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Chane the display option for bigger output
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 200
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 500)
        
# Any results you write to the current directory are saved as output.
!pip install names
# !pip install cpt
# !pip install cython cpt

import names
# names.get_full_name()
import random
def createList(r1, r2): 
    return list(range(r1, r2+1)) 
random.seed( 30 )
r1, r2 = 9000, 9100
# print(createList(r1, r2))
custId = createList(r1, r2)
# CustId

custName = []
for i in range(0,101):
    n = names.get_full_name()
    custName.append(n)
custData = {'custId': custId, 'custName': custName}    

dfcust = pd.DataFrame.from_dict(custData)

#create customer dictionary
custDict = pd.Series(dfcust['custName'].values,index=dfcust['custId']).to_dict()
print(dfcust, custDict)
##Register datafile
# "/kaggle/input/liquorbuying/MLMP-weeklysales-06.04 - 12.04.csv"
file = "/kaggle/input/liquorbuying/MLMP-weeklysales-01.01 - 12.04.csv"
file2 = "/kaggle/input/liquorbuying/productDatabase.csv"
df=pd.read_csv(file)
productDb = pd.read_csv(file2)
# df.shape()
# df.drop(df.index[df.count-1])
df = df.drop(['Customer', 'Register' ], axis=1) #drop columns with NAs and duplicated column
df = df.dropna()
df.shape
df

# productDb = productDb[['productId', 'Name']] 
productDbDict = pd.Series(productDb['Name'].values,index=productDb['productId']).to_dict()
productDbDict
print("The data has " + str(df.shape[0]) + " rows and " + str(df.shape[1]) + " columns" )
print(df.info()) # view all columns and datatype
# >>>Currency String to Float
ls = ["Revenue" , "Items Sold", "Discount Amount", "Sale Total"]
for x in ls:
    df[ls]=df[ls].replace('[\$,]', '', regex=True).astype(float)

# # >>>Percentage String to Percentage
# df["Profit Percentage"]=df["Profit Percentage"].replace('[\%,]', '', regex=True).astype(float)
# df["Profit Percentage"] = pd.Series([(val / 100) for val in df["Profit Percentage"]], index = df.index)
def ToPcnt(df, col, InHundred=True):
    if InHundred==True:
        df[col] = pd.Series(["{0:.2f}%".format(val) for val in df[col]], index = df.index)
    else:
        df[col] = pd.Series(["{0:.2f}%".format(val * 100) for val in df[col]], index = df.index)
    return df
# >>>String to Integer
df['Invoice Number'] = df['Invoice Number'].str.extract('(\d+)', expand=False) #remove all non-numeric char
df["Invoice Number"]= df["Invoice Number"].astype(int)

# >>>Object to Datetime
df['Date and Time'] = pd.to_datetime(df['Date and Time'], errors='coerce')

# pd.to_datetime(1490195805, unit='s')
df.info()

print(df.info())
print(df.User.unique())
print(df.columns)
# >>> rearrange columns order
df = df[['Invoice Number', 'Date and Time','Name',
        'Revenue', 'Cases Sold', 'Items Sold',
         'Discount Amount', 'Sale Total',
         'register', 'User'
       ]]
# >>> Rename columns
df = df.rename(columns={
    "Invoice Number": "invoiceNumber", 
    "Date and Time": "transactionDateTime",
    "Name":"productName",
    "Profit Percentage":"profitPercentage",
    "Cases Sold":"casesSold", 
    "Items Sold":"itemsSold"
})
random.seed(30)
# df['custId'] = 0
dfinvList = pd.DataFrame(pd.Series(df['invoiceNumber'].values).unique(), columns=['invoiceNumber']) #<<<extract unique value of invoice number
dfinvList
dfinvList['custId'] = [random.randint(9000,9100)  for k in dfinvList.index]
#left join from dfcust 
x = dfinvList.merge(dfcust, on='custId')
x.head()
df =df.merge(x, on='invoiceNumber')
df.sort_values(by=['invoiceNumber'])

df.head(200)

df[['invoiceNumber', 'custId']].describe()
####################################################################
#Extractcolumns "invoiceNumber" and  "productName" "transactionDateTime" and "Customer"
dfinvoice = df[["invoiceNumber", "transactionDateTime", "custName" ,"productName"]]
#Create an increment count column within one invoice
dfinvoice['itemNo'] = dfinvoice.groupby('invoiceNumber').cumcount() + 1

###################################################################
productDbX = productDb[["productId", "Name"]]
productDbX = productDbX.rename(columns={"Name": "productName"})
#join the productName with ProductId from ProductDb
dfinvoice1 = pd.merge(dfinvoice, productDbX, on='productName')
dfinvoice1.sort_values(by=['invoiceNumber'], inplace=True)

# dfinvoice1
####################################################################
#reshaping
data = dfinvoice1[["invoiceNumber","itemNo","productId"]]
data = data.pivot_table(index=["invoiceNumber"], columns='itemNo', values='productId')
data = data.fillna(0).astype(int)
# keep only three products for each transaction
data = data.iloc[:,0:4]
data = data.dropna().astype(int)

#join with cust x list
data = data.merge(x, on='invoiceNumber')
data = data[['invoiceNumber', 'custId', 1, 2, 3]]
data
data.columns
# Generate a cumulative count within each customer
data = data.sort_values(by=['custId', 'invoiceNumber'])
data['itemNo'] = data.groupby('custId').cumcount() + 1
data
# isThirdValue =  data[data['itemNo']==3]
# isThirdValue
# listOfDFRowsTrain = data.iloc[:5000,1:5].to_numpy().tolist()
# listOfDFRowsTest = isThirdValue.iloc[:,1].to_numpy().tolist()
# # print(listOfDFRowsTrain)
# ls = listOfDFRowsTest
# listOfDFRowsTest = []
# listOfDFRowsTest.append(ls)
# print(listOfDFRowsTest)
# from cpt.cpt import Cpt
# model = Cpt()

# model.fit(listOfDFRowsTrain)
# model.predict(listOfDFRowsTest)
#Create an increment count column within one invoice
data_chrono = data
data_chrono['BuyNo.'] = data_chrono.groupby('custId').cumcount() + 1
data_chrono.sort_values(by=['custId', 'invoiceNumber'])
data_chrono2 = data_chrono[['custId', 1, 'BuyNo.']]
data_chrono2.sort_values(by=['custId', 'BuyNo.'])
data_chrono2 = data_chrono2.rename(columns={1: 'productBought'})
data_chrono2 = data_chrono2.pivot(index='custId',columns='BuyNo.')[['productBought']]
data_chrono2

# fill NaN with -1
data_chrono2 = data_chrono2.fillna(-1).astype(int)
data_chrono2

data_chrono2.columns = pd.DataFrame(list(data_chrono2.columns.to_series()))[1]
# data_chrono2['custId'] = data_chrono2.index
data_chrono2.reset_index(level=0, inplace=True)
data_chrono2
chronoTrain = data_chrono2.iloc[:,0:80]
p = 'productBought'
random.seed(30)

randcol = [random.randint(0,80)
           ,random.randint(0,80)
           ,random.randint(0,80)
           ,random.randint(0,80)
           ,random.randint(0,80)
           ,random.randint(0,80)
           ,random.randint(0,80)
           ,random.randint(0,80)
          ]
# chronoTest = data_chrono2.iloc[:,0], data_chrono2[randcol]
# data_chrono2.columns
chronoTest = pd.concat([data_chrono2.iloc[:,0], data_chrono2[randcol]], axis=1)
print('PREV BUY TRAINING SET ', '\n', chronoTrain[0:100], '\n', '===============================', '\n', 'PREV TEST TRAINING SET ', '\n', chronoTest)
data[100:200].head(n=60)
train = data[(data['itemNo']!=3) & (data['itemNo']!=7)] 
assocBuytrainX = pd.DataFrame(train.iloc[:,1:5]) #crop the test dataset to only first column
# assocBuytestX = pd.DataFrame(test.iloc[:,1:3]) #crop the test dataset to only first column

testX = data[(data['itemNo']==3) ]  #crop the test dataset to only first column
assocBuytestX = pd.DataFrame(testX.iloc[:,1:3]) #get the 3rd row within every customer group


print('===========================')
print('ASSOC BUY TRAINING SET ' , '\n', assocBuytrainX.head(n=20))
print('.............')
print(assocBuytrainX[100:120])
print('ASSOC BUY TEST SET ' , '\n', assocBuytestX.head(n=20))
assocBuytestX.shape
# Export all train and test sets to folders
# train.to_csv('train.csv',index=False)
# test.to_csv('test.csv',index=False)

chronoTrain.to_csv('chronoTrain.csv',index=False)
chronoTest.to_csv('chronoTest.csv',index=False)

assocBuytrainX.to_csv('assocBuytrainXraw.csv',index=False)
assocBuytestX.to_csv('assocBuytestXraw.csv',index=False)

#run the Compacted Prediction Tree model
########## SAMPLE CODE ###########################
# model = CPTp()
# trainCPT, testCPT = model.load_df(assocBuytrainX,testX)
# model.train(trainCPT)
# predictions = model.predict(trainCPT,testCPT,3,1)
#########################################################
# chronoTrain , chronoTest 
prevBuymodel = CPTp()
prevBuytrainCPT, prevBuytestCPT = prevBuymodel.load_df(chronoTrain,chronoTest)
prevBuymodel.train(prevBuytrainCPT)
prevBuyPred = prevBuymodel.predict(prevBuytrainCPT,prevBuytestCPT,50,2)

#########################################################
# assocBuytrainX , assocBuytestX 
assocBuymodel = CPTp()
assocBuytrainCPT, assocBuytestCPT = assocBuymodel.load_df(assocBuytrainX,assocBuytestX)
assocBuymodel.train(assocBuytrainCPT)
assocBuyPred =assocBuymodel.predict(assocBuytrainCPT,assocBuytestCPT,3,2)

# predictions = model.predict(trainCPT,testCPT,3,2)


# View first 50 lines of predictions datasets
print('Prediction for prev-bought model', '\n') 
print(pd.DataFrame(prevBuyPred).loc[0:50,:])
print('=================================================')
print('Prediction for assoc-bought model', '\n') 
print(pd.DataFrame(assocBuyPred).loc[0:50,:])
# Clean prediction results and join with test set
prevBuyPreddf = pd.DataFrame(prevBuyPred, columns=['predictedToBuyProduct1','predictedToBuyProduct2' ])
prevBuyPreddf = prevBuyPreddf.fillna(0).astype(int)
prevBuyPreddf
prevBuyTestResult = pd.concat([chronoTest, prevBuyPreddf], axis=1)
# prevBuyTestResult
# Test Remap of value in pandas using dictionary
prevBuyTestResult = prevBuyTestResult.replace({"custId": custDict
                    ,69 :  productDbDict
                   ,37 :  productDbDict
                   ,78 :  productDbDict
                   ,3 :  productDbDict
                    ,79 :  productDbDict
                    ,26 :  productDbDict
                    ,32 :  productDbDict
                    ,6 :  productDbDict
                    ,'predictedToBuyProduct1' :  productDbDict
                    ,'predictedToBuyProduct2' :  productDbDict
                   })
chronoTest.to_csv('chronoTest.csv',index=False)
prevBuyTestResult.to_csv('chronoTestResult.csv',index=False)
prevBuyTestResult

chronoTrainRemap = chronoTrain
for i in range(1,len(chronoTrain.columns)+1):
    chronoTrainRemap = chronoTrainRemap.replace({i : productDbDict})
chronoTrainRemap = chronoTrainRemap.replace({"custId": custDict})

chronoTrainRemap.to_csv('chronoTrainRemap.csv',index=False)
chronoTrainRemap
# Clean prediction results and join with test set
assocBuytestX = assocBuytestX.rename(columns= {1: 'Selected product'})
assocBuytestX1 = assocBuytestX

assocBuyPreddf = pd.DataFrame(assocBuyPred, columns=['predictedToBuyProduct1','predictedToBuyProduct2' ])
assocBuyPreddf = assocBuyPreddf.fillna(-1).astype(int)
assocBuyPreddf

# Join with assoc Test result 
assocBuyTestResult = pd.concat([assocBuytestX1.reset_index(), assocBuyPreddf], axis=1)
assocBuyTestResult
productDbDict
# 583 Canadian Club Whisky 1L 12pk
custDict 
# 9018 Curtis Timmer
# Remap of value in pandas using dictionary
assocBuyTestResult = assocBuyTestResult.replace({"custId": custDict
                    ,"Selected product" :  productDbDict
                   ,"predictedToBuyProduct1" :  productDbDict
                   ,"predictedToBuyProduct2" :  productDbDict
                   })

# Rename columns and remap product name in trained assoc dataset
assocBuytrainXRemap= assocBuytrainX.rename(columns={1: 'Trained-Product1', 2: 'Trained-Product2', 3: 'Trained-Product3' })
assocBuytrainXRemap = assocBuytrainXRemap.replace({"custId": custDict
                    ,"Trained-Product1" :  productDbDict
                   ,"Trained-Product2" :  productDbDict
                   ,"Trained-Product3" :  productDbDict
                   })

# Export to csv
assocBuytestX1.to_csv('assocBuytestX1.csv',index=False)
assocBuyTestResult.to_csv('assocBuyTestResult.csv',index=False)
assocBuytrainXRemap.to_csv('assocBuytrainXRemap.csv',index=False)
assocBuyTestResult
print(data[(data['custId'] == 9018) & (data[1] == 583) ])
def decryptProductName(df, database, icol, colname):
    df['productId'] = df.iloc[:,icol-1]
#     df = pd.merge(df, database, on='productId')
    df = df.set_index('productId').join(database.set_index('productId'), on='productId')
    df = df.rename(columns={"productName" : colname})
#     df = df.drop(["productId"], axis = 1)
    return df
trainPName = train
trainPName.reset_index(inplace=True)
testResultPName = testResult.astype(int)
#TrainPName
for col in [2, 3, 4]:
    header = "productName" + str(col)
    trainPName =  decryptProductName(trainPName, productDbX,col, header)
#testResultPName
for col in [2,3,4,5,6]:
    header = "productName" + str(col)
    testResultPName =  decryptProductName(testResultPName,productDbX,col, header)
df.describe()