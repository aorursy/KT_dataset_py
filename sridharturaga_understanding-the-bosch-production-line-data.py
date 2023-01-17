

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Generic function to inspect the data

def inspectDataFiles(anyDF):

    anyDF.head()

    anyDFHeader = anyDF.head(0) #Just grabbing the headers



    l0Count = len(anyDFHeader.T.filter(like='L0_',axis = 0)) #axis = 0 by row

    l1Count = len(anyDFHeader.T.filter(like='L1_',axis = 0)) 

    l2Count = len(anyDFHeader.T.filter(like='L2_',axis = 0))

    l3Count = len(anyDFHeader.T.filter(like='L3_',axis = 0))



    print(l0Count, l1Count, l2Count, l3Count)

    return
inspectDataFiles#Inspecting the Categorical data



trainCategoricalDF = pd.read_csv("../input/train_categorical.csv", nrows = 10)

inspectDataFiles(trainCategoricalDF)

trainCategoricalDF.head()
#Inspecting the Numeric data



trainNumericDF = pd.read_csv("../input/train_numeric.csv", nrows = 10)

inspectDataFiles(trainNumericDF)
#Inspecting the Date data



trainDateDF = pd.read_csv("../input/train_date.csv", nrows = 10)

inspectDataFiles(trainDateDF)
trainCategoricalDF = pd.read_csv("../input/train_categorical.csv", nrows = 10)

inspectDataFiles(trainCategoricalDF)
trainNumericDF = pd.read_csv("../input/train_numeric.csv", nrows = 10)

inspectDataFiles(trainNumericDF)
trainDateDF = pd.read_csv("../input/train_date.csv", nrows = 10)

inspectDataFiles(trainDateDF)
#Inspecting the Numeric data values - TRAIN



trainNumericDF = pd.read_csv("../input/train_numeric.csv", nrows = 1000)

trainNumericDF[['Response','Id']].groupby(['Response']).agg(['count'])
testNumericDF = pd.read_csv("../input/test_numeric.csv", nrows = 1000)

testNumericDF.head().filter(like = 'Response', axis = 0)

#testNumericDF[['Response']]
#Inspecting the Numeric data values - TEST



testNumericDF = pd.read_csv("../input/test_numeric.csv", nrows = 1000)

testNumericDF[['Response','Id']].groupby(['Response']).agg(['count'])