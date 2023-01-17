# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
barcDataFrame = pd.read_csv("/kaggle/input/barc-2018-result/barc.csv")

barcDataFrame.head()
del barcDataFrame["S.No"]

del barcDataFrame["Reg.No"]

barcDataFrame.head()
barcDataFrame["Subject"] = barcDataFrame["Subject"].astype("category")

barcDataFrame["Subject"].value_counts()
barcDataFrame["From which exam"] = barcDataFrame["From which exam"].astype("category")

barcDataFrame["From which exam"].value_counts()
barcDataFrame["Qualified"] = barcDataFrame["Qualified"].astype("category")

barcDataFrame["Qualified"].value_counts()
barcDataFrame.info()
barcDataFrame[ barcDataFrame["From which exam"] == "Online Exam 2018 Score" ].head()
barcDataFrame[ barcDataFrame["From which exam"] == "GATE 2018 Score" ].head()
barcDataFrame[ barcDataFrame["From which exam"] == "GATE 2017 Score" ].head()
barcDataFrame = barcDataFrame[ barcDataFrame["From which exam"] != "GATE 2017 Score" ]

barcDataFrame["From which exam"].value_counts()
CodeToExamTypeMap = dict( zip( barcDataFrame["From which exam"].cat.codes, barcDataFrame["From which exam"] ) )

CodeToQualifiedMap = dict( zip( barcDataFrame["Qualified"].cat.codes, barcDataFrame["Qualified"] ) )

CodeToSubjectdMap = dict( zip( barcDataFrame["Subject"].cat.codes, barcDataFrame["Subject"] ) )



print( CodeToExamTypeMap )

print( CodeToQualifiedMap )

print( CodeToSubjectdMap )



QualifiedToCodeMap = { value:key for key,value in CodeToQualifiedMap.items() }

ExamTypeToCodeMap = { value:key for key,value in CodeToExamTypeMap.items() }

SubjectToCodeMap = { value:key for key,value in CodeToSubjectdMap.items() }
barcDataFrame["From which exam"] = barcDataFrame["From which exam"].cat.codes

barcDataFrame["Qualified"] = barcDataFrame["Qualified"].cat.codes

barcDataFrame["Subject"] = barcDataFrame["Subject"].cat.codes

barcDataFrame.head()
Y = barcDataFrame[ 'Qualified' ]

X = barcDataFrame.drop( 'Qualified', axis = 1 )
X['GATEScore'] = X['GATEScore'].replace({'-':0})

X['OnlineScore'] = X['OnlineScore'].replace({'-':0})
Y.head()
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
X_standardized = X



std_scaler = preprocessing.StandardScaler().fit( X[ ["GATEScore","OnlineScore"] ] ) 

X_standardized[ ["GATEScore","OnlineScore"] ] = std_scaler.transform( X_standardized[ ["GATEScore","OnlineScore"] ] )
X.head()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
from sklearn.svm import SVC

clf = SVC(random_state=29, tol=1e-5, max_iter = 2000, verbose = 2 )
clf.fit( X_train, Y_train)
print( "Training Score:", clf.score( X_train, Y_train  ) )

print( "Testing Score:", clf.score( X_test, Y_test  ) )