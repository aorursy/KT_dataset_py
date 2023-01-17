# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('/kaggle/input/stocknews/Combined_News_DJIA.csv')
data.head()
print(data.shape)
print(data.dtypes)
data.describe()
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
train.head()
test.head()
trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
trainheadlines[:1]
basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
print(basictrain.shape)
basicmodel = LogisticRegression()
basicmodel = basicmodel.fit(basictrain, train['Label'])
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
basictest = basicvectorizer.transform(testheadlines)
predictions = basicmodel.predict(basictest)
pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
basicwords = basicvectorizer.get_feature_names()
basiccoeffs = basicmodel.coef_.tolist()[0]
coeffdf = pd.DataFrame({'Word' : basicwords, 
                        'Coefficient' : basiccoeffs})
coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
coeffdf.head(10)
coeffdf.tail(10)
shifted1 = data['Label'].shift(-1)
shifted1.head()
shifted2 = data['Label'].shift(-2)
shifted2.head()
shifted3 = data['Label'].shift(-3)
shifted3.head()
shifted7 = data['Label'].shift(-7)
shifted7.head()
data['Labelshifted1']=shifted1
data['Labelshifted2']=shifted2
data['Labelshifted3']=shifted3
data['Labelshifted7']=shifted7
data.head()
data['Labelshifted1'] = data['Labelshifted1'].fillna(0)
data['Labelshifted2'] = data['Labelshifted2'].fillna(0)
data['Labelshifted3'] = data['Labelshifted3'].fillna(0)
data['Labelshifted7'] = data['Labelshifted7'].fillna(0)
data.tail()

train1 = data[data['Date'] < '2015-01-01']
test1 = data[data['Date'] > '2014-12-31']
print(train1.shape)
print(test1.shape)
trainheadlines1 = []
for row in range(0,len(train1.index)):
    trainheadlines1.append(' '.join(str(x) for x in train1.iloc[row,2:27]))
trainheadlines1[:1]
basicvectorizer1 = CountVectorizer()
basictrain1 = basicvectorizer1.fit_transform(trainheadlines1)
print(basictrain1.shape)
basicmodel1 = LogisticRegression()
basicmodel1 = basicmodel1.fit(basictrain1, train1['Labelshifted1'])
testheadlines1 = []
for row in range(0,len(test1.index)):
    testheadlines1.append(' '.join(str(x) for x in test1.iloc[row,2:27]))
basictest1 = basicvectorizer1.transform(testheadlines1)
predictions1 = basicmodel1.predict(basictest1)
pd.crosstab(test1["Labelshifted1"], predictions1, rownames=["Actual"], colnames=["Predicted"])
basicwords1 = basicvectorizer1.get_feature_names()
basiccoeffs1 = basicmodel1.coef_.tolist()[0]
coeffdf1 = pd.DataFrame({'Word' : basicwords1, 
                        'Coefficient' : basiccoeffs1})
coeffdf1 = coeffdf1.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
coeffdf1.head(10)
coeffdf1.tail(10)
trainheadlines2 = []
for row in range(0,len(train1.index)):
    trainheadlines2.append(' '.join(str(x) for x in train1.iloc[row,2:27]))
trainheadlines2[:1]
basicvectorizer2 = CountVectorizer()
basictrain2 = basicvectorizer2.fit_transform(trainheadlines2)
print(basictrain2.shape)
basicmodel2 = LogisticRegression()
basicmodel2 = basicmodel2.fit(basictrain2, train1['Labelshifted2'])
testheadlines2 = []
for row in range(0,len(test1.index)):
    testheadlines2.append(' '.join(str(x) for x in test1.iloc[row,2:27]))
basictest2 = basicvectorizer2.transform(testheadlines2)
predictions2 = basicmodel2.predict(basictest2)
predictions2 = basicmodel2.predict(basictest2)
pd.crosstab(test1["Labelshifted2"], predictions2, rownames=["Actual"], colnames=["Predicted"])
basicwords2 = basicvectorizer2.get_feature_names()
basiccoeffs2 = basicmodel2.coef_.tolist()[0]
coeffdf2 = pd.DataFrame({'Word' : basicwords2, 
                        'Coefficient' : basiccoeffs2})
coeffdf2 = coeffdf2.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
coeffdf2.head(10)
coeffdf2.tail(10)
trainheadlines3 = []
for row in range(0,len(train1.index)):
    trainheadlines3.append(' '.join(str(x) for x in train1.iloc[row,2:27]))
trainheadlines3[:1]
basicvectorizer3 = CountVectorizer()
basictrain3 = basicvectorizer3.fit_transform(trainheadlines3)
print(basictrain3.shape)
basicmodel3 = LogisticRegression()
basicmodel3 = basicmodel3.fit(basictrain3, train1['Labelshifted3'])
testheadlines3 = []
for row in range(0,len(test1.index)):
    testheadlines3.append(' '.join(str(x) for x in test1.iloc[row,2:27]))
basictest3 = basicvectorizer3.transform(testheadlines3)
predictions3 = basicmodel3.predict(basictest3)
predictions3 = basicmodel3.predict(basictest3)
pd.crosstab(test1["Labelshifted3"], predictions3, rownames=["Actual"], colnames=["Predicted"])
basicwords3 = basicvectorizer3.get_feature_names()
basiccoeffs3 = basicmodel3.coef_.tolist()[0]
coeffdf3 = pd.DataFrame({'Word' : basicwords3, 
                        'Coefficient' : basiccoeffs3})
coeffdf3 = coeffdf3.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
coeffdf3.head(10)
coeffdf3.tail(10)
trainheadlines7 = []
for row in range(0,len(train1.index)):
    trainheadlines7.append(' '.join(str(x) for x in train1.iloc[row,2:27]))
trainheadlines7[:1]
basicvectorizer7 = CountVectorizer()
basictrain7 = basicvectorizer7.fit_transform(trainheadlines3)
print(basictrain7.shape)
basicmodel7 = LogisticRegression()
basicmodel7 = basicmodel7.fit(basictrain7, train1['Labelshifted7'])
testheadlines7 = []
for row in range(0,len(test1.index)):
    testheadlines7.append(' '.join(str(x) for x in test1.iloc[row,2:27]))
basictest7 = basicvectorizer7.transform(testheadlines7)
predictions7 = basicmodel7.predict(basictest7)
predictions7 = basicmodel7.predict(basictest7)
pd.crosstab(test1["Labelshifted7"], predictions7, rownames=["Actual"], colnames=["Predicted"])
basicwords7 = basicvectorizer7.get_feature_names()
basiccoeffs7 = basicmodel7.coef_.tolist()[0]
coeffdf7 = pd.DataFrame({'Word' : basicwords7, 
                        'Coefficient' : basiccoeffs7})
coeffdf7 = coeffdf7.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
coeffdf7.head(10)
coeffdf7.tail(10)