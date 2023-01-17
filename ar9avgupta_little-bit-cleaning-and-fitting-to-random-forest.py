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
print(os.listdir('../input'))


nRowsRead = 100000 # specify 'None' if want to read whole file
# Dataset-Unicauca-Version2-87Atts.csv has 3577296 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/ip-network-traffic-flows-labeled-with-87-apps/Dataset-Unicauca-Version2-87Atts.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Dataset-Unicauca-Version2-87Atts.csv'
nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')


df1[:100]
df1.dropna(inplace = True) 
# new = df1["Flow.ID"].str.split("-", n = 1, expand = True)
# df1["Flow.IP"]= new[0] 
# df1["Flow.random"]= new[1] 
# df1.drop(columns =["Flow.ID"], inplace = True) 
def ipInfo(addr=''):
    from urllib.request import urlopen
    from json import load
    if addr == '':
        url = 'https://ipinfo.io/json'
    else:
        url = 'https://ipinfo.io/' + addr + '/json'
    res = urlopen(url)
    #response from url(if res==None then check connection)
    data = load(res)
    #will load the json response into data
    for attr in data.keys():
        #will print the data line by line
        print(attr,' '*13+'\t->\t',data[attr])
df1.drop(columns =["Timestamp"], inplace = True) 
df1.columns
df1.drop(columns =["Label"], inplace = True) 
df2 = df1
df2.drop(columns =["Source.IP", "Source.Port", "Destination.IP", "Destination.Port"], inplace = True) #better way is to replace it with country using some lib like geo2ip but I am lazy
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
df2["labels"] = lb_make.fit_transform(df2["ProtocolName"])
df2[["ProtocolName", "labels"]].head(11)
df2.head()
df2.drop(columns = ["Flow.ID"], inplace = True)
df2.drop(columns = ["ProtocolName"], inplace = True)
from sklearn.model_selection import train_test_split

y = df2.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=0.2)


# bestfeatures = SelectKBest(score_func=chi2, k=15)
# fit = bestfeatures.fit(X,y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# featureScores.columns = ['Specs','Score']
# print(featureScores.nlargest(15,'Score'))
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100) #try tuning
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
clf.score(X_test, y_test)
