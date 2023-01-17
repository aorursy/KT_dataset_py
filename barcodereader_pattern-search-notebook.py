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
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
#                                     Import Packages
#=================================================================================
# import packages for the linear regression
import operator
import math
#=================================================================================
#                                     Read Data
#=================================================================================
train = pd.read_csv('../input/models/train.csv', index_col=0)
test = pd.read_csv('../input/models/test.csv', index_col=0)
#=================================================================================
#                                     Process Data
#=================================================================================
#STEP 1
#transform each data into a set and put them into a list
nametag = ('lights','T1','T2','T3','T4','T5','T6','T7','T8','T9','T_out','RH_1','RH_2','RH_3','RH_4','RH_5','RH_6','RH_7','RH_8','RH_9','RH_out','Press_mm_hg','Windspeed','Visibility','Tdewpoint')
mastertrainlist = list()
length = len(train['lights'])
taglength = len(nametag)
for i in nametag:
    mastertrainlist.append(set(train[i]))
#STEP2 
#find record in above data set and collect line numbers
mastertraindic = list()
for i in range(0,taglength):
    dic = dict()
    for j in range(0,length):
        if train[nametag[i]][j] in mastertrainlist[i]:
            dic.setdefault(train[nametag[i]][j], []).append(j) #write all line numbers to a dictionary
    mastertraindic.append(dic)
#=================================================================================
#                                     Compare Data
#=================================================================================
#STEP 3
#put all test data into a list
#histogram function
def count_elements(seq) -> dict:
     hist = {}
     for i in seq:
         hist[i] = hist.get(i, 0) + 1
     return hist

mastertestlist = list()
for i in nametag:
    mastertestlist.append(list(test[i]))
testlength = len(test['lights'])
#STEP4
#search line number using test data
result = list()
mastertestdic = list()
for j in range(0,testlength): #for each row
    linelist = list()
    for k in range(0,taglength): #for each column
        if test[nametag[k]][j] in mastertrainlist[k]: #a record is found in the training data
            key = test[nametag[k]][j]         
            linelist += mastertraindic[k][key]
        else:
            linelist += [0]
    pathhist = count_elements(linelist)  #this will give the longest path
    linenumber = max(pathhist.items(), key=operator.itemgetter(1))[0] #this is the line with the highest similarity
    result.append(linenumber)
#=================================================================================
#                                     Fine-tune
#=================================================================================
APPLIANCES = train['Appliances']
lenapp = len(APPLIANCES)
lenresult = len(result)
PREDICT = list()
SUM = 0
for i in range(0,lenapp):
    SUM += APPLIANCES[i]
mean = math.floor(SUM/lenapp)

for i in range(0,lenresult):
    if result[i] == 0:
        PREDICT.append(mean)
    else:
        PREDICT.append(APPLIANCES[result[i]])
#=================================================================================
#                                     Show the Data
#=================================================================================  
truevalue = test['Appliances']
valuelen = len(truevalue)
print(valuelen)
final = dict()
plot1 = dict()
plot2 = dict()
SUM = 0
for i in range(0,valuelen):
      final[truevalue[i]] = PREDICT[i]
      plot1[i] = PREDICT[i]
      plot2[i] = truevalue[i]
      SUM = SUM+pow((truevalue[i] - PREDICT[i]),2)
RMSE = math.sqrt(SUM/valuelen)
print('===============================')
print('The predicted results are: {True value : Predicted value}')
print(final)
print('===============================')
print('The RMSE of this prediction is:')
print(RMSE)
plt.figure(figsize=(20, 8), dpi=80)
plt.scatter(range(valuelen), PREDICT)
plt.scatter(range(valuelen), truevalue) 