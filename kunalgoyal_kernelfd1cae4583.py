#importing various libraries needed for our model

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import re
ls
# read csv

data = pd.read_csv('../input/round_1_alloted.csv',header=None)
data.head()
data = data.set_index(0)
#print(data)
data.head()

for idx,row in data.iterrows():
    if(row[5] == 'UR PH1'):
        data[5][idx] = 'UR PH'
    if(row[5] == 'OBC PH1'):
        data[5][idx] = 'OBC PH'
    if(row[5] == 'SC PH1'):
        data[5][idx] = 'SC PH'
    if(row[5] == 'UR PH2'):
        data[5][idx] = 'UR PH'
    if(row[5] == 'OBC PH2'):
        data[5][idx] = 'OBC PH'
    if(row[5] == 'SC PH2'):
        data[5][idx] = 'SC PH'
    if(row[5] == 'ST PH1' or row[5] == 'ST PH2'):
        data[5][idx] = 'ST PH'
# drop 6th coloumn as it only has 'Allotted' value.

data = data.drop(6,axis=1)
# dictionary to map college to integer value

from collections import defaultdict
clg_to_int = defaultdict(str)
idx = 0
for i in data[2]:
    if(i not in clg_to_int):
        clg_to_int[i] = idx
        idx+=1
#clg_to_int
#int_to_clg
# there are 217 colleges from 0 to 216 :-)
# map int to college through previous clg_to_int mapping

int_to_clg = ['str']*218
for value in clg_to_int:
    #print(value)
    int_to_clg[clg_to_int[value]] = value
for i in range(len(int_to_clg)):
  int_to_clg[i]=re.sub('\r',' ',int_to_clg[i])
  int_to_clg[i]=re.sub('\n',' ',int_to_clg[i])
# add extra coloumn to convert college name to previously mapped value

data[7] = pd.Series(np.random.randint(len(data)), index=data.index)

for i in range(len(data)):
    #print(datar)
    data.iloc[i,5]=int(clg_to_int[data.iloc[i,1]])
data.head()
# now we can drop college name 

data = data.drop(2,axis = 1)
data.head()
# partition X and Y 

X = data[[1,3,4,5]]
Y = data[7]
Y = pd.DataFrame(Y)
X.head()
Y.head(10)
# creating category of X

category = [3,4,5]
for cat in category:
    dumm = pd.get_dummies(X[cat],prefix=cat)
    X = pd.concat([X,dumm],axis=1)
    X.drop(cat,axis=1,inplace = True)

# after category 
X.head()
# categorizing Y

#dumm = pd.get_dummies(Y,prefix=None)
Y.head(10)
# spliting X and Y in crossvalidation and traing set

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
print(X.shape)
print(Y.shape)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
X_test.head()
# decision tree model
dtree_model = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
#linear_model = LinearRegression().fit(X_train,y_train)
#svr_model = SVR().fit(X_train,y_train)
# predicting values for crossvalidation 'y_test' set

y_pred = dtree_model.predict(X_test)
y_pred = pd.DataFrame(y_pred, index = y_test.index, columns=[7])
#y_pred_linear = pd.DataFrame(linear_model.predict(X_test),index=y_test.index,columns=[7])
#y_pred_svr = pd.DataFrame(svr_model.predict(X_test),index=y_test.index,columns=[7])
#y_pred.rename(columns = )
#y_test.iloc[7][7]
summ = 0
for i in range (len(y_test)):
    if(abs(y_pred.iloc[i][7]-y_test.iloc[i][7])<=3):
        summ+=1
summ
# now we can make prediction according to rank, Alloted_category, course, candidate category.
print('What is your rank?')
rank = input()

print('Select the category which you want to be alloted(from options below)?')
print('1. UR \n2. UR PH\n3. OBC \n4. OBC PH\n5. SC\n6. SC PH\n7. ST\n8. ST PH')
alloted_cat = input()
print('Select the category which you belong to (from options below)?')
print('1. UR \n2. UR PH\n3. OBC \n4. OBC PH\n5. SC\n6. SC PH\n7. ST\n8. ST PH')
candidate_cat = input()
print('Select the course which you wana to take from options below)?')
print('1. MBBS \n2. BDS')
course = input()
clmns = ['3_BDS','3_MBBS','4_OBC','4_OBC PH','4_SC','4_SC PH','4_ST','4_ST PH','4_UR','4_UR PH','5_OBC','5_OBC PH','5_SC','5_SC PH','5_ST','5_ST PH','5_UR','5_UR PH']
test = pd.DataFrame({0:[1],1:[rank]})
for i in clmns:
  test[i] = pd.Series(np.random.randint(1))
test
test = test.set_index(0)
test['3_'+course] = 1
test['4_'+alloted_cat] = 1
test['5_'+candidate_cat] = 1
test
dtree_model = DecisionTreeRegressor(random_state=42).fit(X, Y)
ans = dtree_model.predict(test)

print('Model predicted colleges:\n\n')

if int(ans[0])<3:
  for i in range(6):
    print(int_to_clg[i])
elif int(ans[0])>213:
  for i in range(211,216):
    print(int_to_clg[i])
else:
  for i in range(int(ans[0]),int(ans[0])+3):
    print(int_to_clg[i])
  for i in range(int(ans[0])-3,int(ans[0])):
    print(int_to_clg[i])
  