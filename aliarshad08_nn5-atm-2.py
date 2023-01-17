#Import data

import pandas as pd



df = pd.read_excel('../input/nn522/nn5-2.xls')

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()

df.dtypes
df["u1"] = df["Date"].dt.day

df["u2"] = df["Date"].dt.dayofweek

df["u3"] = df["Date"].dt.week

df["u4"] = df["Date"].dt.month

df["u5"] = 0

df["u5"].astype("int64")



df.head()



df_holiday = pd.read_excel('../input/ukholidays/ukH.xls',header=None)

df_holiday.head()
for i in range(0,len(df['Date'])):

    for s in df_holiday.iloc[:,1]:

        if df.iloc[i,0]==s:

            df.loc[i,"u5"]=1
df["u5"].head()
train = df.iloc[0:735,:]

test = df.iloc[735:,:]



train.tail()
test.head()
train[5:25]
import numpy as np

train.iloc[:,0:111].replace(0,np.nan,inplace=True)
train[5:10]
for i in range(1,111):

    lq = train.iloc[:,i].quantile(0.25)

    uq = train.iloc[:,i].quantile(0.75)

    iq=uq-lq

    train.loc[train.iloc[:,i]<lq-1.5*iq,train.columns[i]] = np.nan

    train.loc[train.iloc[:,i]>uq+1.5*iq,train.columns[i]] = np.nan
train.iloc[640:650,1]
train = train.interpolate(method='cubic')
train.iloc[640:650,1]
df.iloc[:735,1]
mapes = []

ias = []
def index_agreement(s,o):

    """

	index of agreement

	input:

        s: simulated

        o: observed

    output:

        ia: index of agreement

    """

#     s,o = filter_nan(s,o)

    ia = 1 -(np.sum((o-s)**2))/(np.sum(

    			(np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))

    return ia
def smape(A, F):

    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)) + np.finfo(float).eps)
for col in train.columns:

    if train[col].isna().sum()>0:

        print(col,train[col].isna().sum())
train.loc[733,"NN5-021"]=train.loc[732,"NN5-021"]

train.loc[734,"NN5-021"]=train.loc[732,"NN5-021"]

train["NN5-021"]
from sklearn import svm

from sklearn.model_selection import GridSearchCV



param_grid = [

    {'kernel': ['rbf', 'sigmoid'], 'gamma': [0, 10, 100], 'C' : [1,100, 1000]},

#     {'kernel': [3, 10], 'gamma': [2, 3, 4]},

]



forest_reg = svm.SVR()



grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)





d = {'1': train.iloc[:,i], '2': train.iloc[:,i], '3': train.iloc[:,i], '4': train.iloc[:,i],'5':train.loc[:,'u1'],'6':train.loc[:,'u2'],'7':train.loc[:,'u3'],'8':train.loc[:,'u4'], '9': train.loc[:,'u5'],'10': train.iloc[:,i]}

df1 = pd.DataFrame(data=d)



df1['2'] = df1['2'].shift(1)

df1['3'] = df1['3'].shift(2)

df1['4'] = df1['4'].shift(3)

df1['10'] = df1['10'].shift(-1)

df1.head()



df2=df1.iloc[3:-1,:]

df2.head()



df2.tail()



grid_search.fit(df2.iloc[:,0:-1], df2.iloc[:,-1])



grid_search.best_estimator_



grid_search.best_params_
from sklearn import svm



for i in range(1,111):

    d = {'1': train.iloc[:,i], '2': train.iloc[:,i], '3': train.iloc[:,i], '4': train.iloc[:,i],'5':train.loc[:,'u1'],'6':train.loc[:,'u2'],'7':train.loc[:,'u3'],'8':train.loc[:,'u4'], '9': train.loc[:,'u5'],'10': train.iloc[:,i]}

    df1 = pd.DataFrame(data=d)



    df1['2'] = df1['2'].shift(1)

    df1['3'] = df1['3'].shift(2)

    df1['4'] = df1['4'].shift(3)

    df1['10'] = df1['10'].shift(-1)

    df1.head()



    df2=df1.iloc[3:-1,:]

    df2.head()



    df2.tail()







    model = svm.SVR() 

    # there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score

    model.fit(df2.iloc[:,0:-1], df2.iloc[:,-1])

    # model.score(X, y)

    #Predict Output

    predicted= model.predict(df2.iloc[:, 0:-1])





    mapes.append(smape(df2.iloc[:, -1], predicted))

    ias.append(index_agreement(df2.iloc[:, -1], predicted))

    



print("SMAPE ","Mean: ",sum(mapes)/len(mapes)," Best: ",min(mapes)," Worst: ",max(mapes))

print("IA ","Mean: ",sum(ias)/len(ias)," Best: ",max(ias)," Worst: ",min(ias))