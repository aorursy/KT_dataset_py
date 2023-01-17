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
#chess analysis object oriented 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import statistics as st
import pickle
from sklearn.preprocessing import RobustScaler

RS = RobustScaler()



data1 = pd.read_pickle('chess_data.pkl')

#removing errors 
data1.loc[13683, 'result'] = 0.5
data1.loc[15846, 'result'] = 0
data1.loc[33094,'result'] = 1
white= []
for i in data1.white:
    white.append(float(i))
black = []
for i in data1.black:
    black.append(float(i))

data1.white = white 
data1.black = black    
meanElo = []
for i, j in zip(data1.white, data1.black):
    meanElo.append((i+j)/2)
eloDiff = []
for i, j in zip(data1.white, data1.black):
    eloDiff.append(i-j)
data1 = pd.concat([data1,pd.Series(meanElo), pd.Series(eloDiff)], axis = 1)
data1.rename(columns = {0: 'meanElo', 1: 'diffElo'}, inplace = True)
lenmoves = []
for i in data1.moves:
    lenmoves.append(len(i) -1)
lenmoves = pd.Series(lenmoves)
data1 = pd.concat([data1,lenmoves], axis = 1)
data1.rename(columns = {0:'lengths'}, inplace = True)


clippedElo = []
for i in data1.evals:
    clippedElo.append((pd.Series(i)).clip(-400,400))

data1.evals = clippedElo



mean_data = pd.DataFrame(data1.lengths)


class Features:
    def __init__(self, name, data):
        self.name = name
        self.data =  data
    def addtodata(self):
            matrix.data = pd.concat([matrix.data, pd.Series(self.data)], axis = 1)
            matrix.data.rename(columns = {0:self.name}, inplace = True)
        
    def removefromdata(self):
        matrix.drop(self.name, axis = 1, inplace = True)
    

class EvalFeatures(Features):
    def __init__(self, name, data):
        super().__init__(name, data)
        self.std = []
        for i in self.data:
            if i != []:
                self.std.append(np.std(i))
            else:
                self.std.append(None)
        self.std = pd.Series(self.std) 
        self.std.fillna(np.median(self.std.dropna()), inplace = True)
        self.median = []
        for i in self.data:
            if i != []:
                self.median.append(np.median(i))
            else:
                self.median.append(None)
        self.median = pd.Series(self.median) 
        self.median.fillna(np.median(self.median.dropna()), inplace = True)
        self.maximum = []
        for i in self.data:
             if i != []:
                self.maximum.append(np.max(i))
             else:
                self.maximum.append(None)
        self.maximum = pd.Series(self.maximum) 
        self.maximum.fillna(np.median(self.maximum.dropna()), inplace = True)
        self.minimum = []
        for i in self.data:
             if i != []:
                self.minimum.append(np.min(i))
             else:
                self.minimum.append(None)
        self.minimum = pd.Series(self.minimum) 
        self.minimum.fillna(np.median(self.minimum.dropna()), inplace = True)    
    def addtodata(self):
            matrix.data = pd.concat([matrix.data, pd.Series(self.std),pd.Series(self.median),pd.Series(self.maximum),pd.Series(self.minimum)], axis = 1)
            matrix.data.rename(columns = {0 : self.name + '_std',1 : self.name +'_med', 2 : self.name + '_max', 3 : self.name +'_min'}, inplace = True)
    def addtodataMean(self):
            means.data = pd.concat([means.data, pd.Series(self.std),pd.Series(self.median),pd.Series(self.maximum),pd.Series(self.minimum)], axis = 1)
            means.data.rename(columns = {0 : self.name + '_std',1 : self.name +'_med', 2 : self.name + '_max', 3 : self.name +'_min'}, inplace = True)
    
    @staticmethod
    def partitiondiffs(pieces, eval_cutoff = 10000, first_move = 0, last_move = 1000):
        minuses = []
        if pieces == 'white':
            for i in matrix.data.evals:
                avg = []
                if len(i) > first_move:
                    for j in range(first_move, (min(len(i), last_move))):
                        if (j%2 == 0) & (j !=0):
                            if (-(eval_cutoff) < i[j] < eval_cutoff):
                                avg.append(i[j] - i[j-1])
            
                minuses.append(avg)
            minuses = pd.Series(minuses)                
            return minuses
            #return minuses.fillna(np.median(minuses).dropna()) 
        
        elif pieces == 'black':
            for i in matrix.data.evals:
                avg = []
                if len(i) > first_move:
                    for j in range(first_move, (min(len(i), last_move))):
                        if (j%2 != 0):
                            if (-(eval_cutoff) < i[j] < eval_cutoff):
                                avg.append(-1*(i[j] - i[j-1]))
                minuses.append(avg)
            minuses = pd.Series(minuses)                
            return minuses
        
class MoveFeatures(Features):
    def __init__(self, name, data):
        super().__init__(name, data)
        self.queenw, self.queenb = MoveFeatures.firstmove('Q')
        self.kingw, self.kingb = MoveFeatures.firstmove('K')
        self.checkw, self.checkb = MoveFeatures.firstmove('+')  
    def addtodata1(self):
            matrix.data = pd.concat([matrix.data, pd.Series(self.queenw),pd.Series(self.queenb),pd.Series(self.kingw),pd.Series(self.kingb), pd.Series(self.checkw),pd.Series(self.checkb)], axis = 1)
            matrix.data.rename(columns = {0 : 'queenw',1 : 'queenb', 
                                          2 : 'kingw', 3 : 'kingb',4 : 'checkw', 
                                          5 : 'checkb'}, inplace = True)
    @staticmethod
    def firstmove(piece):
        white = []
        black = []
        for i in matrix.data.moves:
            for j in range(len(i)):
                if j == (len(i) -1):
                    white.append(j)
                if (piece in i[j]) & (j%2 == 0):
                    white.append(j)
                    break
        
    
        for i in matrix.data.moves:
            for j in range(len(i)):
                if j == (len(i) -1):
                    black.append(j)
                if (piece in i[j]) & (j%2 != 0):
                    black.append(j)
                    break  
        return white,black
    @staticmethod
    def goodmovecounts(colour, threshold, result, move):
        count = []
        if colour == 'white':
            for i in matrix.data.evals:
                moves = 0
                if len(i) >0: 
                    for j in range(len(i)):                   
                        if (j%2 == 0) & (j !=0):
                            if move == 'good':
                                if i[j] - i[j-1] > threshold:
                                    moves+=1
                            elif move == 'bad':
                                if i[j] - i[j-1] < threshold:
                                    moves+=1
                    if result == 'count':
                        count.append(moves)
                    else:
                        count.append(moves/(len(i)/2))
                else: 
                    count.append(None)                
        if colour == 'black':
            for i in matrix.data.evals:
                moves = 0
                if len(i) >0: 
                    for j in range(len(i)):
                        if (j%2 != 0):
                            if move == 'good':
                                if i[j] - i[j-1] < threshold:
                                    moves+=1
                            elif move == 'bad':
                                if i[j] - i[j-1] > threshold:
                                    moves+=1
                    if result == 'count':
                        count.append(moves)
                    else:
                        count.append(moves/(len(i)/2))
                else: 
                    count.append(None)
        return pd.Series(count)
    
    
matrix = Features('matrix', data1)



diffs = []
for i in matrix.data.evals:
    each_diff = []
    for j in range(len(i)):
        if j !=0:
            each_diff.append(i[j] - i[j-1])
    diffs.append(each_diff)

absdiffs = []
for i in diffs:
    eachabsdiff = []
    for j in range(len(i)):
        eachabsdiff.append(abs(i[j]))
    absdiffs.append(eachabsdiff)



diffsw = []
for i in matrix.data.evals:
    each_diff = []
    for j in range(len(i)):
        if j !=0 and j%2 == 0:
            each_diff.append(i[j] - i[j-1])
    diffsw.append(each_diff)

diffsb = []
for i in matrix.data.evals:
    each_diff = []
    for j in range(len(i)):
        if j%2 !=0:
            each_diff.append(i[j] - i[j-1])
    diffsb.append(each_diff)

means = Features('means', mean_data)
deltas = EvalFeatures('delta', diffs)
absdeltas = EvalFeatures('absdeltas', absdiffs)
deltas.addtodata()
absdeltas.addtodataMean()

deltasw= EvalFeatures('deltasw', diffsw)
deltasw.addtodata()
deltasb = EvalFeatures('deltasb', diffsb)
deltasb.addtodata()


deltaw1 = EvalFeatures('deltaw1',EvalFeatures.partitiondiffs('white', first_move = 0, last_move = 20))
deltaw1.addtodata()
deltaw2 = EvalFeatures('deltaw2',EvalFeatures.partitiondiffs('white', first_move = 21, last_move = 40))
deltaw2.addtodata()
deltaw3 = EvalFeatures('deltaw3',EvalFeatures.partitiondiffs('white', first_move = 41, last_move = 60))
deltaw3.addtodata()
deltaw4 = EvalFeatures('deltaw4',EvalFeatures.partitiondiffs('white', first_move = 61, last_move = 90))
deltaw4.addtodata()
deltaw5 = EvalFeatures('deltaw5',EvalFeatures.partitiondiffs('white', first_move = 91))
deltaw5.addtodata()

deltab1 = EvalFeatures('deltab1',EvalFeatures.partitiondiffs('black', first_move = 0, last_move = 20))
deltab1.addtodata()
deltab2 = EvalFeatures('deltab2',EvalFeatures.partitiondiffs('black', first_move = 21, last_move = 40))
deltab2.addtodata()
deltab3 = EvalFeatures('deltab3',EvalFeatures.partitiondiffs('black', first_move = 41, last_move = 60))
deltab3.addtodata()
deltab4 = EvalFeatures('deltab4',EvalFeatures.partitiondiffs('black', first_move = 61, last_move = 90))
deltab4.addtodata()
deltab5 = EvalFeatures('deltab5',EvalFeatures.partitiondiffs('black', first_move = 91))
deltab5.addtodata()

moves = MoveFeatures('moves', matrix.data.moves)
moves.addtodata1()

goodmovesw = Features('goodmovesw',MoveFeatures.goodmovecounts('white', -10, 'share','good'))
goodmovesb = Features('goodmovesb',MoveFeatures.goodmovecounts('white', 10, 'share','good'))
goodmovescw = Features('goodmovescw',MoveFeatures.goodmovecounts('white', -10, 'count','good'))
goodmovescb = Features('goodmovescb',MoveFeatures.goodmovecounts('white', 10, 'count','good'))

goodmovesw.addtodata()
goodmovesb.addtodata()
goodmovescw.addtodata()
goodmovescb.addtodata()




blundersw = Features('blundersw',MoveFeatures.goodmovecounts('white', -100, 'share','bad'))
blundersb = Features('blundersb',MoveFeatures.goodmovecounts('white', 100, 'share','bad'))
blunderscw = Features('blunderscw',MoveFeatures.goodmovecounts('white', -100, 'count','bad'))
blunderscb = Features('blunderscb',MoveFeatures.goodmovecounts('white', 100, 'count','bad'))

blundersw.addtodata()
blundersb.addtodata()
blunderscw.addtodata()
blunderscb.addtodata()




from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
ohe = OneHotEncoder(sparse = False)

labelencodedv = le.fit_transform(data1.variation)
onehotencodedv = ohe.fit_transform(labelencodedv.reshape(-1,1))

labelencodedo = le.fit_transform(data1.opening)
onehotencodedo = ohe.fit_transform(labelencodedo.reshape(-1,1))

openings = pd.concat([pd.DataFrame(onehotencodedo), pd.DataFrame(onehotencodedv)], axis = 1)

'''x1_mean = pd.concat([means.data, openings], axis = 1)
x1_mean.columns = list(range(1123))
y1_mean = data1.meanElo

x1_diff = matrix.data.drop(['white', 'black', 'meanElo', 'diffElo','result', 'evals', 'opening', 'variation', 'moves','lengths'], axis = 1) 
y1_diff = data1.diffElo

x_trainm = x1_mean.loc[:20000, :]
x_testm = x1_mean.loc[20000:,:]
y_trainm = y1_mean.loc[:20000]
y_testm = y1_mean.loc[20000:]


x_traind = x1_diff.loc[:20000, :]
x_testd = x1_diff.loc[20000:,:]
y_traind = y1_diff.loc[:20000]
y_testd = y1_diff.loc[20000:]




from xgboost import XGBRegressor

xgb = XGBRegressor(max_depth = 4, n_estimators = 250, learning_rate = 0.05, lamda= .1, alpha = .1 )

xgb.fit(x_trainm,y_trainm)
xgbpredsm = xgb.predict(x_testm)

np.sqrt(mse(xgbpreds,y_testm))

xgb.fit(x_traind, y_traind)
xgbpredsd = xgb.predict(x_testd)
finalpredsw = []
finalpredsb = []
for i,j in zip(xgbpredsm, xgbpredsd):
    finalpredsw.append(i + (j/2))
    finalpredsb.append(i - (j/2))

np.sqrt(mse(finalpredsw,y_test.white))'''


train = matrix.data.loc[:24999,:]
train = pd.concat([train, openings.loc[:24999]], axis = 1)


#test data 
X2  = matrix.data.drop(['white', 'black', 'meanElo', 'diffElo','result', 'evals', 'opening', 'variation', 'moves'], axis = 1).loc[25000:,:]
X2 = pd.concat([X2, openings.loc[25000:]], axis = 1)

#outliers 
train.drop(labels = [14854,23781],inplace = True)
train.reset_index(inplace = True, drop = True)




from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
mse  = mean_squared_error




X1= train.drop(labels = train.lengths.loc[(train.lengths<20)].index)
X1= X1.drop(labels = train.lengths.loc[(train.lengths>300)].index)
X1.reset_index(inplace = True, drop = True)




y = X1[['white','black']]
X = X1.drop(['white', 'black', 'meanElo', 'diffElo','result', 'evals', 'opening', 'variation', 'moves'], axis = 1)

X.columns = list(range(1306))



x_train = X.loc[ :20000,:]
x_test = X.loc[20000:,:]
y_train = y.loc[:20000,:]
y_test = y.loc[20000:,:]





x_train = pd.DataFrame(RS.fit_transform(x_train))
x_test = pd.DataFrame(RS.fit_transform(x_test))



from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostRegressor
xgb = XGBRegressor(max_depth = 4, n_estimators = 250, learning_rate = 0.05)
#en = ElasticNet(alpha = 0,l1_ratio = .1, max_iter = 500)
#ada = AdaBoostRegressor(n_estimators = 25, learning_rate = .05)

#en.fit(x_train,y_train.white)
#enpreds = en.predict(x_test)

#ada.fit(x_train,y_train.white)
#adapreds = ada.predict(x_test)

xgb.fit(x_train,y_train.white)
xgbpreds = xgb.predict(x_test)

print('xgb error is ' + str(np.sqrt(mse(xgbpreds,y_test.white))))
#print('en error is ' + str(np.sqrt(mse(enpreds,y_test.white))))
#print('ada error is ' + str(np.sqrt(mse(adapreds,y_test.white))))

misses =[]
for i,j in zip(xgbpreds,y_test.white):
    misses.append(i-j)

'''missesada =[]
for i,j in zip(adapreds,y_test.white):
    missesada.append(i-j)'''



#final preds
xgbw = XGBRegressor(max_depth = 4, n_estimators = 250, learning_rate = 0.05)
xgbb = XGBRegressor(max_depth = 4, n_estimators = 250, learning_rate = 0.05)
X2.columns = list(range(1306))


X = pd.DataFrame(RS.fit_transform(X))
X2 = pd.DataFrame(RS.fit_transform(X2))




xgbw.fit(X,y.white)
xgbb.fit(X,y.black)
white_preds = xgbw.predict(X2)
black_preds = xgbb.predict(X2)

events = list(range(25001,50001))
intev= []
for i in events:
    intev.append(int(i))

preds = pd.concat([pd.Series(events), pd.Series(white_preds), pd.Series(black_preds)], axis = 1 )
preds.columns = ['Event', 'WhiteElo','BlackElo']
preds.Event = intev

preds.to_csv('first_preds.csv', index = False)

