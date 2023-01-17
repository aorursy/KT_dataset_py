import pandas as pd 

import os

from datetime import datetime



print(os.listdir("../input"))

pd.options.mode.chained_assignment = None



df = pd.read_csv("../input/eurusd-mar2019/eurusd_mar2019.csv", sep='\t')

df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Tickvol', 'Vol', 'Spread']

df.drop(['Vol'], axis=1, inplace=True)



df['Datetime'] =  pd.to_datetime(df['Date'] + " " + df['Time'], format='%Y.%m.%d  %H:%M:%S')

df.set_index('Datetime', inplace=True)
price = df[['Close']]



#NORMALIZATION: y = (x - min) / (max - min)

price['max'] = price['Close'].rolling(10).max()

price['min'] = price['Close'].rolling(10).min()



price['+1'] = (price['Close'].shift(-1) - price['min']) / (price['max'] - price['min'])

price['0'] = (price['Close'] - price['min']) / (price['max'] - price['min'])

price['-1'] = (price['Close'].shift(1) - price['min']) / (price['max'] - price['min'])

price['-2'] = (price['Close'].shift(2) - price['min']) / (price['max'] - price['min'])

price['-3'] = (price['Close'].shift(3) - price['min']) / (price['max'] - price['min'])

price['-4'] = (price['Close'].shift(4) - price['min']) / (price['max'] - price['min'])

price['-5'] = (price['Close'].shift(5) - price['min']) / (price['max'] - price['min'])

price['-6'] = (price['Close'].shift(6) - price['min']) / (price['max'] - price['min'])

price['-7'] = (price['Close'].shift(7) - price['min']) / (price['max'] - price['min'])

price['-8'] = (price['Close'].shift(8) - price['min']) / (price['max'] - price['min'])

price['-9'] = (price['Close'].shift(9) - price['min']) / (price['max'] - price['min'])



price.dropna(inplace=True)

from sklearn.cluster import KMeans



ml = price[['-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '0', '+1']]



kmeans = KMeans(n_clusters=20)

kmeans.fit(ml)

classe = kmeans.predict(ml)



centers = kmeans.cluster_centers_

centers_df = pd.DataFrame(data=centers,     # values

index=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'],

columns=['-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '0', '+1'])

centers_df['Target'] = (centers_df['+1'] - centers_df['0']) > 0
df1 = pd.read_csv("../input/eurusd-apr/eurusd_apr2019.csv", sep='\t')

df1.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Tickvol', 'Vol', 'Spread']

df1.drop(['Vol'], axis=1, inplace=True)

df1['Datetime'] = pd.to_datetime(df1['Date'] + " " + df1['Time'], format='%Y.%m.%d  %H:%M:%S')

df1.set_index('Datetime', inplace=True)



test = df1[['Close']]

test['max'] = test['Close'].rolling(10).max()

test['min'] = test['Close'].rolling(10).min()



test['+1'] = (test['Close'].shift(-1) - test['min']) / (test['max'] - test['min'])

test['0'] = (test['Close'] - test['min']) / (test['max'] - test['min'])

test['-1'] = (test['Close'].shift(1) - test['min']) / (test['max'] - test['min'])

test['-2'] = (test['Close'].shift(2) - test['min']) / (test['max'] - test['min'])

test['-3'] = (test['Close'].shift(3) - test['min']) / (test['max'] - test['min'])

test['-4'] = (test['Close'].shift(4) - test['min']) / (test['max'] - test['min'])

test['-5'] = (test['Close'].shift(5) - test['min']) / (test['max'] - test['min'])

test['-6'] = (test['Close'].shift(6) - test['min']) / (test['max'] - test['min'])

test['-7'] = (test['Close'].shift(7) - test['min']) / (test['max'] - test['min'])

test['-8'] = (test['Close'].shift(8) - test['min']) / (test['max'] - test['min'])

test['-9'] = (test['Close'].shift(9) - test['min']) / (test['max'] - test['min'])

test.dropna(inplace=True)



test_ml = test[['-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '0', '+1']]
X = test[['-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '0']]

y = test['+1']-test['0']>0



count = 0

for index in range(len(X['0'])):

    x = X.iloc[index]

    y_iterated = y[index]

    check = []

    for i in range(20):

        msd = 0

        for j in range(10):

            msd += (centers_df.iloc[i,j]-x[j])**2

        check.append(msd/10)

    count+=int(centers_df.iloc[check.index(min(check))]['Target']==y_iterated)

print(count/len(X['0']))
y.value_counts()

653/(820+653)