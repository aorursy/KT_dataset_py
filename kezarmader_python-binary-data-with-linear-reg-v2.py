# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plot graph



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn import linear_model;

from sklearn import preprocessing;

from sklearn import utils;

from patsy import dmatrices;

from sklearn.preprocessing import MinMaxScaler;
def convertBin(data, sIndex):

    data.insert(8, 'X7', '');

    data.insert(10, 'X9', '');

    data.insert(73, 'X72', 0);

    data.insert(121, 'X121', 0);

    data.insert(150, 'X149', 0);

    data.insert(189, 'X188', 0);

    data.insert(194, 'X193', 0);

    data.insert(304, 'X303', 0);

    data.insert(382, 'X381', 0);



    X0_9 = data.iloc[:,sIndex+2:sIndex+12].apply(lambda x: ''.join(x).ljust(3), axis=1);

    le = preprocessing.LabelEncoder()

    le.fit(X0_9)

    X0_9t = le.transform(X0_9)

    

    #data.iloc[:,sIndex+2]=le.fit_transform(data.iloc[:,sIndex+2]);

    #data.iloc[:,sIndex+3]=le.fit_transform(data.iloc[:,sIndex+3]);

    #data.iloc[:,sIndex+4]=le.fit_transform(data.iloc[:,sIndex+4]);

    #data.iloc[:,sIndex+5]=le.fit_transform(data.iloc[:,sIndex+5]);

    #data.iloc[:,sIndex+6]=le.fit_transform(data.iloc[:,sIndex+6]);

    #data.iloc[:,sIndex+7]=le.fit_transform(data.iloc[:,sIndex+7]);

    #data.iloc[:,sIndex+8]=le.fit_transform(data.iloc[:,sIndex+8]);

    #data.iloc[:,sIndex+9]=le.fit_transform(data.iloc[:,sIndex+9]);

    #data.iloc[:,sIndex+10]=le.fit_transform(data.iloc[:,sIndex+10]);

    #data.iloc[:,sIndex+11]=le.fit_transform(data.iloc[:,sIndex+11]);

    #data.iloc[:,sIndex+12]=le.fit_transform(data.iloc[:,sIndex+12]);

    

    

    bits = data.iloc[:,sIndex+12:sIndex+389].copy();

    bits.insert(0, 'X0_9t', X0_9t);

    

    data_x = bits;

    

    

    return data_x;
def convert(data, sIndex):

    data.insert(8, 'X7', '');

    data.insert(10, 'X9', '');

    data.insert(73, 'X72', 0);

    data.insert(121, 'X121', 0);

    data.insert(150, 'X149', 0);

    data.insert(189, 'X188', 0);

    data.insert(194, 'X193', 0);

    data.insert(304, 'X303', 0);

    data.insert(382, 'X381', 0);



    X0_9 = data.iloc[:,sIndex+2:sIndex+12].apply(lambda x: ''.join(x).ljust(3), axis=1);

    le = preprocessing.LabelEncoder()

    le.fit(X0_9)

    X0_9t = le.transform(X0_9)

        

    X10_385 = data.iloc[:,sIndex+12:sIndex+389].astype(str).apply(lambda x: ''.join(x), axis=1);

    a = X10_385.astype(str).apply(lambda x: int(x[:63],2));

    b = X10_385.astype(str).apply(lambda x: int(x[63:126],2));

    c = X10_385.astype(str).apply(lambda x: int(x[126:189],2));

    d = X10_385.astype(str).apply(lambda x: int(x[189:252],2));

    e = X10_385.astype(str).apply(lambda x: int(x[252:315],2));

    f = X10_385.astype(str).apply(lambda x: int(x[315:378],2));

    #g = X10_385.astype(str).apply(lambda x: int(x[378:441],2));



    data_x = pd.DataFrame(data = list(zip(X0_9t, a, b, c, d, e, f)), 

                          columns=['X0_9', 'a', 'b','c','d','e','f'])

        

    return data_x;
data_train = pd.read_csv('../input/train.csv');

data_test = pd.read_csv('../input/test.csv');



train_x = convertBin(data_train, 0);



test_x = convertBin(data_test, -1);

train_y = data_train.y;
train_x
linreg = linear_model.LinearRegression()

linreg.fit(train_x, train_y)

print(linreg.score(train_x, train_y))



pred_y = linreg.predict(test_x)
linreg.predict(test_x)[0:20]
finalResult = pd.DataFrame(data = list(zip(data_test.ID, pred_y)), columns=['ID', 'y'])

finalResult.to_csv('submisstion.csv')

finalResult[0:20]
plt.plot(test_x, pred_y)

plt.show()



plt.plot(train_x, train_y)

plt.show()
data_train = pd.read_csv('../input/train.csv');

data_test = pd.read_csv('../input/test.csv');



train_x = convertBin(data_train, 0);



train_x
data_train = pd.read_csv('../input/train.csv');

data_test = pd.read_csv('../input/test.csv');



train_x = convert(data_train, 0);



test_x = convert(data_test, -1);

train_y = data_train.y;



linreg = linear_model.LinearRegression()

linreg.fit(train_x, train_y)

print(linreg.score(train_x, train_y))



pred_y = linreg.predict(test_x)



linreg.predict(test_x)[0:20]



plt.plot(test_x, pred_y)

plt.show()



plt.plot(train_x, train_y)

plt.show()