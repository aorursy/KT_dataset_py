# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#import required packages

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression
remove=['Alley','LandContour', 'Utilities', 'LandSlope', 'Condition2', 'RoofMatl', 'Electrical','Functional']
#read input files

train=pd.read_csv("../input/train.csv",na_values="NA")

test=pd.read_csv("../input/test.csv", na_values="NA")
#separate the output column from the rest of your data

prices=train['SalePrice']

train.drop('SalePrice',axis=1,inplace=True)

#concat data to get all columns

all_data=pd.concat([train,test])

#remove some columns

for col in remove:

    all_data.drop(col,axis=1,inplace=True)

#convert categorical columns into one hot encoding

all_data=pd.get_dummies(all_data)

X=all_data.as_matrix()

#handle NA values

X=np.nan_to_num(X)
#split data into training,development and test set

X_train=X[:int(train.shape[0]*0.8)]

prices_train= prices[:int(train.shape[0]*0.8)]

X_dev= X[int(train.shape[0]*0.8):train.shape[0]]

prices_dev= prices[int(train.shape[0]*0.8):]

X_test=X[train.shape[0]:]

prices_train.shape
#create model and train

lr=LinearRegression()

lr.fit(X_train,prices_train)
#evaluate on development set

Y=lr.predict(X_dev)

sq_diff = np.square(np.log(prices_dev)-np.log(Y))

error= np.sqrt(np.sum(sq_diff)/prices_dev.shape[0])

error
#prepare output for submission

Y=lr.predict(X_test)

out=pd.DataFrame()

out['Id']=[i for i in range(X_train.shape[0]+1,X_train.shape[0]+X_test.shape[0]+1)]

out['SalePrice']=Y

out.to_csv('output.csv',index=False)
