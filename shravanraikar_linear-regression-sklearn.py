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
from sklearn.linear_model import LinearRegression

import pandas as pd

import matplotlib.pyplot as plt
#load Data



train_data=pd.read_csv('../input/random-linear-regression/train.csv')

test_data=pd.read_csv('../input/random-linear-regression/test.csv')



train_data.info
test_data.info
#drop missing varables 



train_data.dropna(inplace=True)
#set up training and testing data



x_train=train_data.x.values.reshape(-1,1)

y_train=train_data.y.values.reshape(-1,1)

x_test=test_data.x.values.reshape(-1,1)

y_test=test_data.y.values.reshape(-1,1)



reg=LinearRegression()



#predict ytrain values

reg.fit(x_train,y_train)

y_train_pred=reg.predict(x_train)
#plot Graph

plt.scatter(x_train,y_train)

plt.plot(x_train,y_train_pred,color="red",linewidth=3)

plt.title("train_data_prediction")

plt.show()
#predict y_test values

reg.fit(x_test,y_test)

y_test_pred=reg.predict(x_test)





plt.scatter(x_test,y_test)

plt.plot(x_test,y_test_pred,color="red",linewidth=3)

plt.title("test_data_prediction")

plt.show()
#check r sqauare value



r2_score_train=reg.score(x_train,y_train)

r2_score_test=reg.score(x_test,y_test)



print("R square value of trian data: ",r2_score_train," R square value of test data: ",r2_score_test)