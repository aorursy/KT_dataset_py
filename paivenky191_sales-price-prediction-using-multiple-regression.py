# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.vb
data=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

x=data.iloc[:,[46,62]]
y=data.iloc[:,80]
x_test=test.iloc[:,[46,62]]

xt=x.transpose()
m1=np.matmul(xt,x)
d1=np.linalg.inv(m1)

d2=np.matmul(xt,y)
b=np.matmul(d1,d2)
b0=float(b[0])
b1=float(b[1])
def predict(x,y):
    return ((x*b0)+(y*b1))
my_prediction=[]
test_ID=test['Id']
pred_area=test['GrLivArea']
pred_g_area=test['GarageArea']
for i in range(0,len(pred_area)):
    predicted_price=predict(pred_area[i],pred_g_area[i])
    my_prediction.append(predicted_price)
plt.plot(my_prediction)
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = my_prediction
print(sub)
sub.to_csv('submission.csv',index=False)