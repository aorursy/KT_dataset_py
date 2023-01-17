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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
x=np.linspace(0,10,100)

yorg=np.sin(x)

y=np.sin(x)+np.random.randn(100)*0.2
num=5  #移動平均の個数

b=np.ones(num) /num



y2=np.convolve(y, b, mode='same')  #移動平均



plt.plot(x,yorg,'r',label='オリジナルsin')

plt.plot(x,y,'k-',label='元系列')

plt.plot(x,y2,'b--', label='移動平均')

plt.legend()