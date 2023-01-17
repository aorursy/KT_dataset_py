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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv("../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv")

print(train)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv("../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv")

train = train.iloc[::-1]

x = train['Open']

y = train['Close']



x_data = x.as_matrix()

xdata_n = npdata = np.zeros((x_data.size, 1))



b = np.reshape(x_data, (x_data.size, 1))

#print(b)

#xdata_n[0] = x_data

y_data = y.as_matrix()



from sklearn.neighbors import KNeighborsClassifier

model = LinearRegression()

model.fit(b, y_data)



train = pd.read_csv("../input/Uniqlo(FastRetailing) 2017 Test - stocks2017.csv")



#model_ols = pd.ols(y=y, x=x, intercept = True)

#print(model_ols)