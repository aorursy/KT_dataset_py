# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Dataseti okuduk ve head komutu ile ilk 5'ini ekrana bastırdık.
# We read the dataset and printed the first 5 with the head command.
dataset=pd.read_csv("../input/column_2C_weka.csv")
dataset.head()
# Regression için yalnızca Anormal olanları yeni bir data değişkenine eşitliyoruz.
# For regression, we equate the abnormal ones to a new data list variable.
data=dataset[dataset['class']=='Abnormal']
data.head()
# X a feature and Y a label. 
x=np.array(data['pelvic_incidence']).reshape(-1,1)
y=np.array(data['sacral_slope']).reshape(-1,1)
# Scatter komutu ile noktaları x ve y üzerinde belirliyoruz.
# With the Scatter command we set the points on the x and y plane.
import matplotlib.pyplot as plt
plt.scatter(x,y,color="red")
plt.xlabel('Pelvic incidence')
plt.ylabel('Sacral Slope')
plt.title('Data')
plt.show()
from sklearn.linear_model import LinearRegression
linearReg=LinearRegression()
linearReg.fit(x,y)
predictvalue=np.linspace(min(x),max(x)).reshape(-1,1)
y_head=linearReg.predict(predictvalue)
print(' X, Y score :',linearReg.score(x,y))
plt.plot(predictvalue,y_head,color="blue")
plt.scatter(x,y,color="red")
plt.xlabel('Pelvic incidence')
plt.ylabel('Sacral Slope')
plt.title('Data')
plt.show()