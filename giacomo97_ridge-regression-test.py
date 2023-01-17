# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pl



pl.style.use('seaborn-whitegrid')

pl.rcParams['figure.figsize'] = [14,10]





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
trainingData=pd.read_csv("/kaggle/input/sofia-air-quality-dataset/2017-12_bme280sof.csv")

trainingData.head()
from sklearn.linear_model import Ridge

trainingData = trainingData[trainingData.sensor_id == 6698]

trainingData.dropna(subset=["pressure","humidity"])

trainingData.sort_values(by=['timestamp'])

trainingData=trainingData.iloc[int(len(trainingData)*0.33):int(len(trainingData)*0.66)]





x=np.array(trainingData[["pressure"]])

y=np.array(trainingData[["humidity"]])
pl.plot(x,y,".",label="Training Data")

for alpha in [0,1,100,1000,10000,100000]:

    ridgeModel=Ridge(alpha=alpha)

    ridgeModel.fit(x,y)

    prediction=ridgeModel.predict(x)

    pl.plot(x,prediction,label="Alpha="+str(alpha))

pl.legend()

pl.show()
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline



for alpha in [0,1,10,100,1000,10000,100000]:

    for degree in [1,2,3,4,5]:

        pl.figure()

        pl.title("alpha="+str(alpha)+" degree="+str(degree),fontsize=30,fontweight="bold")

        pl.plot(x,y,".",label="Training Data")

        polynomialRidgeModel = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))

        polynomialRidgeModel.fit(x, y)

        results = polynomialRidgeModel.predict(x)

        pl.plot(x, results, label="alpha="+str(alpha)+" degree="+str(degree))

        pl.legend()

        pl.show()
