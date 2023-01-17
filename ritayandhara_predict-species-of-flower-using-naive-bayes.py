# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
iris=pd.read_csv('/kaggle/input/iris/Iris.csv')
iris
plt.style.use('fivethirtyeight')

for i in iris['Species'].unique():

    plt.subplots(figsize=(15,10))

    plt.suptitle(i)

    for count,ele in enumerate(iris.columns[1:-1],1): 

        plt.subplot(2, 2, count)

        sns.distplot(iris[iris['Species']==i][ele])
def pred_species(sl,sw,pl,pw):

    # Probability of being Iris-setosa

    pSe=iris[iris['Species']=='Iris-setosa'].shape[0]/iris.shape[0]

    # Probability of being Iris-versicolor

    pVe=iris[iris['Species']=='Iris-versicolor'].shape[0]/iris.shape[0]

    # Probability of being Iris-virginica

    pVi=iris[iris['Species']=='Iris-virginica'].shape[0]/iris.shape[0]

    

    # Iris-setosa

    

    # Standard Deviation of sepal_length

    sd=np.std(iris[iris['Species']=='Iris-setosa']['SepalLengthCm'])

    # Mean of sepal_length

    mean=np.mean(iris[iris['Species']=='Iris-setosa']['SepalLengthCm'])

    # Probability of having the Sepal_length, given the flower is Iris-setosa 

    pSL_Se=(1/(sd*math.sqrt(2*math.pi)))*math.e**(-0.5*math.pow((sl-mean)/sd,2))

    

    # Standard Deviation of sepal_width

    sd=np.std(iris[iris['Species']=='Iris-setosa']['SepalWidthCm'])

    # Mean of sepal_width

    mean=np.mean(iris[iris['Species']=='Iris-setosa']['SepalWidthCm'])

    # Probability of having the sepal_width, given the flower is Iris-setosa 

    pSW_Se=(1/(sd*math.sqrt(2*math.pi)))*math.e**(-0.5*math.pow((sw-mean)/sd,2))

    

    # Standard Deviation of petal_length

    sd=np.std(iris[iris['Species']=='Iris-setosa']['PetalLengthCm'])

    # Mean of petal_length

    mean=np.mean(iris[iris['Species']=='Iris-setosa']['PetalLengthCm'])

    # Probability of having the petal_length, given the flower is Iris-setosa 

    pPL_Se=(1/(sd*math.sqrt(2*math.pi)))*math.e**(-0.5*math.pow((pl-mean)/sd,2))

    

    # Standard Deviation of petal_width

    sd=np.std(iris[iris['Species']=='Iris-setosa']['PetalWidthCm'])

    # Mean of petal_width

    mean=np.mean(iris[iris['Species']=='Iris-setosa']['PetalWidthCm'])

    # Probability of having the petal_width, given the flower is Iris-setosa 

    pPW_Se=(1/(sd*math.sqrt(2*math.pi)))*math.e**(-0.5*math.pow((pw-mean)/sd,2))

    

    # Probabilty of the flower being Iris-setosa given the inputs

    setosa=pSL_Se*pSW_Se*pPL_Se*pPW_Se*pSe

    

    

    

    # Iris-versicolor

    

    # Standard Deviation of sepal_length

    sd=np.std(iris[iris['Species']=='Iris-versicolor']['SepalLengthCm'])

    # Mean of sepal_length

    mean=np.mean(iris[iris['Species']=='Iris-versicolor']['SepalLengthCm'])

    # Probability of having the Sepal_length, given the flower is Iris-setosa 

    pSL_Ve=(1/(sd*math.sqrt(2*math.pi)))*math.e**(-0.5*math.pow((sl-mean)/sd,2))

    

    # Standard Deviation of sepal_width

    sd=np.std(iris[iris['Species']=='Iris-versicolor']['SepalWidthCm'])

    # Mean of sepal_width

    mean=np.mean(iris[iris['Species']=='Iris-versicolor']['SepalWidthCm'])

    # Probability of having the sepal_width, given the flower is Iris-setosa 

    pSW_Ve=(1/(sd*math.sqrt(2*math.pi)))*math.e**(-0.5*math.pow((sw-mean)/sd,2))

    

    # Standard Deviation of petal_length

    sd=np.std(iris[iris['Species']=='Iris-versicolor']['PetalLengthCm'])

    # Mean of petal_length

    mean=np.mean(iris[iris['Species']=='Iris-versicolor']['PetalLengthCm'])

    # Probability of having the petal_length, given the flower is Iris-setosa 

    pPL_Ve=(1/(sd*math.sqrt(2*math.pi)))*math.e**(-0.5*math.pow((pl-mean)/sd,2))

    

    # Standard Deviation of petal_width

    sd=np.std(iris[iris['Species']=='Iris-versicolor']['PetalWidthCm'])

    # Mean of petal_width

    mean=np.mean(iris[iris['Species']=='Iris-versicolor']['PetalWidthCm'])

    # Probability of having the petal_width, given the flower is Iris-setosa 

    pPW_Ve=(1/(sd*math.sqrt(2*math.pi)))*math.e**(-0.5*math.pow((pw-mean)/sd,2))

    

    # Probabilty of the flower being Iris-versicolor given the inputs

    versicolor=pSL_Ve*pSW_Ve*pPL_Ve*pPW_Ve*pVe

    

    

    

    # Iris-virginica

    

    # Standard Deviation of sepal_length

    sd=np.std(iris[iris['Species']=='Iris-virginica']['SepalLengthCm'])

    # Mean of sepal_length

    mean=np.mean(iris[iris['Species']=='Iris-virginica']['SepalLengthCm'])

    # Probability of having the Sepal_length, given the flower is Iris-setosa

    pSL_Vi=(1/(sd*math.sqrt(2*math.pi)))*math.e**(-0.5*math.pow((sl-mean)/sd,2))

    

    # Standard Deviation of sepal_width

    sd=np.std(iris[iris['Species']=='Iris-virginica']['SepalWidthCm'])

    # Mean of sepal_width

    mean=np.mean(iris[iris['Species']=='Iris-virginica']['SepalWidthCm'])

    # Probability of having the sepal_width, given the flower is Iris-setosa

    pSW_Vi=(1/(sd*math.sqrt(2*math.pi)))*math.e**(-0.5*math.pow((sw-mean)/sd,2))

    

    # Standard Deviation of petal_length

    sd=np.std(iris[iris['Species']=='Iris-virginica']['PetalLengthCm'])

    # Mean of petal_length

    mean=np.mean(iris[iris['Species']=='Iris-virginica']['PetalLengthCm'])

    # Probability of having the petal_length, given the flower is Iris-setosa

    pPL_Vi=(1/(sd*math.sqrt(2*math.pi)))*math.e**(-0.5*math.pow((pl-mean)/sd,2))

    

    # Standard Deviation of petal_width

    sd=np.std(iris[iris['Species']=='Iris-virginica']['PetalWidthCm'])

    # Mean of petal_width

    mean=np.mean(iris[iris['Species']=='Iris-virginica']['PetalWidthCm'])

    # Probability of having the petal_width, given the flower is Iris-setosa

    pPW_Vi=(1/(sd*math.sqrt(2*math.pi)))*math.e**(-0.5*math.pow((pw-mean)/sd,2))

    

    # Probabilty of the flower being Iris-virginica given the inputs

    virginica=pSL_Vi*pSW_Vi*pPL_Vi*pPW_Vi*pVi

    

    

    if setosa>versicolor and setosa>virginica:

        return ("Iris-setosa")

    if versicolor>setosa and versicolor>virginica:

        return ("Iris-versicolor")

    if virginica>versicolor and virginica>setosa:

        return ("Iris-virginica")
pred_species(4.7,3.7,2,0.3)
pred_species(5.7,2.5,5.0,2.0)