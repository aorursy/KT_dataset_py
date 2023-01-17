# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math as m

import matplotlib as ml

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

ml.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/iris/Iris.csv')

print(data.shape)

data.head()
data.drop(columns=['Id'],inplace=True)



# Renaming columns for better usability

new_column_name = {'SepalLengthCm':'SL','SepalWidthCm':'SW','PetalLengthCm':'PL','PetalWidthCm':'PW'}

data = data.rename(columns = new_column_name )

data
data.info()
sns.distplot(data.SL)

plt.show()

sns.distplot(data.SW)

plt.show()

sns.distplot(data.PL)

plt.show()

sns.distplot(data.PW)

plt.show()
plt.figure(figsize=(20,10))

sns.boxplot(x='Species',y='SL',data=data)

plt.show()
ind = data[(data.Species == 'Iris-virginica') & (data.SL < 5.0)].index

data.iloc[ind,0] = np.median(data[data.Species=='Iris-virginica'].SL)



plt.figure(figsize=(20,10))

sns.boxplot(x='Species',y='SL',data=data)

plt.show()
plt.figure(figsize=(20,10))

sns.boxplot(x='Species',y='SW',data=data)

plt.show()
indsw = [data[(data.Species == 'Iris-virginica') & (data.SW < 2.5)].index,data[(data.Species == 'Iris-virginica') & (data.SW > 3.6)].index]

for i in indsw:

    data.iloc[i,1] = np.median(data[data.Species=='Iris-virginica'].SW)



plt.figure(figsize=(20,10))

sns.boxplot(x='Species',y='SL',data=data)

plt.show()
plt.figure(figsize=(20,10))

sns.boxplot(x='Species',y='PL',data=data)

plt.show()
# setosa

indpl_set = [data[(data.Species == 'Iris-setosa') & (data.PL < 1.3)].index,data[(data.Species == 'Iris-setosa') & (data.PL > 1.6)].index]

# versicolor

indpl_vers = data[(data.Species == 'Iris-versicolor') & (data.PL < 3.1)].index



for i1 in indpl_set:

    data.iloc[i1,2] = np.mean(data[data.Species=='Iris-setosa'].PL)

data.iloc[indpl_vers,2] = np.mean(data[data.Species=='Iris-versicolor'].PL)



plt.figure(figsize=(20,10))

sns.boxplot(x='Species',y='PL',data=data)

plt.show()
plt.figure(figsize=(20,10))

sns.boxplot(x='Species',y='PW',data=data)

plt.show()
# setosa

indpw_set = list(data[(data.Species == 'Iris-setosa') & (data.PW >= 0.5)].index)



for i2 in indpw_set:

    data.iloc[i2,3] = np.median(data[data.Species=='Iris-setosa'].PW)



plt.figure(figsize=(20,10))

sns.boxplot(x='Species',y='PW',data=data)

plt.show()
sns.distplot(data.SL)

plt.show()

sns.distplot(data.SW)

plt.show()

sns.distplot(data.PL)

plt.show()

sns.distplot(data.PW)

plt.show()
def norm(mn,sd,x):

    p = (-((x-mn)/sd)**2)/2

    res = (1/(sd*(np.sqrt(2*np.pi))))*(m.exp(p))

    return res
# For setosa

setosa = data[data.Species=='Iris-setosa']

ps = setosa.shape[0]/data.shape[0]



sns.distplot(setosa.SL)

plt.show()

sns.distplot(setosa.SW)

plt.show()

sns.distplot(setosa.PL)

plt.show()

sns.distplot(setosa.PW)

plt.show()
# Conditional probabilities

p_sl = norm(np.mean(setosa.SL),np.std(setosa.SL),4.7)

p_sw = norm(np.mean(setosa.SW),np.std(setosa.SW),3.7)

p_pl = norm(np.mean(setosa.PL),np.std(setosa.PL),2)

p_pw = norm(np.mean(setosa.PW),np.std(setosa.PW),0.3)



ps_final = p_sl*p_sw*p_pl*p_pw*ps

print(ps_final)
# For versicolor

versi = data[data.Species=='Iris-versicolor']

pvs = versi.shape[0]/data.shape[0]



sns.distplot(versi.SL)

plt.show()

sns.distplot(versi.SW)

plt.show()

sns.distplot(versi.PL)

plt.show()

sns.distplot(versi.PW)

plt.show()
# Conditional probabilities

pv_sl = norm(np.mean(versi.SL),np.std(versi.SL),4.7)

pv_sw = norm(np.mean(versi.SW),np.std(versi.SW),3.7)

pv_pl = norm(np.mean(versi.PL),np.std(versi.PL),2)

pv_pw = norm(np.mean(versi.PW),np.std(versi.PW),0.3)



pvs_final = pv_sl*pv_sw*pv_pl*pv_pw*pvs

print(pvs_final)
# For virginica

vi = data[data.Species=='Iris-virginica']

pvi = vi.shape[0]/data.shape[0]



sns.distplot(vi.SL)

plt.show()

sns.distplot(vi.SW)

plt.show()

sns.distplot(vi.PL)

plt.show()

sns.distplot(vi.PW)

plt.show()
# Conditional probabilities

pvi_sl = norm(np.mean(vi.SL),np.std(vi.SL),4.7)

pvi_sw = norm(np.mean(vi.SW),np.std(vi.SW),3.7)

pvi_pl = norm(np.mean(vi.PL),np.std(vi.PL),2)

pvi_pw = norm(np.mean(vi.PW),np.std(vi.PW),0.3)



pvi_final = pvi_sl*pvi_sw*pvi_pl*pvi_pw*pvi

print(pvi_final)
final,labels = [ps_final,pvs_final,pvi_final],['Iris-setosa','Iris-versicolor','Iris-virginica']

print('Predicted species is : {}'.format(labels[final.index(max(final))]))
data.head(50)     # First 50 are setosa