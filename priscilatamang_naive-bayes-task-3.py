# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import seaborn as sns
df=pd.read_csv('/kaggle/input/iris/Iris.csv')
df

#P(Iris-setosa)
#P(Iris-versicolor)
#P(Iris-virginica)
df['Species'].value_counts()
p_setosa=50/150
p_versicolor=50/150
p_virginica=50/150
print("P(isetosa) =", p_setosa)
print("P(iversicolor) =", p_versicolor)
print("P(iverginica) =", p_virginica)
data=df.groupby('Species')
mean=data.mean().reset_index()
mean
m_sl_setosa = mean.iloc[0,2]
m_sw_setosa = mean.iloc[0,3]
m_pl_setosa = mean.iloc[0,4]
m_pw_setosa = mean.iloc[0,5]

m_sl_versicolor = mean.iloc[1,2]
m_sw_versicolor = mean.iloc[1,3]
m_pl_versicolor = mean.iloc[1,4]
m_pw_versicolor = mean.iloc[1,5]

m_sl_virginica=mean.iloc[2,2]
m_sw_virginica=mean.iloc[2,3]
m_pl_virginica=mean.iloc[2,4]
m_pw_virginica=mean.iloc[2,5]

print(m_sl_setosa)
print(m_sw_setosa)
print(m_pl_setosa)
print(m_pw_setosa)

print(m_sl_versicolor)
print(m_sw_versicolor)
print(m_pl_versicolor)
print(m_pw_versicolor)

print(m_sl_virginica)
print(m_sw_virginica)
print(m_pl_virginica)
print(m_pw_virginica)
sd=data.std().reset_index()
sd
sd_sl_setosa=sd.iloc[0,2]
sd_sw_setosa=sd.iloc[0,3]
sd_pl_setosa=sd.iloc[0,4]
sd_pw_setosa=sd.iloc[0,5]

sd_sl_versicolor=sd.iloc[1,2]
sd_sw_versicolor=sd.iloc[1,3]
sd_pl_versicolor=sd.iloc[1,4]
sd_pw_versicolor=sd.iloc[1,5]

sd_sl_virginica=sd.iloc[2,2]
sd_sw_virginica=sd.iloc[2,3]
sd_pl_virginica=sd.iloc[2,4]
sd_pw_virginica=sd.iloc[2,5]

print(sd_sl_setosa)
print(sd_sw_setosa)
print(sd_pl_setosa)
print(sd_pw_setosa)

print(sd_sl_versicolor)
print(sd_sw_versicolor)
print(sd_pl_versicolor)
print(sd_pw_versicolor)

print(sd_sl_virginica)
print(sd_sw_virginica)
print(sd_pl_virginica)
print(sd_pw_virginica)
#When x i.e SepalLength = 4.7

p_sl_is= (1/(sd_sl_setosa*np.sqrt(2*np.pi)))*np.exp((((4.7-m_sl_setosa)/sd_sl_setosa)**2)/-2) #pslis is P(SL=4.7/Iris-Setosa)

#When x i.e SepalWidth = 3.7

p_sw_is= (1/(sd_sw_setosa*np.sqrt(2*np.pi)))*np.exp((((3.7-m_sw_setosa)/sd_sw_setosa)**2)/-2) #pswis is P(SW=3.7/Iris-Setosa)

#When x i.e PetalLength = 2

p_pl_is= (1/(sd_pl_setosa*np.sqrt(2*np.pi)))*np.exp((((2-m_pl_setosa)/sd_pl_setosa)**2)/(-2)) #pplis is P(PL=2/Iris-Setosa)

#When x i.e PetalWidth = 0.3

p_pw_is= (1/(sd_pw_setosa*np.sqrt(2*np.pi)))*np.exp((((0.3-m_pw_setosa)/sd_pw_setosa)**2)/(-2)) #ppwis is P(PW=0.3/Iris-Setosa)
print(p_sl_is)
print(p_sw_is)
print(p_pl_is)
print(p_pw_is)
#When x i.e SepalLength = 4.7

p_sl_iversicolor= (1/(sd_sl_versicolor*np.sqrt(2*np.pi)))*np.exp((((4.7-m_sl_versicolor)/sd_sl_versicolor)**2)/(-2)) #psliversicolor is P(SL=4.7/Iris-Setosa)

#When x i.e SepalWidth = 3.7

p_sw_iversicolor= (1/(sd_sw_versicolor*np.sqrt(2*np.pi)))*np.exp((((3.7-m_sw_versicolor)/sd_sw_versicolor)**2)/(-2)) #pswiversicolor is P(SW=3.7/Iris-Setosa)

#When x i.e PetalLength = 2

p_pl_iversicolor= (1/(sd_pl_versicolor*np.sqrt(2*np.pi)))*np.exp((((2-m_pl_versicolor)/sd_pl_versicolor)**2)/(-2)) #ppliversicolor is P(PL=2/Iris-Setosa)

#When x i.e PetalWidth = 0.3

p_pw_iversicolor= (1/(sd_pw_versicolor*np.sqrt(2*np.pi)))*np.exp((((0.3-m_pw_versicolor)/sd_pw_versicolor)**2)/(-2)) #ppwiversicolor is P(PW=0.3/Iris-Setosa)
print(p_sl_iversicolor)
print(p_sw_iversicolor)
print(p_pl_iversicolor)
print(p_pw_iversicolor)
#When x i.e SepalLength = 4.7

p_sl_ivirginica= (1/(sd_sl_virginica*np.sqrt(2*np.pi)))*np.exp((((4.7-m_sl_virginica)/sd_sl_virginica)**2)/(-2)) #pslivirginica is P(SL=4.7/Iris-virginica)

#When x i.e SepalWidth = 3.7

p_sw_ivirginica= (1/(sd_sw_virginica*np.sqrt(2*np.pi)))*np.exp((((3.7-m_sw_virginica)/sd_sw_virginica)**2)/(-2)) #pswivirginica is P(SW=3.7/Iris-virginica)

#When x i.e PetalLength = 2

p_pl_ivirginica= (1/(sd_pl_virginica*np.sqrt(2*np.pi)))*np.exp((((2-m_pl_virginica)/sd_pl_virginica)**2)/(-2)) #pplivirginica is P(PL=2/Iris-virginica)

#When x i.e PetalWidth = 0.3

p_pw_ivirginica= (1/(sd_pw_virginica*np.sqrt(2*np.pi)))*np.exp((((0.3-m_pw_virginica)/sd_pw_virginica)**2)/(-2)) #ppwivirginica is P(PW=0.3/Iris-virginica)
print(p_sl_ivirginica)
print(p_sw_ivirginica)
print(p_pl_ivirginica)
print(p_pw_ivirginica)
PIrisSetosa= p_setosa*p_sl_is*p_sw_is*p_pl_is*p_pw_is
PIrisSetosa
PIrisVersicolor= p_versicolor*p_sl_iversicolor*p_sw_iversicolor*p_pl_iversicolor*p_pw_iversicolor
PIrisVersicolor
PIrisVirginica= p_virginica*p_sl_ivirginica*p_sw_ivirginica*p_pl_ivirginica*p_pw_ivirginica
PIrisVirginica