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
sns.distplot(df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]])
df['Species'].value_counts()
pisetosa=50/150
piversicolor=50/150
pivirginica=50/150
print(pisetosa)
print(piversicolor)
print(pivirginica)
data=df.groupby('Species')
mean=data.mean().reset_index()
mean
mslirissetosa=mean.iloc[0,2]
mswirissetosa=mean.iloc[0,3]
mplirissetosa=mean.iloc[0,4]
mpwirissetosa=mean.iloc[0,5]

mslirisversicolor=mean.iloc[1,2]
mswirisversicolor=mean.iloc[1,3]
mplirisversicolor=mean.iloc[1,4]
mpwirisversicolor=mean.iloc[1,5]

mslirisvirginica=mean.iloc[2,2]
mswirisvirginica=mean.iloc[2,3]
mplirisvirginica=mean.iloc[2,4]
mpwirisvirginica=mean.iloc[2,5]

print(mslirissetosa)
print(mswirissetosa)
print(mplirissetosa)
print(mpwirissetosa)

print(mslirisversicolor)
print(mswirisversicolor)
print(mplirisversicolor)
print(mpwirisversicolor)

print(mslirisvirginica)
print(mswirisvirginica)
print(mplirisvirginica)
print(mpwirisvirginica)
sd=data.std().reset_index()
sd
sdslirissetosa=sd.iloc[0,2]
sdswirissetosa=sd.iloc[0,3]
sdplirissetosa=sd.iloc[0,4]
sdpwirissetosa=sd.iloc[0,5]

sdslirisversicolor=sd.iloc[1,2]
sdswirisversicolor=sd.iloc[1,3]
sdplirisversicolor=sd.iloc[1,4]
sdpwirisversicolor=sd.iloc[1,5]

sdslirisvirginica=sd.iloc[2,2]
sdswirisvirginica=sd.iloc[2,3]
sdplirisvirginica=sd.iloc[2,4]
sdpwirisvirginica=sd.iloc[2,5]

print(sdslirissetosa)
print(sdswirissetosa)
print(sdplirissetosa)
print(sdpwirissetosa)

print(sdslirisversicolor)
print(sdswirisversicolor)
print(sdplirisversicolor)
print(sdpwirisversicolor)

print(sdslirisvirginica)
print(sdswirisvirginica)
print(sdplirisvirginica)
print(sdpwirisvirginica)
#When x i.e SepalLength = 4.7

pslis= 1/(sdslirissetosa*np.sqrt(2*np.pi))*(np.exp(-np.abs((4.7-mslirissetosa)**2)/2*sdslirissetosa*sdslirissetosa)) #pslis is P(SL=4.7/Iris-Setosa)

#When x i.e SepalWidth = 3.7

pswis= 1/(sdswirissetosa*np.sqrt(2*np.pi))*(np.exp(-np.abs((3.7-mswirissetosa)**2)/2*sdswirissetosa*sdswirissetosa)) #pswis is P(SW=3.7/Iris-Setosa)

#When x i.e PetalLength = 2

pplis= 1/(sdplirissetosa*np.sqrt(2*np.pi))*(np.exp(-np.abs((2-mplirissetosa)**2)/2*sdplirissetosa*sdplirissetosa)) #pplis is P(PL=2/Iris-Setosa)

#When x i.e PetalWidth = 0.3

ppwis= 1/(sdpwirissetosa*np.sqrt(2*np.pi))*(np.exp(-np.abs((0.3-mpwirissetosa)**2)/2*sdpwirissetosa*sdpwirissetosa)) #ppwis is P(PW=0.3/Iris-Setosa)
print(pslis)
print(pswis)
print(pplis)
print(ppwis)
#When x i.e SepalLength = 4.7

psliversicolor= 1/(sdslirisversicolor*np.sqrt(2*np.pi))*(np.exp(-np.abs((4.7-mslirisversicolor)**2)/2*sdslirisversicolor*sdslirisversicolor)) #psliversicolor is P(SL=4.7/Iris-Setosa)

#When x i.e SepalWidth = 3.7

pswiversicolor= 1/(sdswirisversicolor*np.sqrt(2*np.pi))*(np.exp(-np.abs((3.7-mswirisversicolor)**2)/2*sdswirisversicolor*sdswirisversicolor)) #pswiversicolor is P(SW=3.7/Iris-Setosa)

#When x i.e PetalLength = 2

ppliversicolor= 1/(sdplirisversicolor*np.sqrt(2*np.pi))*(np.exp(-np.abs((2-mplirisversicolor)**2)/2*sdplirisversicolor*sdplirisversicolor)) #ppliversicolor is P(PL=2/Iris-Setosa)

#When x i.e PetalWidth = 0.3

ppwiversicolor= 1/(sdpwirisversicolor*np.sqrt(2*np.pi))*(np.exp(-np.abs((0.3-mpwirisversicolor)**2)/2*sdpwirisversicolor*sdpwirisversicolor)) #ppwiversicolor is P(PW=0.3/Iris-Setosa)
print(psliversicolor)
print(pswiversicolor)
print(ppliversicolor)
print(ppwiversicolor)
#When x i.e SepalLength = 4.7

pslivirginica= 1/(sdslirisvirginica*np.sqrt(2*np.pi))*(np.exp(-np.abs((4.7-mslirisvirginica)**2)/2*sdslirisvirginica*sdslirisvirginica)) #pslivirginica is P(SL=4.7/Iris-virginica)

#When x i.e SepalWidth = 3.7

pswivirginica= 1/(sdswirisvirginica*np.sqrt(2*np.pi))*(np.exp(-np.abs((3.7-mswirisvirginica)**2)/2*sdswirisvirginica*sdswirisvirginica)) #pswivirginica is P(SW=3.7/Iris-virginica)

#When x i.e PetalLength = 2

pplivirginica= 1/(sdplirisvirginica*np.sqrt(2*np.pi))*(np.exp(-np.abs((2-mplirisvirginica)**2)/2*sdplirisvirginica*sdplirisvirginica)) #pplivirginica is P(PL=2/Iris-virginica)

#When x i.e PetalWidth = 0.3

ppwivirginica= 1/(sdpwirisvirginica*np.sqrt(2*np.pi))*(np.exp(-np.abs((0.3-mpwirisvirginica)**2)/2*sdpwirisvirginica*sdpwirisvirginica)) #ppwivirginica is P(PW=0.3/Iris-virginica)
print(pslivirginica)
print(pswivirginica)
print(pplivirginica)
print(ppwivirginica)
PIrisSetosa= pisetosa*pslis*pswis*pplis*ppwis
PIrisSetosa
PIrisVersicolor= piversicolor*psliversicolor*pswiversicolor*ppliversicolor*ppwiversicolor
PIrisVersicolor
PIrisVirginica= pivirginica*pslivirginica*pswivirginica*pplivirginica*ppwivirginica
PIrisVirginica
