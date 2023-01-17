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
import pandas as pd
import numpy as np
import seaborn as sns

df=pd.read_csv('/kaggle/input/iris/Iris.csv')
df
ps=50/150
piv=50/150
pv=50/150
#Segregating the different kinds of flower into different dataframe.
#For the flower:- Iris-setosa
ISDF=df[df['Species']=='Iris-setosa']
ISDF
#For the flower:- Iris-virginica
IVDF=df[df['Species']=='Iris-virginica']
IVDF
#For the flower:- Iris-versicolor
IVCDF=df[df['Species']=='Iris-versicolor']
IVCDF
ISDF.shape[0]
IVDF.shape[0]
IVCDF.shape[0]
##For setosa Flower
ISDF.mean(axis=0)
ISDFMSL=5.006
ISDFMSW=3.418
ISDFMPL=1.464
ISDFMPW=0.244
ISDF.std(axis=0)
ISDFSTDSL=0.352490
ISDFSTDSW=0.381024
ISDFSTDPL=0.173511
ISDFSTDPW=0.107210

#When x i.e SepalLength = 4.7

x=4.7
ssl=1/(ISDFSTDSL*np.sqrt(2*np.pi))*(np.exp((-np.abs(x-ISDFMSL)**2)/2*ISDFSTDSL*ISDFSTDSL))
print(ssl)    

# #When x i.e Sepalwidth = 3.7

x=3.7
ssw=1/(ISDFSTDSW*np.sqrt(2*np.pi))*(np.exp((-np.abs(x-ISDFMSW)**2)/2*ISDFSTDSW*ISDFSTDSW))
print(ssw)    

#When x i.e PetalLength = 2

x=2
spl=1/(ISDFSTDPL*np.sqrt(2*np.pi))*(np.exp((-np.abs(x-ISDFMPL)**2)/2*ISDFSTDPL*ISDFSTDPL))
print(spl)   

#When x i.e PetalWidth = 0.3

x=0.3
spw=1/(ISDFSTDPW*np.sqrt(2*np.pi))*(np.exp((-np.abs(x-ISDFMPW)**2)/2*ISDFSTDPW*ISDFSTDPW))
print(spw)    
                
                
                
                
pis=ssl*ssw*spl*spw*ps
pis
print("Probability of Iris-Setosa is:",str(round(pis,4)))
IVCDF.mean(axis=0)
IVCDFMSL=5.936
IVCDFMSW=2.770
IVCDFMPL=4.260
IVCDFMPW=1.326
IVCDF.std(axis=0)
IVCDFSTDSL=0.516171
IVCDFSTDSW=0.313798
IVCDFSTDPL=0.469911
IVCDFSTDPW=0.197753
#When x i.e SepalLength = 4.7

x=4.7
vcsl=1/(IVCDFSTDSL*np.sqrt(2*np.pi))*(np.exp((-np.abs(x-IVCDFMSL)**2)/2*IVCDFSTDSL*IVCDFSTDSL))
print(vcsl)    

# #When x i.e Sepalwidth = 3.7

x=3.7
vcsw=1/(IVCDFSTDSW*np.sqrt(2*np.pi))*(np.exp((-np.abs(x-IVCDFMSW)**2)/2*IVCDFSTDSW*IVCDFSTDSW))
print(vcsw)    

#When x i.e PetalLength = 2

x=2
vcpl=1/(IVCDFSTDPL*np.sqrt(2*np.pi))*(np.exp((-np.abs(x-IVCDFMPL)**2)/2*IVCDFSTDPL*IVCDFSTDPL))
print(vcpl)   

#When x i.e PetalWidth = 0.3

x=0.3
vcpw=1/(IVCDFSTDPW*np.sqrt(2*np.pi))*(np.exp((-np.abs(x-IVCDFMPW)**2)/2*IVCDFSTDPW*IVCDFSTDPW))
print(vcpw)    
                
                
                
                
pivc=vcsl*vcsw*vcpl*vcpw*piv
pivc
print("Probability of Iris-Versicolor is:",str(round(pivc,4)))
IVDF.mean(axis=0)
IVDFMSL=6.588
IVDFMSW=2.974
IVDFMPL=5.552
IVDFMPW=2.026
IVDF.std(axis=0)
IVDFSTDSL=0.635880
IVDFSTDSW=0.322497
IVDFSTDPL=0.551895
IVDFSTDPW=0.274650
#When x i.e SepalLength = 4.7

x=4.7
vsl=1/(IVDFSTDSL*np.sqrt(2*np.pi))*(np.exp((-np.abs(x-IVDFMSL)**2)/2*IVDFSTDSL*IVDFSTDSL))
print(vsl)    

# #When x i.e Sepalwidth = 3.7

x=3.7
vsw=1/(IVDFSTDSW*np.sqrt(2*np.pi))*(np.exp((-np.abs(x-IVDFMSW)**2)/2*IVDFSTDSW*IVDFSTDSW))
print(vsw)    

#When x i.e PetalLength = 2

x=2
vpl=1/(IVDFSTDPL*np.sqrt(2*np.pi))*(np.exp((-np.abs(x-IVDFMPL)**2)/2*IVDFSTDPL*IVDFSTDPL))
print(vpl)   

#When x i.e PetalWidth = 0.3

x=0.3
vpw=1/(IVDFSTDPW*np.sqrt(2*np.pi))*(np.exp((-np.abs(x-IVDFMPW)**2)/2*IVDFSTDPW*IVDFSTDPW))
print(vpw)    

piv=vsl*vsw*vpl*vpw*pv
piv
print("Probability of Iris-Virginica is:",str(round(piv,4)))
print("Probability of Iris-Setosa is:",str(round(pis,4)))
print("Probability of Iris-Verginica is:",str(round(piv,4)))
print("Probability of Iris-Versicolor is:",str(round(pivc,4)))
