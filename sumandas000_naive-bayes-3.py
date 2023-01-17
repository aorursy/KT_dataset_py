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
import numpy

import pandas

import seaborn as sns
flower=pd.read_csv('/kaggle/input/iris/Iris.csv')
x=flower.groupby('Species').count()

x
p=1/3
flower.head()
a=flower[flower['Species']=="Iris-setosa"]

b=flower[flower['Species']=="Iris-virginica"]

c=flower[flower['Species']=="Iris-versicolor"]
a.describe()
a_std_sl=0.3525

a_std_sw=0.381

a_std_pl=0.1735

a_std_pw=0.1072

a_mean_sl=5.006

a_mean_sw=3.418

a_mean_pl=1.464

a_mean_pw=0.244
x=1/(a_std_sl*(2*3.14)**0.5)

y=2.7182**(-0.5*((4.7-a_mean_sl)/a_std_sl)**2)

pslyes=x*y

pslyes
x=1/(a_std_sw*(2*3.14)**0.5)

y=2.7182**(-0.5*((3.7-a_mean_sw)/a_std_sw)**2)

pswyes=x*y

pswyes
x=1/(a_std_pl*(2*3.14)**0.5)

y=2.7182**(-0.5*((2-a_mean_pl)/a_std_pl)**2)

pplyes=x*y

pplyes
u=a[a['PetalWidthCm']==0.3].shape[0]

v=a.shape[0]

ppwyes=u/v

ppwyes
setosa=ppwyes*pplyes*pslyes*pswyes*p
setosa
b.describe()
spl_mean=6.58

spw_mean=2.974

ptl_mean=5.552

ptw_mean=2.026

spl_std=0.635

spw_std=0.322

ptl_std=0.551

ptw_std=0.274
x=1/(spl_std*(2*3.14)**0.5)

y=2.7182**(-0.5*((4.7-spl_mean)/spl_std)**2)

pslyes=x*y

pslyes
x=1/(spw_std*(2*3.14)**0.5)

y=2.7182**(-0.5*((3.7-spw_mean)/spw_std)**2)

pswyes=x*y

pswyes
x=1/(ptl_std*(2*3.14)**0.5)

y=2.7182**(-0.5*((2-ptl_mean)/ptl_std)**2)

pplyes=x*y

pplyes
x=1/(ptw_std*(2*3.14)**0.5)

y=2.7182**(-0.5*((0.3-ptw_mean)/ptw_std)**2)

ppwyes=x*y

ppwyes
virginica=ppwyes*pplyes*pswyes*pslyes*p
virginica
c.describe()
spl_mean2=5.936

spw_mean2=2.770

ptl_mean2=4.26

ptw_mean2=1.326

spl_std2=0.516

spw_std2=0.313

ptl_std2=0.469

ptw_std2=0.197
x=1/(ptw_std2*(2*3.14)**0.5)

y=2.7182**(-0.5*((0.3-ptw_mean2)/ptw_std2)**2)

ppwyes=x*y

ppwyes
x=1/(ptl_std2*(2*3.14)**0.5)

y=2.7182**(-0.5*((2-ptl_mean2)/ptl_std2)**2)

pplyes=x*y

pplyes
x=1/(spl_std2*(2*3.14)**0.5)

y=2.7182**(-0.5*((4.7-spl_mean2)/spl_std2)**2)

pslyes=x*y

pslyes
x=1/(spw_std2*(2*3.14)**0.5)

y=2.7182**(-0.5*((3.7-spw_mean2)/spw_std2)**2)

pswyes=x*y

pswyes
versicolor=p*pswyes*pslyes*pplyes*ppwyes

versicolor