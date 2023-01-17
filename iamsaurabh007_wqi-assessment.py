# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/mine-water-dataset/minewater.csv",index_col="S.No")
para=pd.read_csv("/kaggle/input/parameters/para.csv")
para
a=0
for idx in para.index:
     a=a+1/para["Standard Values"][idx]
k=1/a
k
wi=[]
for idx in para.index:
    wi.append(k/para["Standard Values"][idx])
para["Weight"]=wi
para
sumwi=para["Weight"].sum()
sumwi
df.drop(["BOD","Turbidity","Iron"],axis=1,inplace=True)
df
q=[]
a=0
for j in df.columns[1:11]:
    p=[]
    for i in df.index:
        qi=100*(abs(df[j][i]-para["Ideal Values"][a])/abs(para["Standard Values"][a]-para["Ideal Values"][a]))
        p.append(para["Weight"][a]*qi)
    q.append(p)
    a=a+1
a
qw=pd.DataFrame(q,index=para["Parameters"],columns=df.index)
qw=qw.T
qw
qw.columns
qw["WQI"]=qw.sum(axis=1)
qw
qw.corr()