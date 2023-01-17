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
import pandas as pd

#WTI Oil price data from Jan-01 to Jun-8
wti_crude = pd.read_csv('../input/wtioil2/Crude_oil_price_ytd_.csv')
wti_crude.head()

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

#Read csv files
wti_crude = pd.read_csv('../input/wtioil2/Crude_oil_price_ytd_.csv')
#wti_crude.head()
#df=pd(wti_crude,columns=['Date','Price'])
x1=wti_crude.Date
y1=wti_crude.Price
y2=wti_crude.High
y3=wti_crude.Low
#print(x1)
plt.figure(figsize=(25, 10))
#plt.tick_params(labelsize=11,pad=5);
plt.plot(x1, y1,label='Price',color='black')
plt.plot(x1, y2,label='High',color='Green')
plt.plot(x1, y3,label='Low',color='Red')
plt.xlabel('Date Jan-01 to June-30 ------>',fontsize=21)
plt.ylabel('Oil Price',fontsize=21) 
plt.title('Oil price variation YTD 2020',fontsize=21)
plt.rc('ytick', labelsize=30)
plt.rc('xtick', labelsize=7.6)
plt.legend(fontsize=26)
plt.xticks(rotation=90)
plt.show()
#wti_crude.plot(x2='Date', y2='High',kind = "line")

x1=wti_crude.Date
y1=wti_crude.Change_percent
#print(y1)
plt.figure(figsize=(25, 10))
#plt.tick_params(labelsize=11,pad=5);
plt.xticks(rotation=90)
plt.plot(x1, y1,label='Change%',color='black')
plt.xlabel('Date Jan-01 to June-30 ------>',fontsize=21)
plt.ylabel('Change %',fontsize=21) 
plt.title('Oil price change% YTD 2020',fontsize=21)

#print(y1)
#plt.ylim(-305,38)
#plt.yticks(-0.1,0.38)