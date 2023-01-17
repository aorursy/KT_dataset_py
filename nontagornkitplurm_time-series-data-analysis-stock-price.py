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
import pandas as pd #ที่มีความสามารถในการทำdata cleanning หรือ อ่านไฟล์CSVออกมาเป็นdata frameไว้สำหรับการทำ Data Visualization 
import numpy as np #เกี่ยวกับคณิตศาสตร์และการคำนวณต่างๆ
from pandas import Series,DataFrame
import matplotlib.pyplot as plt #ไว้สำหรับวาดกราฟ
import seaborn as sns #แสดงภาพข้อมูล Python ที่ใช้ matplotlib 
sns.set_style('whitegrid')
#ไว้พล็อตสวยๆ
from datetime import datetime
#from __future__ import division
AAPL=pd.read_csv('../input/sandp500/individual_stocks_5yr/individual_stocks_5yr/AAPL_data.csv')
GOOG=pd.read_csv('../input/sandp500/individual_stocks_5yr/individual_stocks_5yr/GOOG_data.csv')
MSFT=pd.read_csv('../input/sandp500/individual_stocks_5yr/individual_stocks_5yr/MSFT_data.csv')
AMZN=pd.read_csv('../input/sandp500/individual_stocks_5yr/individual_stocks_5yr/AMZN_data.csv')
AAPL.head() #ถ้าไม่ใส่argumentจะโชว์5แถว
AAPL.index  #ประเภทดัชนีเริ่มต้นที่ใช้โดย DataFrame และ Seriesช่วยประหยัดหน่วยความจำ
AAPL.index=pd.to_datetime(AAPL.index)
#AAPL.index
AAPL.describe().T
AAPL.info()
AAPL['date']=pd.to_datetime(AAPL['date'])
GOOG['date']=pd.to_datetime(GOOG['date'])
MSFT['date']=pd.to_datetime(MSFT['date'])
AMZN['date']=pd.to_datetime(AMZN['date'])
AAPL.info()
GOOG.info()
AAPL.plot(x='date', y='close',figsize=(10,4)) #figsizeขนาดกราฟหากไม่ระบุค่าเริ่มต้น
plt.show()
title='VOLUME TRADED'
ylabel='Volume'
xlabel='Time'
ax=AAPL.plot(x='date', y='volume',figsize=(10,4));
ax.autoscale(axis='x')  # use both if want to scale both axis
ax.set(xlabel=xlabel,ylabel=ylabel)
plt.show()
AAPL.plot(x='date', y='close',xlim=['2016-01-01','2017-12-31'],ylim=[80,180],legend=True,figsize=(10,4),ls='--',c='red')
plt.show()
AAPL['close_10']=AAPL['close'].rolling(10).mean()
AAPL['close_50']=AAPL['close'].rolling(50).mean()
ax=AAPL.plot(x='date',y='close',title='AAPL Close Price',figsize=(10,4))
AAPL.plot(x='date',y='close_10',color='red',ax=ax)
AAPL.plot(x='date',y='close_50',color='k',ax=ax)
plt.show()
AAPL['Daily Return']=AAPL['close'].pct_change() #เปอร์เซ็นต์การเปลี่ยนแปลงระหว่างองค์ประกอบปัจจุบันและองค์ประกอบก่อนหน้า
AAPL['Daily Return'].plot(figsize=(15,4),linestyle='--',marker='o')
plt.show()
sns.distplot(AAPL['Daily Return'].dropna(),bins=2000,color='purple')
#Distplot ย่อมาจากพล็อตการกระจาย ซึ่งมันจะใช้เป็นอินพุตอาร์เรย์และพล็อตโค้งที่สอดคล้องกับการกระจายของคะแนนในอาร์เรย์
plt.show() #bin=ลำดับของสเกลาร์ #dropna ลบแถวที่เป็นNANหรือช่องว่าง สามารถใส่ค่าให้ลบทั้งคอลัมแทนได้
AAPL['Daily Return'].hist(bins=100)
plt.show()
df=AAPL['date'].copy()
df=pd.DataFrame(df)
df['AAPL']=AAPL['close']
df['GOOG']=GOOG['close']
df['MSFT']=MSFT['close']
df['AMZN']=AMZN['close']
df.drop(['date'], axis = 1, inplace = True, errors = 'ignore')
tech_rets=df.pct_change()
tech_rets=pd.DataFrame(tech_rets)
tech_rets['date']=AAPL['date']
tech_rets.shape
import scipy.stats as stats
sns.jointplot('GOOG','GOOG',tech_rets,kind='scatter',color='seagreen').annotate(stats.pearsonr)
plt.ioff()
sns.jointplot('AMZN','AAPL',tech_rets,kind='scatter',color='seagreen').annotate(stats.pearsonr)

rets=tech_rets.dropna() #ลบแถวที่มีค่าว่างในdataframe
area=np.pi*20
plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x,y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(
        label,
        xy=(x,y),xytext=(50,50),
        textcoords='offset points',ha='right',va='bottom',
        arrowprops=dict(arrowstyle='-',connectionstyle='arc,rad=-0.3'))
