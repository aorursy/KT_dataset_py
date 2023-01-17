import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
from datetime import datetime
challenge9 = pd.read_csv('../input/exercising-september/2020.9_.csv')
challenge9.head()
challenge9.info()
challenge9['일자'] = challenge9['일자'].astype(str)
challenge9.info()
challenge9['일자'] = pd.to_datetime(challenge9['일자'], format="%Y/%m/%d")
challenge9.info()
challenge9.head()
challenge9['시작시간'] = pd.to_datetime(challenge9['시작시간'], format="%H:%M")
challenge9.head()
challenge9.info()
challenge9['시작시간'].mean()
challenge9.describe()
fig = plt.figure(figsize=(10,5)) 

plt.plot(challenge9['일자'], challenge9['시작시간'], marker='.', color='dodgerblue')

plt.title('Exercise Starting Time On September 2020', fontsize=20) 

plt.xlabel('Date', fontsize=15)

plt.ylabel('Starting Time', fontsize=15)

plt.show()
challenge9['시작시간'].mean()
challenge9['시작시간'].dt.time