import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
data=pd.read_csv('../input/titanic/train.csv')
data
f,ax=plt.subplots(1,2,figsize=(18,8))

#data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

#ax[0].set_title('Survived')

#ax[0].set_ylabel('')

sns.countplot('Pclass',data=data,ax=ax[1])

ax[0].set_title('Pclass')

plt.show()
