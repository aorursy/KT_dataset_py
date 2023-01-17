#Loading Liberary

import numpy as np 

import pandas as pd 

import os

import json

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')
print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ad_compaign_data = pd.read_csv('../input/data.csv')

ad_compaign_data.head()

ad_compaign_data.info()
ad_compaign_data.describe()
ad_compaign_data['age'].unique()
fig, ax = plt.subplots(figsize = (16, 6))

plt.subplot(1, 1, 1)

plt.title('Distribution of clicks');

ad_compaign_data['clicks'].plot('hist', label='clicks');

plt.legend();