# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn

sn.set(style = "white", color_codes = True)

import matplotlib.pyplot as mp

%matplotlib inline 

#how the graphs are printed out

import warnings #suppress certain warnings from Libraries

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



battles = pd.read_csv("../input/battles.csv", header=0, sep = ",")
battles.shape

battles.head(10)
battles.columns
battles[['name', 'year']][:5]
lannister_battles = battles.ix[battles['attacker_1'] == 'Lannister']

lannister_battles.head(5)
len(lannister_battles)

len(lannister_battles['attacker_outcome'] == 'win')
lannister_battles [['name','attacker_king', 'attacker_outcome']]
char_pred = pd.read_csv('../input/character-predictions.csv', header = 0, sep = ',')
char_pred.shape
char_pred.corr()
import seaborn as sns

%matplotlib inline

sns.heatmap(char_pred.corr())
pred_correl = char_pred.corr()

print(pred_correl ['actual'])
pred_correl.ix[pred_correl['actual']>0.1]