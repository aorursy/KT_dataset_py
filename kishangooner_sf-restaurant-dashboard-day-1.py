# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()
import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
data_res=pd.read_csv('../input/restaurant-scores-lives-standard.csv')
data_res.head()
data_res.shape
data_res.isna().sum()
data_res.inspection_score.isna().value_counts()

data_res.groupby(data_res.risk_category).count()
data_new1=data_res.copy()
data_new1=data_new1.dropna(axis=0,subset=['risk_category','inspection_score'])
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
sns.catplot(x="risk_category", y="inspection_score",kind='box' ,data=data_new1);

# pie chart of workers
#labels = ['Self-employed', 'Works at tech company', 'Has a tech role in non-tech company', 'Has a non-tech role at a non-tech company']
sizes = [data_new1['inspection_type'].value_counts()]

print(sizes) # adds up to 1433, which is the total number of participants



temp = data_new1[['inspection_type', 'risk_category']]
for col in temp.columns:
    fig = {
      "data": [
        {
          "values": temp[col].value_counts(),
          "labels": list(temp[col].value_counts().index),
          "name":col,
          "hoverinfo":"label+percent+name",
          "hole": .1,
          "type": "pie"
        }],
        
      "layout": {
            "title":col
        }
        }
    iplot(fig, filename='pie')
