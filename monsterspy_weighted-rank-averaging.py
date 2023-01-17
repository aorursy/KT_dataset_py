import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')
res_1 = pd.read_csv('../input/results/test_1.csv')
res_2 = pd.read_csv('../input/results/test_2.csv')
res = res_1[['Image_Id']]
columns = res_1.columns[1:]
columns
res_1['Atelectasis'].head()
for col in columns:

    data = {}
    data['file_1'] = res_1[['Image_Id', col]]
    data['file_2'] = res_2[['Image_Id', col]]

    ranks = pd.DataFrame(columns=data.keys())
    for key in data.keys():
        ranks[key] = data[key][col].rank(method='min')
    ranks['Average'] = ranks.mean(axis=1)
    ranks['Scaled Rank'] = (ranks['Average'] - ranks['Average'].min()) / (ranks['Average'].max() - ranks['Average'].min())
    ranks.corr()[:1]

    weights = [0.6, 0.4]
    ranks['Score'] = ranks[['file_1', 'file_2']].mul(weights).sum(1) / ranks.shape[0]
    
    res[col] = ranks['Score']
res.head()


