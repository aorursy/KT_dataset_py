import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 130
train = pd.read_csv('../input/train.csv')
pd.set_option('display.max_rows', None)
train.sample(10).T
pd.reset_option('display.max_rows')
num_feat = train.select_dtypes(include=[np.number]).columns
num_feat
len(num_feat)
train.describe().T
cat_feat = train.select_dtypes(exclude=[np.number]).columns
cat_feat
len(cat_feat)
(train.isnull().sum()/len(train)*100).sort_values(ascending=False)[:25]
corrmat = train.corr(method='spearman')
plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, mask=np.eye(len(corrmat)), vmax=1.0, vmin=-1.0, square=True, center=0);
sns.distplot(train['SalePrice'], bins=100);







