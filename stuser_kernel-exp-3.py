import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
plt.style.use('ggplot')
plt.rcParams['font.family']='SimHei' #顯示中文(for Mac)
plt.rcParams['axes.unicode_minus']=False #正常顯示負號

pd.set_option("display.max_columns",40) #設定pandas最多顯示出40個欄位資訊
#pd.set_option("display.html.table_schema",True)
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/input-iris/train.csv")
test = pd.read_csv("../input/input-iris/test.csv")
train = train.dropna().reset_index(drop=True)
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['type'], axis=1, inplace=True)
all_data.head(3)
from scipy import stats
from scipy.stats import norm, skew #for some statistics

numeric_feats = ['花萼寬度','花萼長度','花瓣寬度','花瓣長度']

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
import seaborn as sns
#histogram and normal probability plot
plt.figure(figsize=(16, 4))
plt.rcParams['axes.unicode_minus']=False #正常顯示負號
plt.rcParams['font.size']=12

plt.subplot(1,4,1)
plt.title('花萼寬度')
sns.distplot(all_data['花萼寬度'], fit=norm);
plt.subplot(1,4,2)
res = stats.probplot(all_data['花萼寬度'], plot=plt)

plt.subplot(1,4,3)
plt.title('花萼長度')
sns.distplot(all_data['花萼長度'], fit=norm);
plt.subplot(1,4,4)
res = stats.probplot(all_data['花萼長度'], plot=plt)
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.90 #0.75
# lam 參數說明:
# y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
#     log(1+x)                    if lmbda == 0
box_cols = ['花萼寬度','花萼長度']
all_data[box_cols] = boxcox1p(all_data[box_cols], lam)

#all_data[numeric_feats] = np.log1p(all_data[numeric_feats])
import seaborn as sns
#histogram and normal probability plot
plt.figure(figsize=(16, 4))
plt.rcParams['axes.unicode_minus']=False #正常顯示負號
plt.rcParams['font.size']=12

plt.subplot(1,4,1)
plt.title('花萼寬度')
sns.distplot(all_data['花萼寬度'], fit=norm);
plt.subplot(1,4,2)
res = stats.probplot(all_data['花萼寬度'], plot=plt)

plt.subplot(1,4,3)
plt.title('花萼長度')
sns.distplot(all_data['花萼長度'], fit=norm);
plt.subplot(1,4,4)
res = stats.probplot(all_data['花萼長度'], plot=plt)
