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
df=pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Feature-Engineering-Live-sessions/master/mobile_dataset.csv')
df.head()
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x.head()
y.head()
from sklearn.feature_selection import SelectKBest, chi2
df
###apply select kbest algorithm
ordered_rank_feature=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_feature.fit(x,y)
df_score=pd.DataFrame(ordered_feature.scores_,columns=['Score'])
df_columns=pd.DataFrame(x.columns)
feature_rank=pd.concat([df_columns,df_score],axis=1)##
feature_rank.columns=['feature','Score']
feature_rank
feature_rank.nlargest(10,'Score')
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
%matplotlib inline

model=ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
ranked_feature=pd.Series(model.feature_importances_,index=x.columns)
ranked_feature.nlargest(10,).plot(kind='barh')
plt.show()
import seaborn as sns
corr=df.iloc[:,:-1].corr()
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(df[top_features].corr(),annot=True)

#remove the correlated
threshold=0.5
def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
correlation(df.iloc[:,:-1],threshold)
from sklearn.feature_selection import mutual_info_classif
mutual_info=mutual_info_classif(x,y)
mutual_data=pd.Series(mutual_info,index=x.columns)
mutual_data.sort_values(ascending=False)
