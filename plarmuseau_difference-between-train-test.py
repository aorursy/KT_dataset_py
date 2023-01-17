from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
from datetime import date
import pandas as pd
import numpy as np 
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
import random 
import warnings
import operator
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

## list of features to be used
features = [c for c in train.columns if c not in ['Id', 'Target']]

## target variable 
target = train['Target'].values
target_index = {1:0, 2:1, 3:2, 4:3}
target = np.array([target_index[c] for c in target])
def label_encoding(col):
    le = LabelEncoder()
    le.fit(list(train[col].values) + list(test[col].values))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

num_cols = train._get_numeric_data().columns 
cat_cols = list(set(features) - set(num_cols))
for col in cat_cols:
    label_encoding(col)
totaal=train.append(test)
from sklearn.decomposition import TruncatedSVD
totaal=(train.append(test)).fillna(0)
totaal=totaal.drop(['Id','Target'],axis=1)  #.reset_index()
temp=totaal.iloc[:,1:135].divide(totaal['rooms'],axis=0)
totaal=(totaal.T.append(temp.T)).T
totaal=totaal.join(temp, lsuffix='', rsuffix='persons')
svd = TruncatedSVD(n_components=140, n_iter=7, random_state=42)
e_=svd.fit_transform(totaal)
#A_,e1_,e_,s_=robustSVD(e_,100)
New_features =  e_[:len(train)]
Test_features= e_[-len(test):]
pd.DataFrame(New_features).plot.scatter(x=0,y=1,c=train['Target']+1)
pd.DataFrame(np.concatenate((Test_features,New_features))).plot.scatter(x=0,y=1,c=[1 for x in range(len(test))]+[2 for x in range(len(train))],colormap='viridis')    

def cohen_effect_size(X, y):
    """Calculates the Cohen effect size of each feature.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        Returns
        -------
        cohen_effect_size : array, shape = [n_features,]
            The set of Cohen effect values.
        Notes
        -----
        Based on https://github.com/AllenDowney/CompStats/blob/master/effect_size.ipynb
    """
    print(X.shape,y.shape,y.mean())
    medi=y.mean()
    group1, group2 = X[y<medi], X[y>=medi]
    diff = group1.mean() - group2.mean()
    var1, var2 = group1.var(), group2.var()
    n1, n2 = group1.shape[0], group2.shape[0]
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d
excluded_feats = ['ID','Id','Target'] #['SK_ID_CURR']

features = [f_ for f_ in train.drop('Target',axis=1).columns if f_ not in excluded_feats]
print('Number of features %d' % len(features),train.shape,target.shape)
#effect_sizes = cohen_effect_size(Xtrain[:len(ytrain)], ytrain)
effect_sizes = cohen_effect_size(train.drop('Target',axis=1)[:len(target)],pd.DataFrame(train).reset_index().set_index('index')['Target'])
effect_sizes.reindex(effect_sizes.abs().sort_values(ascending=False).nlargest(50).index)[::-1].plot.barh(figsize=(6, 10));
print('Features with the 30 largest effect sizes')
significant_features2 = [f for f in features if np.abs(effect_sizes.loc[f]) > 0.1]
print('Significant features %d: %s' % (len(significant_features2), significant_features2))
