import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
y = train[['P']]
tr = train[['id']]
te = test[['id']]
# print(tr,te)

df_null = train.isnull().sum().sort_values(ascending=False)
df_null_percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([df_null,df_null_percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
#Other options for the strategy parameter are median or most_frequent
imr.fit(train[['N','B']])
train[['N','B']] = imr.transform(train[['N','B']].values)
test[['N','B']] = imr.transform(test[['N','B']].values)
# train
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


X = pd.DataFrame(train[['A','G','F','D','E']])
te_x = pd.DataFrame(train[['A','G','F','D','E']])
train[['A','G','F','D','E']] = DataFrameImputer().fit_transform(X)
test[['A','G','F','D','E']] = DataFrameImputer().fit_transform(te_x)


# print('before...')
# print(X)
# print('after...')
# print(xt)
from collections import defaultdict
d = defaultdict(LabelEncoder)

fit = train.apply(lambda x: d[x.name].fit_transform(x))
test2 = test.apply(lambda x: d[x.name].fit_transform(x))
sns.set()
sns.pairplot(fit, size = 2.5)
plt.show()
col = fit.columns
un = {} 
for i in col:
    un[i] = fit[i].unique() #getting the unique number of a column
# print(un)
# print(len(un['B']))
binary = {}
for u in un:
#     print(u)
    if len(un[u]) <4:
        binary[u] = fit[u].unique()
# fit['A'].unique()
print(binary)

zero_list = []
one_list = []
two_list = []
for col in binary:
    zero_list.append((fit[col]==0).sum())
    one_list.append((fit[col]==1).sum())
    two_list.append((fit[col]==2).sum())
trace1 = go.Bar(
    x=binary,
    y=zero_list ,
    name='Zero count'
)
trace2 = go.Bar(
    x=binary,
    y=one_list,
    name='One count'
)
trace3 = go.Bar(
    x=binary,
    y=two_list,
    name='two count'
)

data = [trace1, trace2, trace3]
layout = go.Layout(
    barmode='stack',
    title='Count of 2, 1 and 0 in binary variables'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')
colormap = plt.cm.magma
plt.figure(figsize=(16,24))
sns.heatmap(fit.corr(), linewidths=0.1, vmax=0.8, square= True, cmap=colormap, linecolor='white', annot = True)


# fit.drop('D', axis=1, inplace=True)
# test2.drop('D', axis=1, inplace=True)
fit.drop('P', axis=1, inplace=True)
# fit.drop('M', axis=1, inplace=True)
fit.drop('id', axis=1, inplace=True)
# test2.drop('M', axis=1, inplace=True)
test2.drop('id', axis=1, inplace=True)
# fit.drop('A', axis=1, inplace=True)
fit.columns
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
fit_std = stdsc.fit_transform(fit)
test2_std = stdsc.transform(test2)
fit_std = pd.DataFrame(fit_std)
test2_std = pd.DataFrame(test2_std)
fit_std.head()
fit_std.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','O']
test2_std.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','O']
# fit_std = fit
# test2_std = test2

fit_std.head()
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
fit_pca = pca.fit_transform(fit_std[['A', 'D', 'E','G' ,'L', 'M']])
test2_pca = pca.transform(test2_std[['A', 'D', 'E','G' ,'L', 'M']])
fit_pca = pd.DataFrame(fit_pca)
test2_pca = pd.DataFrame(test2_pca)
# fit_pca.Columns([ 'B', 'C',  'F', 'H', 'I',  'N', 'O'])
fit_pca[[ 'B', 'C', 'J', 'F', 'H', 'I', 'K',  'N', 'O']] =fit_std[[ 'B', 'C', 'J' ,'F', 'H', 'I', 'K',  'N', 'O']]
test2_pca[[ 'B', 'C', 'J', 'F', 'H', 'I', 'K', 'N', 'O']] = test2_std[['B', 'C', 'J', 'F', 'H', 'I', 'K', 'N', 'O']]
# fit_pca[[ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','O']] =fit[[ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','O']]
# test2_pca[[ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','O']] = test2[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','O']]
# fit_pca.drop('G', axis=1, inplace=True)
# test2_pca.drop('G', axis=1, inplace=True)
test2_pca.head()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(fit,y, test_size=0.3, random_state=1 )
X_train_pca,X_test_pca,y_train_pca,y_test_pca = train_test_split(fit_pca,y, test_size=0.3, random_state=100 )
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
feat_label = fit_pca.columns[:]
forest = RandomForestClassifier(n_estimators=1400, max_depth=12, min_samples_leaf=4, max_features=0.5, n_jobs=-1, random_state=0)
forest.fit(X_train_pca,y_train_pca) #in the random forest we dont need to standardize the data
importances = forest.feature_importances_
indeces = np.argsort(importances)[::-1]
for f in range(X_train_pca.shape[1]):
    print(feat_label[indeces[f]], importances[indeces[f]])
y_pred = forest.predict(X_test_pca)
forest_predic = forest.predict(test2_pca)
random_cm = confusion_matrix(y_test_pca,y_pred)
random_cm
from  sklearn.linear_model  import LogisticRegression
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 20)
lr = LogisticRegression(C=100, class_weight='balanced', max_iter=10000)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)
# test_data_pca = pca.transform(test_data)
lr.fit(X_train_pca,y_train_pca)
lr.score(X_test_pca,y_test_pca)

logistic_predict = lr.predict(test2_pca)
logistic_predict
sub = pd.DataFrame()
sub['id'] = te
sub['P'] = forest_predic
sub.to_csv('random_forest_submit9.csv', float_format='%.6f', index=False)
