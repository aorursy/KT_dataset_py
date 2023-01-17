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
import pandas as pd

# Plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline

# Models
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Over
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from scipy.stats import norm
from scipy import stats

import missingno as msno
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# To draw a tree
from graphviz  import Source
from IPython.display import SVG, display, HTML
style = "<style>svg{width: 40% !important; height: 50% !important;} </style>"
HTML( style )
import os
os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\graphviz2.38\\bin" + os.pathsep + "C:\\Program Files (x86)\\graphviz2.38"
df = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
pd.options.display.max_columns = df.shape[1] # let's show all columns
df.head()
msno.matrix(df)
df.isnull().sum()
df.describe()
y = df['price_range']
sns.distplot(y, kde=False)
df.info()
for i, j in enumerate(list(df)):
    print(j, len(df[j].unique()))
def categorical(df, a):
    cat_col = []
    for i in df.columns:
        if len(df[i].unique()) <= a:
            cat_col.append(i)
    return(cat_col)
cat_columns = categorical(df, 8)
df[cat_columns].hist(figsize=(15,10), bins=50)
sns.heatmap(data=df[cat_columns].corr(), cbar=True, annot=True, square=True, annot_kws={'size': 10},)
df[df['three_g'] == 0]['four_g'].hist()
num_columns = [x for x in df.columns if x not in cat_columns]
for i in num_columns:
    sns.distplot(df[i])
    plt.show()
df_size = df[['sc_w', 'px_width', 'sc_h', 'px_height']]
df_size_not_zero = df_size.drop(df_size[(df_size['sc_w'] == 0) | (df_size['px_height'] == 0)].index)
df_size_not_zero.corr()
sns.heatmap(data=df_size_not_zero.corr(), yticklabels=['sc_w           ', 'px_width      ', 'sc_h             ', 'px_height      '],
           cbar=True, annot=True, square=True)
df_size_not_zero['area_sc'] = df_size_not_zero['sc_w'] * df_size_not_zero['sc_h']
df_size_not_zero['area_px'] = df_size_not_zero['px_width'] * df_size_not_zero['px_height']
sns.heatmap(data=df_size_not_zero.corr(), cbar=True, annot=True, square=True, annot_kws={'size': 10},)
sns.jointplot(x='sc_w' , y='sc_h' , data=df_size_not_zero)
df['sc_w'] = df[~df.apply(lambda x: x.eq(0))]['sc_w']
df['px_height'] = df[~df.apply(lambda x: x.eq(0))]['px_height']
df['sc_w'] = df.groupby('sc_h')['sc_w'].transform(lambda x: x.fillna(x.median()))
df['px_height'] = df.groupby('px_width')['px_height'].transform(lambda x: x.fillna(x.median()))
fig = plt.figure(figsize=(15,12))
r = sns.heatmap(df.corr(), cmap='Purples')
r.set_title("Correlation ")
num_top10_corr = df.corr()['price_range'].sort_values(ascending=False).head(10).to_frame()
cm = sns.light_palette("blue", as_cmap=True)
s = num_top10_corr.style.background_gradient(cmap=cm)
s
sns.boxplot(x='price_range', y='ram', data=df)
plt.figure(figsize=(10,6))
df['fc'].hist(alpha=0.5, color='blue', label='Front camera')
df['pc'].hist(alpha=0.5, color='red', label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')
g = sns.FacetGrid(df, col="dual_sim", hue="price_range", palette="Set1", height=5)
g = g.map(sns.distplot, "ram").add_legend()
g = sns.FacetGrid(df, col="touch_screen", hue="price_range", palette="Set1", height=5)
g = g.map(sns.distplot, "ram").add_legend()
g = sns.FacetGrid(df, col="wifi", hue="price_range", palette="Set1", height=5)
g = g.map(sns.distplot, "ram").add_legend()
g = sns.FacetGrid(df, col="blue", hue="price_range", palette="Set1", height=5)
g = g.map(sns.distplot, "ram").add_legend()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["price_range"]):
    df_train = df.loc[train_index]
    df_test = df.loc[test_index]
df_train["price_range"].hist(figsize=(10,5), bins=10)
print('Размер train выборки', df_train.size)
df_test["price_range"].hist(figsize=(10,5), bins=10)
print('Размер test выборки', df_test.size)
y_train = df_train['price_range']
X_train = df_train.drop('price_range', axis=1)
y_test = df_test['price_range']
X_test = df_test.drop('price_range', axis=1)
max_depth_values = range(1,100)
scores_data = pd.DataFrame()
best_depth = 0
best_score = 0
for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    if test_score > best_score:
        best_score = test_score
        best_depth = max_depth
    temp_score_data = pd.DataFrame({'max_deph': [max_depth],
                                   'train_score': [train_score],
                                   'test_score': [test_score]})
    scores_data = scores_data.append(temp_score_data)
scores_data_long = pd.melt(scores_data,
                           id_vars=['max_deph'], 
                           value_vars=['train_score','test_score'],
                           var_name=['set_type'], 
                           value_name='score')
sns.lineplot(x='max_deph', y='score', hue='set_type', data=scores_data_long)
print(best_score, best_depth)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=best_depth)
clf.fit(X_train, y_train)
graph = Source(tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=list(X_train),
                                    filled=True))
display(SVG(graph.pipe(format='svg')))
kf = KFold(n_splits=5, random_state=42, shuffle=True)
for k in range(1, 102, 10):
    clf = RandomForestRegressor(n_estimators=k)
    acc = cross_val_score(clf, X_train, y_train, cv=kf)
    acc_mean = np.mean(acc)
    print(acc_mean, k)
clf = RandomForestRegressor(n_estimators=50)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
n_list = []
score_list_knn = []
for n in range(1, 20):
    clf = KNeighborsClassifier(n_neighbors=n)
    aсс = cross_val_score(clf, X_train, y_train, cv=kf)
    score_list_knn.append(np.mean(aсс))
    n_list.append(n)
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))
for n in range(1, 102, 10):
    clf = GradientBoostingClassifier(n_estimators=n)
    acc = cross_val_score(clf, X_train, y_train, cv=kf)
    print(np.mean(acc), n)
for n in range(151, 182, 10):
    clf = GradientBoostingClassifier(n_estimators=n)
    acc = cross_val_score(clf, X_train, y_train, cv=kf)
    print(np.mean(acc), n)
skewness_data = []
kurtosis_data = []
for i in num_columns:
    print('-----', i ,'-----')
    print("Skewness: %f" % df_train[i].skew())
    if abs(df_train[i].skew()) > 0.5:
        print('Check skewness')
        skewness_data.append(i)
    print("Kurtosis: %f" % df_train[i].kurt())
    if abs(df_train[i].kurt()) > 3:
        print('Check kurtosis')
        kurtosis_data.append(i)
    sns.distplot(df[i], fit=norm)
    plt.show()
print('skewness_data', skewness_data)
print('kurtosis_data', kurtosis_data)
df_num_sc = df_train[num_columns]
df_num_sc['px_height'] = np.sqrt(df_num_sc['px_height'])
sns.distplot(df_num_sc['px_height'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_num_sc['px_height'], plot=plt)
print(df_num_sc['px_height'].skew())
df_num_sc['fc'] = np.sqrt(df_num_sc['fc'])
df_num_sc['sc_w'] = np.sqrt(df_num_sc['sc_w'])
print(df_num_sc['fc'].skew())
print(df_num_sc['sc_w'].skew())
scaler = StandardScaler()
df_temp = df_num_sc.copy()
df_num_sc[num_columns] = scaler.fit_transform(df_temp)
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax = sns.boxplot(data=df_num_sc[num_columns] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
df_train[num_columns] = df_num_sc
df_num_sc_t = df_test[num_columns]
df_num_sc_t['px_height'] = np.sqrt(df_num_sc_t['px_height'])
df_num_sc_t['fc'] = np.sqrt(df_num_sc_t['fc'])
df_num_sc_t['sc_w'] = np.sqrt(df_num_sc_t['sc_w'])
df_temp = df_num_sc_t.copy()
df_num_sc_t[num_columns] = scaler.fit_transform(df_temp)
df_test[num_columns] = df_num_sc_t
y_train = df_train['price_range']
X_train = df_train.drop('price_range', axis=1)
y_test = df_test['price_range']
X_test = df_test.drop('price_range', axis=1)
C_list = []
Score_list = []
for C_ in np.arange(0.1, 11, 0.2):
    clf = LogisticRegression(C=C_)
    aсс = cross_val_score(clf, X_train, y_train, cv=kf)
    Score_list.append(np.mean(aсс))
    C_list.append(C_)
plt.plot(C_list, Score_list)
print(max(Score_list))
C_list = []
Score_list = []
for C_ in np.arange(0.25, 10, 0.5):
    clf = SVC(C=C_)
    aсс = cross_val_score(clf, X_train, y_train, cv=kf)
    Score_list.append(np.mean(aсс))
    C_list.append(C_)
plt.plot(C_list, Score_list)
print(max(Score_list))
C_list = []
Score_list = []
for C_ in np.arange(0.1, 11, 0.2):
    clf = OneVsOneClassifier(LogisticRegression(C=C_))
    aсс = cross_val_score(clf, X_train, y_train, cv=kf)
    Score_list.append(np.mean(aсс))
    C_list.append(C_)
plt.plot(C_list, Score_list)
print(max(Score_list))
clf = OneVsOneClassifier(LogisticRegression())

parameters = {
    'estimator__C': np.arange(0.1, 21, 0.2),
    'estimator__penalty': ['l1','l2'],
    'estimator__solver': ['saga']
}

grid_logres = GridSearchCV(clf, param_grid=parameters, cv=kf)
grid_logres.fit(X_train, y_train)
print('best score: ', grid_logres.best_score_)
print('best param: ', grid_logres.best_params_)
parameters = {
    'C': np.arange(0.1, 6, 0.2),
    'kernel': ['linear', 'rbf'],
    'gamma': ['auto', 1, 0.1, 0.01, 0.001],
    'decision_function_shape': ['ovo', 'ovr']
}
# Instead of the OneVsOneClassifier module, we use the SVC parameter - decision_function_shape
clf = SVC()
grid_svc = GridSearchCV(clf, param_grid=parameters)
grid_svc.fit(X_train, y_train)
print('best score: ', grid_svc.best_score_)
print('best param: ', grid_svc.best_params_)
clf = grid_logres.best_estimator_
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
prediction = clf.predict(X_test)
conf_mx = confusion_matrix(y_test, prediction)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums # let's move from absolute indicators to relative
np.fill_diagonal(norm_conf_mx, 0)
sns.heatmap(data=norm_conf_mx, cbar=True, annot=True, square=True)
