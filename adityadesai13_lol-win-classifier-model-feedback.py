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
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from xgboost import plot_tree as plot_xgboost
df = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
df.sample(5)
print(df['blueWins'].value_counts())  # checking the dataset is balanced
df.dtypes # checking if any of the input values need to be turned into numbers e.g. if had categorical info. Not the case here.
# checking if any entries are duplicates - all entries unique so all good
print(len(df['gameId'].unique()))
print(df.shape)
df = df.drop('gameId', axis=1) # unnecessary column
# looking at edge cases i.e. pure white and pure black
corr = df.corr()
plt.figure(figsize=(15,10))
ax= plt.subplot()
sns.heatmap(corr, ax=ax)
# removing columns that have correlated values greater than 0.9, or less than -0.9
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
        if corr.iloc[i,j] <= -0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
df = df[selected_columns]
df.shape # 12 unecessary columns have been removed
X = df.drop('blueWins', axis=1)
y = df['blueWins']
from sklearn import preprocessing
X_scaled = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,random_state=1) 
clf = SVC(kernel = 'linear')
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
rf = RandomForestClassifier(n_estimators=10, max_depth=3)
rf.fit(X_train,y_train)
rf.score(X_test,y_test)
# 5x2cv significance test (Deittrich, 1998)
# p value indicates no significance difference in model performance on the datasets
# will proceed with RF as can use analysis tools (e.g. SHAP, eli5)

clf1 = RandomForestClassifier(n_estimators=10, max_depth=3,random_state=1)
clf2 = SVC(kernel='linear', random_state=1)

from mlxtend.evaluate import paired_ttest_5x2cv


t, p = paired_ttest_5x2cv(estimator1=clf1,
                          estimator2=clf2,
                          X=X, y=y, random_seed=123)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
# confusion matrix

predictions = rf.predict(X_test)
cm = confusion_matrix(y_test, predictions)
cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Learning Curve (RF)"
cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)

estimator = RandomForestClassifier(n_estimators=10, max_depth=3)
plot_learning_curve(estimator, title, X, y, ylim=(0.5, 1.01), cv=cv, n_jobs=4)

# plt.savefig('learningcurve.png', format='png', dpi=600)
plt.show()
import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(rf).fit(X_test, y_test)
dfs = eli5.formatters.as_dataframe.explain_weights_df(perm)
dfs.head()
# replacing feature numbers with names

features  = dfs.iloc[:,0]
features = features.values.tolist()

a = np.hstack(features)
a
feature = []
for n in range(0,27):
    string = str(a[n])
    cut = int(string[1:])
    feature.append(cut)
    
feature_list = []

for n in range(0,27):
    num = feature[n] +1                # adding 1 as first column is blueWins
    name = df.columns[num]
    feature_list.append(name)
    
dfs.insert(0, 'Features', feature_list)
dfs=dfs.drop('feature',axis=1)
dfs.head()
import shap
shap.initjs()
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)

#class 0 = Red win
#class 1 = Blue win
# shows correlation between blue having more gold and getting more kills

shap.dependence_plot("blueGoldDiff", shap_values[1], X, interaction_index="blueKills")
conf = []

for i in range(0,len(X_test)):
    x = rf.predict_proba([X_test.iloc[i]])
    y = tuple(x[0])
    conf.append(y)
sortedd = sorted(conf, key=lambda tup: tup[1])
rank = [i[0] for i in sortedd]
import matplotlib.pyplot  as plt
plt.figure(figsize=(5,4))
plt.xlabel('Ranked Test Samples')
plt.ylabel('Prediction Confidence')
plt.title('Confidence Ranking')
plt.plot(rank)
plt.show()
blue_wins = sorted(range(len(conf)),key = lambda k: conf[k])
red_wins = sorted(range(len(conf)),key = lambda k: conf[k], reverse = True)
top_blue = blue_wins[0:10]
top_red = red_wins[0:10]
print(top_blue)
print(top_red)
eli5.show_prediction(rf, X_test.iloc[55], show_feature_values=True)
eli5.show_prediction(rf, X_test.iloc[80], show_feature_values=True)
selected_columns = selected_columns[1:]
import statsmodels.api as sm
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns
SL = 0.05
data_modeled, selected_columns = backwardElimination(df.iloc[:,1:].values, df.iloc[:,0].values, SL, selected_columns)
result = pd.DataFrame()
result['blueWin'] = df.iloc[:,0]
data = pd.DataFrame(data = data_modeled, columns = selected_columns)
data
# remove these categories as predominatley value is 0
data1 = data.drop(['blueTowersDestroyed','redTowersDestroyed'], axis=1)
fig = plt.figure(figsize = (20, 25))
j = 0

for i in data1.columns:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(data1[i][result['blueWin']==0], color='r', label = 'Red Win')
    sns.distplot(data1[i][result['blueWin']==1], color='b', label = 'Blue Win')
    plt.legend(loc='best')
fig.suptitle('LoL Data Analysis')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
