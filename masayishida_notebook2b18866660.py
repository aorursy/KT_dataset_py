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
import gc
import numpy as np
import scipy as sp
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

import os
from sklearn.metrics import roc_auc_score,mean_squared_error,mean_squared_log_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import quantile_transform,StandardScaler,MinMaxScaler
from sklearn.decomposition import SparsePCA,TruncatedSVD
from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder
from tqdm import tqdm_notebook as tqdm

from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from lightgbm import LGBMRegressor


df_train = pd.read_csv('../input/exam-for-students20200923/train.csv')
df_test = pd.read_csv('../input/exam-for-students20200923/test.csv')
df_co_info = pd.read_csv('../input/exam-for-students20200923/country_info.csv')
df_co_info
df_co_info.loc[df_co_info['Region'].str.contains('WESTERN'),'Region']=1
df_co_info.loc[df_co_info['Region'] != 1,'Region'] = 0
df_train = pd.merge(df_train,df_co_info,on='Country',how='left')
df_test = pd.merge(df_test,df_co_info,on='Country',how='left')
df_train
df_test
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique() #ユニークな要素の個数、頻度（出現回数）をカウント
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

#histgram呼び出し関数
def checkhist(f,train,test,bns):
    plt.figure(figsize=[10,5])
    train[f].hist(density=True, alpha=0.5, bins=bns)
    # testデータに対する可視化を記入してみましょう
    test[f].hist(density=True, alpha=0.5, bins=bns)
    plt.xlabel(f)
    plt.ylabel('density')
    plt.show()
def hist_train_vs_test(feature,train,test,bins,clip = False):
    plt.figure(figsize=(16, 8))
    if clip:
        th_train = np.percentile(train[feature], 99)
        th_test = np.percentile(test[feature], 99)
        plt.hist(x=[train[train[feature]<th_train][feature], test[test[feature]<th_test][feature]])
    else:
        plt.hist(x=[train[feature], test[feature]])
    plt.legend(['train', 'test'])
    plt.show()
#ヒートマップ
def heatmap(y_data,x_data):
    fig, ax = plt.subplots(figsize=(12, 9)) 
    sns.heatmap(pd.concat([y_data,x_data], axis=1).corr(), square=True, vmax=1, vmin=-1, center=0)
#Xとyに分割
target = 'ConvertedSalary'
y_train = df_train[target] #yは貸し倒れフラグ
X_train = df_train.drop([target], axis=1) #xはそれ以外

X_test = df_test

# dtypeがobjectとnumericのものに分ける
cats = []
nums = []

for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype == 'datetime64[ns]' or X_train[col].dtype == 'bool':
        cats.append(col)
    else:
        nums.append(col)
nums
checkhist('Respondent',X_train,X_test,10)
X_train['Respondent']


cats
X_test[cats]
#惜しい
X_train['FormalEducation'].value_counts()
#惜しい
X_train['UndergradMajor'].value_counts()
#OEかな
X_train['CompanySize'].value_counts()
#カンマ区切りだが、text候補
X_train['DevType'].value_counts()
#OE
X_train['YearsCoding'].value_counts()

X_train['YearsCodingProf'].value_counts()
checkhist('YearsCodingProf',X_train,X_test,10)
X_train['JobSatisfaction'].value_counts()
X_train['CareerSatisfaction'].value_counts()
#選択肢は少ないけど、文章だからテキスト？
X_train['UpdateCV'].value_counts()
X_train['Currency'].value_counts()
X_test['Currency'].value_counts()
X_test['CurrencySymbol'].value_counts()
X_train['CommunicationTools'].value_counts()
X_train['TimeFullyProductive'].value_counts()
X_train.loc[X_train['TimeFullyProductive'].str.contains('Less',na=False),'TimeFullyProductive']=1 
X_train.loc[X_train['TimeFullyProductive'].str.contains('One',na=False),'TimeFullyProductive']=2
X_train.loc[X_train['TimeFullyProductive'].str.contains('to six',na=False),'TimeFullyProductive']=3
X_train.loc[X_train['TimeFullyProductive'].str.contains('Six to',na=False),'TimeFullyProductive']=4
X_train.loc[X_train['TimeFullyProductive'].str.contains('Nine months',na=False),'TimeFullyProductive']=5
X_train.loc[X_train['TimeFullyProductive'].str.contains('More',na=False),'TimeFullyProductive']=6
X_train['TimeFullyProductive'].fillna(0,inplace=True)
X_train['TimeFullyProductive'] = X_train['TimeFullyProductive'].astype(int)


X_test.loc[X_test['TimeFullyProductive'].str.contains('Less',na=False),'TimeFullyProductive']=1 
X_test.loc[X_test['TimeFullyProductive'].str.contains('One',na=False),'TimeFullyProductive']=2
X_test.loc[X_test['TimeFullyProductive'].str.contains('to six',na=False),'TimeFullyProductive']=3
X_test.loc[X_test['TimeFullyProductive'].str.contains('Six to',na=False),'TimeFullyProductive']=4
X_test.loc[X_test['TimeFullyProductive'].str.contains('Nine months',na=False),'TimeFullyProductive']=5
X_test.loc[X_test['TimeFullyProductive'].str.contains('More',na=False),'TimeFullyProductive']=6
X_test['TimeFullyProductive'].fillna(0,inplace=True)
X_test['TimeFullyProductive'] = X_test['TimeFullyProductive'].astype(int)
X_train['TimeAfterBootcamp'].value_counts()

checkhist('TimeAfterBootcamp',X_train,X_test,10)

X_train['CheckInCode'].value_counts()

X_train['AIDangerous'].value_counts()

X_train['AIInteresting'].value_counts()

X_train['AIResponsible'].value_counts()

X_train['StackOverflowParticipate'].value_counts()

X_train['StackOverflowDevStory'].value_counts()


#OE
X_train['Exercise'].value_counts()
#OE
X_train['EducationParents'].value_counts()
#Saw an online advertisement and then researched it (without clicking on the ad) 
#Clicked on an online advertisement
#Paid to access a website advertisement-free
#Stopped going to a website because of their advertising
X_train['AdsActions'].value_counts()
X_train['Country'].value_counts()
#trainにGermany、Franceはないので削除
X_test['Country'].value_counts()

#自分でEncodingしたものはcatsからDrop
cats_remove = ['TimeFullyProductive']
for l in cats_remove:
    cats.remove(l)
    nums.append(l)
#テキスト特徴量はcatsからDrop
txt= ['DevType','AdsActions','UpdateCV','CommunicationTools']
for l in txt:
    cats.remove(l)
remove_nums=['Respondent']
X_train = X_train.drop(remove_nums,axis=1)
X_test = X_test.drop(remove_nums,axis=1)
for i in remove_nums:
    try:
        nums.remove(i)
    except: pass
X_train[nums].fillna(-9999,inplace=True)
X_test[nums].fillna(-9999,inplace=True)
nums


remove_cats=['Country','YearsCoding','MilitaryUS','CurrencySymbol']
X_train = X_train.drop(remove_cats,axis=1)
X_test = X_test.drop(remove_cats,axis=1)
for i in remove_cats:
    try:
        cats.remove(i)
    except: pass    
#ordinal encoder
oe = OrdinalEncoder(cols=cats, return_df=False)
X_train[cats] = oe.fit_transform(X_train[cats])
X_test[cats] = oe.fit_transform(X_test[cats])
X_train[cats].fillna(-9999,inplace=True)
X_test[cats].fillna(-9999,inplace=True)

for l in txt:
    TXT_train = X_train[l].copy() # この方式で変数を指定しないと動かない
    TXT_test = X_test[l].copy()

    # 欠損値埋
    TXT_train.fillna('#', inplace=True)
    TXT_test.fillna('#', inplace=True)

    # TfidfVectorizer
    tfidf     = TfidfVectorizer(max_features=5, analyzer='word', ngram_range=(1, 2))  
    TXT_train = tfidf.fit_transform(TXT_train)
    TXT_test  = tfidf.transform(TXT_test)

    # svdする場合
    n_components = 2
    svd = TruncatedSVD(n_components=n_components)
    TXT_train = svd.fit_transform(TXT_train.toarray())
    TXT_test  = svd.transform(TXT_test.toarray()) #.todense()

    columns = []
    for i in range(0,n_components):
        columns.append(l+"_{}".format(i))
    TXT_train = pd.DataFrame(TXT_train,columns=columns,index=X_train.index)
    TXT_test  = pd.DataFrame(TXT_test,columns=columns,index=X_test.index)
    X_train   = pd.concat([X_train,TXT_train],axis=1)
    X_test    = pd.concat([X_test,TXT_test],axis=1)
X_train = X_train.drop(txt,axis=1)
X_test = X_test.drop(txt,axis=1)
X_train['Emp_GDP']=X_train['Employment'] * X_train['GDP ($ per capita)']
X_test['Emp_GDP']=X_test['Employment'] * X_test['GDP ($ per capita)']

#X_train['Emp_Last']=X_train['Employment'] * X_train['LastNewJob']
#X_test['Emp_Last']=X_test['Employment'] * X_test['LastNewJob']

X_train['Emp_Years']=X_train['Employment'] * X_train['YearsCodingProf']
X_test['Emp_Years']=X_test['Employment'] * X_test['YearsCodingProf']

#X_train['Emp_SalaryType']=X_train['Employment'] * X_train['SalaryType']
#X_test['Emp_SalaryType']=X_test['Employment'] * X_test['SalaryType']

#---
#X_train['GDP_LastNewJob']=X_train['GDP ($ per capita)'] * X_train['LastNewJob']
#X_test['GDP_LastNewJob']=X_test['GDP ($ per capita)'] * X_test['LastNewJob']

X_train['GDP_YearsCodingProf']=X_train['GDP ($ per capita)'] * X_train['YearsCodingProf']
X_test['GDP_YearsCodingProf']=X_test['GDP ($ per capita)'] * X_test['YearsCodingProf']

X_train['GDP_SalaryType']=X_train['GDP ($ per capita)'] * X_train['SalaryType']
X_test['GDP_SalaryType']=X_test['GDP ($ per capita)'] * X_test['SalaryType']

#---
#X_train['LastNewJob_YearsCodingProf']=X_train['LastNewJob'] * X_train['YearsCodingProf']
#X_test['LastNewJob_YearsCodingProf']=X_test['LastNewJob'] * X_test['YearsCodingProf']

#X_train['LastNewJob_SalaryType']=X_train['LastNewJob'] * X_train['SalaryType']
#X_test['LastNewJob_SalaryType']=X_test['LastNewJob'] * X_test['SalaryType']

#---
X_train['YearsCodingProf_SalaryType']= X_train['YearsCodingProf'] * X_train['SalaryType']
X_test['YearsCodingProf_SalaryType']= X_test['YearsCodingProf'] * X_test['SalaryType']


# 学習用と検証用に分割する
X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.05, random_state=71)
#まずはチューニングなし
scores = []

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)


clf = LGBMRegressor(boosting_type='gbdt', num_leaves=15, max_depth=- 1, learning_rate=0.1,
                        n_estimators=1000, subsample_for_bin=200000, objective=None, 
                        class_weight=None, min_split_gain=0.0, min_child_weight=0.001,
                        min_child_samples=30, subsample=0.8, subsample_freq=0, colsample_bytree=0.8,
                        reg_alpha=1, reg_lambda=1, random_state=None, n_jobs=- 1,loss_function = 'RMSE',eval_metric = 'RMSE')
%%time
clf.fit(X_train_, np.log1p(y_train_))            #RMSLEなので np.log1p(y_train_)にする
clf.booster_.feature_importance(importance_type='gain')
imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)
fig, ax = plt.subplots(figsize=(5, 8))
lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
th=2000

use_col = imp.index[imp.importance > th] #importance閾値を切って特長項目を厳選。
X_train_=X_train_[use_col]
X_train=X_train[use_col]
X_val=X_val[use_col]
X_test=X_test[use_col]
heatmap(y_train,X_train[use_col])
scores = []
lgb_y_pred_train = np.zeros(X_train.shape[0])
lgb_y_pred_test = np.zeros(X_test.shape[0])

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):
    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]
    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]
        
    clf = LGBMRegressor(boosting_type='gbdt', num_leaves=15, max_depth=- 1, learning_rate=0.1,
                        n_estimators=1000, subsample_for_bin=200000, objective=None, 
                        class_weight=None, min_split_gain=0.0, min_child_weight=0.001,
                        min_child_samples=30, subsample=0.8, subsample_freq=0, colsample_bytree=0.8,
                        reg_alpha=1, reg_lambda=1, random_state=None, n_jobs=- 1,loss_function = 'RMSE',eval_metric = 'RMSE')
    
    
    clf.fit(X_train_, np.log1p(y_train_))            #RMSLEなので np.log1p(y_train_)にする
    y_pred = np.expm1(clf.predict(X_val)) #RMSLEなので expm1にする
    lgb_y_pred_train[test_ix] = y_pred
    
    y_pred = np.where(y_pred<0,0,y_pred) #負の値は0にする
    score = mean_squared_log_error(y_val, y_pred)**0.5 #RMSLE
    scores.append(score)
    lgb_y_pred_test += np.expm1(clf.predict(X_test))
    
    print('CV Score of Fold_%d is %f' % (i, score))

lgb_y_pred_test /= 5
ave_scores=0
for l in scores:
    ave_scores += l/len(scores)
print('Average_score is %f' % ( score))
lgb_y_pred_test
submission = pd.read_csv('../input/exam-for-students20200923/sample_submission.csv', index_col=0)
submission.ConvertedSalary = lgb_y_pred_test
submission.to_csv('submission.csv')
submission
