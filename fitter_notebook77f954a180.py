import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
# %matplotlib qt5
%matplotlib inline
data_test = pd.read_csv('../input/give-me-some-credit-dataset/cs-test.csv',index_col=0)
data_test.info()
data_train = pd.read_csv('../input/give-me-some-credit-dataset/cs-training.csv',index_col=0)
col_replace = {'SeriousDlqin2yrs':'target', ## 违约客户及超过90天逾期客户，bool型；
            'RevolvingUtilizationOfUnsecuredLines':'percentage', ## 贷款以及信用卡可用额度与总额度比例，百分比；
           'NumberOfOpenCreditLinesAndLoans':'open_loan', ## 开放式信贷和贷款数量，开放式贷款（分期付款如汽车贷款或抵押贷款）和信贷（如信用卡）的数量，整型；
           'NumberOfTimes90DaysLate':'90-', ## 90天逾期次数：借款者有90天或更高逾期的次数，整型；
           'NumberRealEstateLoansOrLines':'estate_loan', ## 不动产贷款或额度数量：抵押贷款和不动产放款包括房屋净值信贷额度，整型；
           'NumberOfTime60-89DaysPastDueNotWorse':'60-89', ## 60-89天逾期但不糟糕次数，整型；
           'NumberOfDependents':'Dependents', ## 家属数量：不包括本人在内的家属数量，整型；
           'NumberOfTime30-59DaysPastDueNotWorse':'30-59' ## 35-59天逾期但不糟糕次数，整型；
              }
data_train.rename(columns=col_replace,inplace=True)
data_train.head()
data_train.info() ## 说明只有两项有缺失值
data_train.duplicated().sum()
data_train.drop_duplicates(inplace=True)
data_train.loc[data_train.MonthlyIncome.isna(),'Dependents'].isna().mean()
data_train.loc[data_train.Dependents.isna(),'MonthlyIncome'].isna().mean() ## 结果1，说明家属没填的，月收入都没填。
data_train.dropna(subset=['Dependents'],inplace=True) ## 去掉两项都缺的项。
## 考察其他项有什么特征
plt.figure(figsize=(16,9))
for i,col in enumerate(data_train.columns):
    plt.subplot(3,4,i+1)
    data_train[col].hist(bins=50)
    plt.ylabel(col)
## 填充缺失值
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import KNNImputer
miss_imputer = IterativeImputer(ExtraTreesRegressor(10),initial_strategy="most_frequent",max_iter=5)
# miss_imputer2 = KNNImputer()
data_train_bak = data_train.copy()
data_train_bak.iloc[:,:-1]=miss_imputer.fit_transform(X=data_train.iloc[:,:-1])

plt.subplot(121)
plt.hist(data_train_bak.loc[data_train_bak.MonthlyIncome<16000,'MonthlyIncome'])
plt.subplot(122)
plt.hist(data_train.loc[data_train.MonthlyIncome<16000,'MonthlyIncome'])
## 填充结果改变了数据分布，不采用。
## 有缺失值的两列均改用未缺失项随机抽样填充
fill_len = data_train.MonthlyIncome.isna().sum()
data_train.loc[data_train.MonthlyIncome.isna(),'MonthlyIncome']=\
np.random.choice(data_train.loc[data_train.MonthlyIncome.notna(),'MonthlyIncome'],fill_len,False)
data_train.info()
plt.figure(figsize=(16,9))
sns.heatmap(data_train.corr(method='spearman'),annot=True)
## 无>0.5的相关性, 暂不考虑降维
## 特征工程前，重置索引
data_train.reset_index(drop=True, inplace=True)
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(data_train,stratify=data_train.target)
import scipy

def init_box(df,n_split):
    ## 待份变量等频初始化分箱，统计分箱后标签频数
    categ = pd.cut(df.iloc[:,0],n_split,duplicates='raise')
    freq = pd.crosstab(categ, df.iloc[:,1]).reset_index()
    freq.iloc[:,0] = freq.iloc[:,0].astype(object)

    ## 计算WOE,IV
    freq['woe'] = np.log(freq[1]/freq[0]*freq[0].sum()/freq[1].sum())
    iv = freq.woe.dot((freq[1]/freq[1].sum()-freq[0]/freq[0].sum()))
    
    return freq.set_index(freq.columns[0]), iv ## reindex方便后续计算
def mychi2(freq):
    R = freq.sum(1,keepdims=True)
    C = freq.sum(0)
    N = R.sum()
    E = R*C / N
    sqr = np.square(freq-E)/E
    sqr[E==0] = 0
    return sqr.sum()
chi2 = lambda x: scipy.stats.chi2_contingency(x,correction=False)[1]
add = lambda in1,in2: pd.Interval(in1.left,in2.right)
def chi2_box(df, n_split=20, p_thr=.05):
    '''
    df:两列，第一列是待分箱变量，第二列是标签
    n_split: 初始划分区间数
    p_thr: 拒绝概率
    '''
    ## 待份变量等频初始化分箱，统计分箱后标签频数
    categ = pd.qcut(df.iloc[:,0],n_split,duplicates='drop')
    freq = pd.crosstab(categ, df.iloc[:,1]).reset_index()
    freq.iloc[:,0] = freq.iloc[:,0].astype(object)
    
    ## 相邻箱计算p值
    while True:
        p_val = [chi2(freq.iloc[i:i+2,1:]) for i in range(len(freq))]
        p_max = np.argmax(p_val[:-1])
        if p_val[p_max] <= p_thr:
            break
        freq.iloc[p_max,0] = add(freq.iloc[p_max,0], freq.iloc[p_max+1,0])
        freq.iloc[p_max,1:] = freq.iloc[p_max,1:] + freq.iloc[p_max+1,1:]
        freq.drop(index=freq.index[p_max+1],inplace=True)
    freq.iloc[-1,0] = add(freq.iloc[-1,0], pd.Interval(0,np.inf)) ## 适配数据
    freq.iloc[0,0] = add(pd.Interval(-np.inf,0), freq.iloc[0,0]) ## 适配数据

    ## 计算WOE,IV
    freq['p_val'] = p_val
    freq['woe'] = np.log(freq[1]/freq[0]*freq[0].sum()/freq[1].sum())
    iv = freq.woe.dot((freq[1]/freq[1].sum()-freq[0]/freq[0].sum()))
    
    return freq.set_index(freq.columns[0]),iv
    
## test function 

num = 10000  ##构造一个有40000数据量的数据
x1 = np.random.randint(1,10,num)
x2 = np.random.randint(10,30,num)
x3 = np.random.randint(30,45,num)
x4 = np.random.randint(45,80,num)
x = np.r_[x1,x2,x3,x4]

y1 = np.random.choice([0,1],num,p=[.9,.1])
y2 = np.random.choice([0,1],num,p=[.7,.3])
y3 = np.random.choice([0,1],num,p=[.5,.5])
y4 = np.random.choice([0,1],num,p=[.3,.7])
y = np.r_[y1,y2,y3,y4]

testdata = pd.DataFrame({"age":x,"y":y})
testdata.groupby('age')['y'].mean().plot()
chi2_box(testdata,100,.001)
spearman = lambda x: scipy.stats.spearmanr(x.iloc[:,0], x.iloc[:,1])[0]
def spearman_box(df, n = 20, r_thr=1):
    r = 0
    while np.abs(r) < r_thr:
        categ = pd.qcut(df.iloc[:,0], n, duplicates='drop')
        d2 = df.groupby(categ).mean()
        r = spearman(d2)
        n = n - 1
    freq = pd.crosstab(categ, df.iloc[:,1]).reset_index()
    freq['woe'] = np.log(freq[1]/freq[0]*freq[0].sum()/freq[1].sum())
    iv = freq.woe.dot((freq[1]/freq[1].sum()-freq[0]/freq[0].sum()))
    return freq.set_index(freq.columns[0]),iv
age_box,age_iv = chi2_box(df_train[['age','target']],50,.001)
display(age_box)
_ = age_box.woe.plot(kind='bar',title=f'IV: {age_iv:.4f}')
MI_box,MI_iv = chi2_box(df_train[['MonthlyIncome','target']],20,.01)
display(MI_box,MI_box.woe.plot(kind='bar',title=f'IV: {MI_iv:.4f}'))
per_box,per_iv = chi2_box(df_train[['percentage','target']],20,.01)
display(per_box,per_iv,per_box.woe.plot(kind='bar',title=f'IV: {per_iv:.4f}'))
ax1 = plt.subplot(122)
df_train.loc[df_train['30-59'].lt(20),['30-59']].hist(ax=ax1,bins=50)
ax2=plt.subplot(121)
df_train[['30-59']].boxplot(ax=ax2)
n30_box,n30_iv = chi2_box(df_train[['30-59','target']],20,.05)
display(n30_box,n30_iv) ## 每箱数量差别太大，改手动分箱。
split = [-np.inf,.1,*np.arange(1.1,5,1),np.inf]
n30_box,n30_iv = init_box(df_train[['30-59','target']],split)
display(n30_box,n30_box.woe.plot(kind='bar',title=f'IV: {n30_iv:.4f}'))
ax1 = plt.subplot(122)
df_train.loc[df_train['60-89'].lt(20),['60-89']].hist(ax=ax1,bins=50)
ax2=plt.subplot(121)
df_train[['60-89']].boxplot(ax=ax2)
n60_box,n60_iv = chi2_box(data_train[['60-89','target']],20,.05)
display(n60_box,n60_iv) ## 每箱数量差别太大，改手动分箱。
split = [-np.inf,.1,1.1, 2.1, 3.1,  5.1,np.inf]
n60_box,n60_iv = init_box(df_train[['60-89','target']],split)
display(n60_box,n60_box.woe.plot(kind='bar',title=f'IV: {n60_iv:.4f}'))
ax1 = plt.subplot(122)
df_train.loc[df_train['90-'].lt(20),['90-']].hist(ax=ax1,bins=50)
ax2=plt.subplot(121)
df_train[['90-']].boxplot(ax=ax2)
n90_box,n90_iv = chi2_box(data_train.loc[data_train.MonthlyIncome>1,['90-','target']],20,.05)
display(n90_box,n90_iv) ## 每箱数量差别太大，改手动分箱。
split = [-np.inf,.1,*np.arange(1.1,4,1),np.inf]
n90_box,n90_iv = init_box(df_train[['90-','target']],split)
display(n90_box,n90_box.woe.plot(kind='bar',title=f'IV: {n60_iv:.4f}')) ## 每箱数量差别太大，改手动分箱。
debt_box,debt_iv = chi2_box(df_train.loc[:,['DebtRatio','target']],25,.001)
display(debt_box)
debt_box.woe.plot(kind='bar',title=f'IV: {debt_iv:.4f}') ## 效果差，用手动分箱
ax1 = plt.subplot(211)
data_train.query('6>DebtRatio')[['DebtRatio']].plot(kind='hist',ax=ax1, bins=500,figsize=(8,9))
## 0.04之前有异常，单独分一个箱
ax2 = plt.subplot(212)
data_train.query('6<=DebtRatio')[['DebtRatio']].plot(kind='hist',ax=ax2, bins=500,figsize=(8,9))
## 后面是一个长尾分布
## 尝试多种分箱方法后的结果：
split = [-np.inf,.08,*np.arange(.2,1.1,.1),]
split2 = df_train.DebtRatio[data_train.DebtRatio.gt(2)].quantile(np.arange(10)/10).tolist()
b=split+split2+[np.inf]
debt_box,debt_iv = init_box(df_train.loc[:,['DebtRatio','target']],b)
# debt_box,debt_iv = init_box(data_train.loc[:,['DebtRatio','target']],b)
# debt_box,debt_iv = spearman_box(data_train.loc[:,['DebtRatio','target']],r_thr=.5)
display(debt_box)
debt_box.woe.plot(kind='bar',title=f'IV: {debt_iv:.4f}')
data_train.open_loan.hist(bins=500)
# loan_box,loan_iv = init_box(data_train.loc[:,['open_loan','target']],np.arange(15))
# loan_box,loan_iv = spearman_box(data_train.loc[:,['open_loan','target']],r_thr=.1)
loan_box,loan_iv = chi2_box(df_train.loc[:,['open_loan','target']])
display(loan_box)
loan_box.woe.plot(kind='bar',title=f'IV: {loan_iv:.4f}')
df_train.query('estate_loan<6').estate_loan.hist(bins=15)
estate_box,estate_iv = init_box(df_train[['estate_loan','target']],
                               n_split = [-np.inf, 0, 1, 2, 3, 5, np.inf])
# estate_box,estate_iv = spearman_box(data_train[['estate_loan','target']],r_thr=.05)
display(estate_box,estate_iv)
estate_box.woe.plot(kind='bar',title=f'IV: {estate_iv:.4f}')
df_train.Dependents.hist(bins=500)
depend_box,depend_iv = init_box(df_train[['Dependents','target']],
                               n_split = [-np.inf, 0, 1, 2, 3, 5, np.inf])
# depend_box,depend_iv = chi2_box(data_train[['Dependents','target']])
display(depend_box)
depend_box.woe.plot(kind='bar',title=f'IV: {depend_iv:.4f}')
woe_map = { 'percentage':per_box,
            'age':age_box,
            '30-59':n30_box,
            'DebtRatio':debt_box,
            'MonthlyIncome':MI_box,
            'open_loan':loan_box,
            '90-':n90_box,
            'estate_loan':estate_box,
            '60-89':n60_box,
            'Dependents':depend_box}
def woe_swag(df,woe=woe_map):
    df2={k:v.loc[df[k],'woe'].reset_index(drop=True) for k,v in woe.items()}
    df2['target'] = df.target.reset_index(drop=True)
    return pd.DataFrame(df2)
df_train = woe_swag(df_train)
df_train.head()
df_test = woe_swag(df_test)
## 将正负样本SMOTE
# X, y = df_train.iloc[:,:-1].values, df_train.iloc[:,-1:].values
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=.5,n_jobs=-1)
X, y = smote.fit_resample(df_train.iloc[:,:-1], df_train.iloc[:,-1])
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve
my_scorer = make_scorer(roc_auc_score,needs_proba=True)
from sklearn.metrics import plot_roc_curve
params = {'C':[.0005,.001,.01,.015,.02]}
lr = LogisticRegression(solver='liblinear')
clf = GridSearchCV(lr,params,n_jobs=-1,scoring=my_scorer)
clf.fit(X,y)
display(clf.best_score_, clf.best_params_)
## 模型评估--auc
_ = plot_roc_curve(clf.best_estimator_,df_test.iloc[:,:-1],df_test.target.values)
roc_auc_score(df_test.target.values, clf.best_estimator_.predict_proba(df_test.iloc[:,:-1])[:,1])
from xgboost import XGBClassifier
xgb = XGBClassifier(n_jobs=-1)
xgb.fit(X,y)
_ = plot_roc_curve(xgb,df_test.iloc[:,:-1],df_test.target.values)
roc_auc_score(df_test.target.values, xgb.predict_proba(df_test.iloc[:,:-1])[:,1])
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
rf = RandomForestClassifier()
params2 = {'max_depth':[3,5,7],'min_samples_leaf':[1,10,20]}
clf2 = GridSearchCV(rf,params2,scoring=my_scorer,n_jobs=-1)
clf2.fit(X,y)
print(clf2.best_params_,clf2.best_score_)
## 模型评估--auc
_ = plot_roc_curve(clf2.best_estimator_,df_test.iloc[:,:-1],df_test.target.values)
roc_auc_score(df_test.target.values, clf2.best_estimator_.predict_proba(df_test.iloc[:,:-1])[:,1])
clf_t = VotingClassifier([('lr',clf.best_estimator_),
                          ('xgb',xgb),
                          ('rf',clf2.best_estimator_)],voting='soft',weights=[3.5,4.5,2])
clf_t.fit(X,y)
## 模型评估--auc
_ = plot_roc_curve(clf_t,df_test.iloc[:,:-1],df_test.target.values)
roc_auc_score(df_test.target.values, clf_t.predict_proba(df_test.iloc[:,:-1])[:,1])
data_test.rename(columns=col_replace,inplace=True)
## 准备测试集
data_test.isna().sum()
fill_len = data_test.Dependents.isna().sum()
data_test.loc[data_test.Dependents.isna(),'Dependents']=\
np.random.choice(data_train.loc[:,'Dependents'],fill_len,False)
fill_len = data_test.MonthlyIncome.isna().sum()
data_test.loc[data_test.MonthlyIncome.isna(),'MonthlyIncome']=\
np.random.choice(data_train.loc[:,'MonthlyIncome'],fill_len,False)
data_test2 = woe_swag(data_test)
data_test2.info()
y_hat = clf_t.predict_proba(data_test2.iloc[:,:-1])[:,1]
to_push = pd.DataFrame({'Id':data_test.index.tolist(),
                        'Probability':y_hat})
to_push.to_csv('./submission.csv', index=False)
to_push.head()
def PSI(model, X_train,X_test):
    '''
    训练集样本分箱频率统计与待预测样本分箱频率统计的KL散度求和
    '''
    bins = np.arange(11)/10.0
    
    y_train = model.predict_proba(X_train)[:,1]
    train_count = pd.value_counts(y_train,sort=False,bins=bins,normalize=True)
    
    y_test = model.predict_proba(X_test)[:,1]
    test_count = pd.value_counts(y_test,sort=False,bins=bins,normalize=True)
    
    return (train_count-test_count).dot(np.log(train_count/test_count))
psi=PSI(clf_t, df_train.iloc[:,:-1], data_test2.iloc[:,:-1])*100
print(f'PSI= {psi:.4f}%, 低于10%, 模型稳定。')
## 确定阈值--KS检验
def KS(model, X, y):
    y_prob = model.predict_proba(X)[:,1]
    df = pd.DataFrame(roc_curve(y, y_prob),index=['fpr','tpr','thre']).T
    df['gap'] = df.tpr - df.fpr
    ks = df.gap.max()
    fpr_max, tpr_max, thre, _  = df.loc[df.gap.idxmax(),:]
    
    df.plot(x='thre',y=['fpr','tpr'],xlim=(0,1),
           title=f'KS: {ks:.4f}, thre: {thre:.4f}')
    plt.vlines(thre,fpr_max,tpr_max,color='r')
    
    return thre
thre = KS(clf_t, X, y)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit([[thre], [1]],[60,100])
print(f'y = {lr.coef_[0]:.4f} * X + {lr.intercept_:.4f}')
