import pandas as pd

#import swifter

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#pd.set_option('display.max_rows', 500)

#pd.set_option('display.max_columns', 500)

##pd.set_option('display.width', 1000)



from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.ensemble import BaggingClassifier

from sklearn.svm import SVC as svc

from sklearn.linear_model import LogisticRegression







import warnings

warnings.filterwarnings("ignore")
! ls ../input/bank-loan-data/
data_borrower = pd.read_csv('../input/bank-loan-data/borrower_table.csv')
data_borrower
data_borrower.shape
data_borrower.info(verbose=True)
data_borrower.describe()
data_borrower.isna().any()
data_borrower.isna().sum()
data_borrower.isna().sum()/len(data_borrower)
data_loan = pd.read_csv('../input/bank-loan-data/loan_table.csv',parse_dates=['date'])
data_loan
data_loan.shape
len(data_loan)==len(data_borrower)
data_loan.describe()
data_loan.info(verbose=True)
data_loan.isna().any()
data_loan.isna().sum()/len(data_loan)
data_loan.loan_id.nunique()
data_borrower.loan_id.nunique()
# great let's merge
df = data_borrower.merge(data_loan,how='outer',on='loan_id')
len(df)
df.describe()
df.info(verbose=True)
# create target variable
df['target']=0
df.loc[(df['loan_granted']==1) & (df['loan_repaid']==1),'target']=1
df.loc[(df['loan_granted']==1) & (df['loan_repaid']==0),'target']=-1
df.filter(regex='loan_granted|loan_repaid|target')
# too unique. will just cause overfitting



df.drop(['loan_id'],1,inplace=True)
df.columns.values
# renaming columns

df = df.rename(columns={

    "fully_repaid_previous_loans": "repaid_prv_loans", 

    "currently_repaying_other_loans": "repaying_curr_loans",

    "avg_percentage_credit_card_limit_used_last_year": 'cc_limit_used_last_year'})
# cat features: 

feat_cat = ['target', 'loan_purpose']



# binary features:

feat_bin =  ['is_first_loan', 'repaid_prv_loans', 'repaying_curr_loans', 'is_employed']



# num features 

feat_num = ['total_credit_card_limit', 'cc_limit_used_last_year', 'saving_amount', 

            'checking_amount', 'yearly_salary','age', 'dependent_number']



# datetime type

feat_date = ['date']
# checking distribution of categorical features 
for col in feat_cat+feat_bin:

    print(df[col].value_counts(normalize=True,dropna =False, sort=True))

    print()

    print()
# most loans were of type 0, then 1, the -1

# most were of home or business . but the loan_purpose distribution is somewhat similar

# loan_purpose had similar ratios as well

# is first loan also had similar ratios 

# half missing if repaid prev loans, the other half  did repay the loans, very few didn't

# half missing if repaying current loans, those that have data, so 30% doesn't have current loans and 17% do

# most borrowers are employed 



# note: when saying similary distributed,

# i mean give or take. differences aren't too significant relativley

# handling missing continuous values 
df.isnull().sum()/len(df)
# avg_percentage_credit_card_limit_used_last_year only has 7% missing. it's low enough not to need to drop anything, but

# rather can impute those missing values 
df['cc_limit_used_last_year'].describe()
# interesting note that the max is above one. 

# max          1.090000



# assuming this is valid data and therefore not dropping vals > 1 for cc_limit_used_last_year
df['cc_limit_used_last_year'].median()
# the mean and median are close. will use median
df.loc[df['cc_limit_used_last_year'].isna(),'cc_limit_used_last_year'] = df['cc_limit_used_last_year'].median()
# viewing the distribution of categorical and binary features per target value 
for feat in feat_cat+feat_bin:

    if feat == 'target':

        continue

    

    g = sns.FacetGrid(df,col='target',row=feat)

    g.map(sns.countplot,feat)
#most business, investment, and home loans were 0, then 1, the -1.

#since the since the loan_purpose was  similarly distributed and the facet grid was similar to the target's disribution(0 then 1 then -1), it seems like business, investment, and home loans didn't much affect the target

#

#other loans and emgergency had equal disributions for target -1 and target 1, and the most for target 0. seems like this had some affect to the target

#

#similar to the business loans, most first loan was 0, then 1 and -1. and first loan had similar distributions. seems like they didn't much affect the target either

#

#same for repaid prv loans  

#

#not repaying_curr loans was similar for target 0 and 1, and lower for target -1.  

#yes repaying current loans was vey low for target 1 and higher for target -1.

#seems like this affects the target well

#

#not employed is very high for target 0 and greater for target -1 then 1

#yes employed is greatest for target 1, then 0, then -1. this seems like a strong feature

#
# graph discrete + bin features based on target sum
# It is also important to keep in mind that a bar plot by default shows only the mean (or other estimator) value

# that's why also i have it showing the total sum and total count (sum/count=mean)

# also to show distribution of values at each level, show boxplot and violin plot 





def create_barplot(x, y, df, estimator, fill_na='**MISSING**', figure_size=(12, 8)):

    plt.figure(figsize=figure_size)

    graph_type = ''



    if estimator is 'sum':

        bp = sns.barplot(x=x, y=y, data=df.fillna(fill_na), estimator=np.sum)

        graph_type = estimator

    elif estimator is 'mean':

        bp = sns.barplot(x=x, y=y, data=df.fillna(fill_na))

        graph_type = estimator

    elif estimator is 'count':

        # will use count plot for len (distribution count)

        graph_type = estimator

        bp = sns.countplot(x=x,data=df.fillna(fill_na))

    else:

        bp = sns.barplot(x=x, y=y, data=df.fillna(fill_na), estimator=estimator)

        graph_type = estimator.__name__



    if estimator is 'count':

        title = 'countplot: ' + x 

    else:

        bp.set_xticklabels(bp.get_xticklabels(), rotation=90)

        title = 'barplot: ' + graph_type + ' ' + x + ' vs ' + y

        

    bp.set_title(title)



    return title





def create_violinplot(x=None, y=None, df=None, fill_na=-1, figure_size=(12, 8)):

    plt.figure(figsize=figure_size)

    vp = sns.violinplot(x=x, y=y, data=df)

    plt.xticks(rotation=45)

    # vp.set_xticklabels(vp.get_xticklabels(),rotation=30)

    title = 'violinplot: ' + str(x) + ' vs ' + str(y)

    vp.set_title(title)



    return title







def analyze_discrete_distributions(df,cols,target_col):

    figure_size = (6, 4)



    for col in cols:

        if col == target_col:

            continue



        print('Column: ' + col)

        title = create_barplot(col, target_col, df, 'sum', figure_size=figure_size)

        plt.show()



        title = create_barplot(col, target_col, df, 'mean', figure_size=figure_size)

        plt.show()



        title = create_barplot(col, target_col, df, 'count', figure_size=figure_size)

        plt.show()



        title = create_violinplot(col, target_col, df, figure_size=figure_size)

        plt.show()



        print()

        print()
analyze_discrete_distributions(df,feat_cat+feat_bin,'target')
#with loan purpose similarly distributed. able to see business and investment and home loans had the highest scores, other was low but still postive, emergency funds had the target sum in the minus. this makes sense since businesses and invesments seem to have more money to return loan. whereas emergency funds are usually taken when money is tight

#

#first loan didn't seem to variate much 

#

#repaid prev loans - similar to distribution. 

#missing  was the highest (even on average). it means the first time loan receivers were the most to return the loan. then came those that previously received the loan and were successful to return it. the least was for those who didn't previously return their loan when they had it. makes logical sense 

#

#those that were repaying current loans, had their target sum in the minus. this makes sense as they prob didn't have enough money to return both loans. however on the contrary, those who are not currently repaying other loans and those with NA (first time borrowers) were in the positive

#

#those that are not employed had their target in minus, while those that are employed had it in the plus. this is logical 
def analyze_continuous_distributions(df,cols,target_col,log_scale=False):



    for col in cols:

        

        print(col)

        print()

        

        title = 'Hist FacetGrid:'+col+'_vs_'+target_col

        g = sns.FacetGrid(df, col=target_col) #creats N graphs based off the amount of unique labels in COL

        g.map(plt.hist, col, bins=50,log=log_scale) #apply plotting function to each facet

        plt.show()



        title = 'Dist Plot: '+ col

        #dp = sns.distplot(df[col].fillna(-100), kde=False,norm_hist =False,rug=True)

        dp = sns.distplot(df[col], kde=False,norm_hist =False,rug=True)

        dp.set_title(title)

        plt.show()

        

        title = 'Boxplot FacetGrid:'+col+'_vs_'+target_col

        g = sns.FacetGrid(df, col=target_col) #creats N graphs based off the amount of unique labels in COL

        g.map(sns.boxplot, col,notch=True) #apply plotting function to each facet

        plt.show()

        

        title = 'Box Plot: '+ col 

        sns.boxplot(col,data=df,notch=True)

        plt.title(title)

        plt.show()

        

        print()

        print()
analyze_continuous_distributions(df,feat_num,'target',log_scale=False)
#those with target 1 had greater cc limits. that makes sense since a higher cc limit prob means he has higher income and can pay the loan

#most cc limits of 0 was in target 0, some in target -1 and very few in target 1

#as the target increases, so does the cc limit . this makes sense

#

#cc limit used last year was bascially similar for target -1 and 0. but less for target 1.

#this make sense since if have didn't use much of the cc limit previous year, have more moeny to pay loan

#

#checking and saving amount is very low for -1, and increases as target increases. this also makes sense, the more money you have the more likley u can return the loan 

#

#yearly sallary also increases as targte increases. same logic as before.

#interesting note:most 0 values are found in target 0, some in target -1, and very few in target 1.

#

#ages don't seem to contribute much to the target, as they seem to be distributed the same, just different porportions due to the sample sizes of the target. it makes sense to see age distribute normally 

#

#dependent number seems to have some affect on the target. the mean is greater in target -1, and target 0 and 1 have similar means . seems like the more dependents they had, the harder it was to return the loan. make sense

#
# is there a value in the datetime?

# let's unravel datetime and see 
def unravel_datetime_column(df, col,drop_cols=False):

    df[col + '_day'] = df[col].apply(lambda x: x.day if x else None)

    df[col + '_dayofweek'] = df[col].apply(lambda x: x.dayofweek if x else None)

    df[col + '_month'] = df[col].apply(lambda x: x.month if x else None)

    df[col + '_year'] = df[col].apply(lambda x: x.year if x else None)



    if drop_cols:

        df.drop([col], axis=1,inplace=True)



    return df
for col in feat_date:

    print(col)

    unravel_datetime_column(df,col,drop_cols=False)
for col in df.filter(regex='date_').columns:

    print(col)

    print(df[col].value_counts(dropna=False))

    print()

    print()
df.drop('date_year',1,inplace=True)
for col in df.filter(regex='date_').columns:

    print(col)

    print(df[col].value_counts(dropna=False))

    print()

    print()
date_month_corr=pd.get_dummies(df.filter(regex='date_month|target').astype(str)).corr()

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(date_month_corr,square = False,annot =True,cmap='Spectral',ax=ax)

plt.show()
date_day_corr=pd.get_dummies(df.filter(regex='date_day|target').astype(str)).corr()

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(date_day_corr,square = False,annot =True,cmap='Spectral',ax=ax)

plt.show()
date_dayofweek_corr=pd.get_dummies(df.filter(regex='date_dayofweek|target').astype(str)).corr()

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(date_dayofweek_corr,square = False,annot =True,cmap='Spectral',ax=ax)

plt.show()
# as i expected, date doesn't serve much purpose. correlations are very low for all the unraveled datetime elements

#dropping all date features 
df.drop(df.filter(like='date').columns.values,1,inplace=True)

del feat_date
# are the saving and checking amounts highly (pos/neg) correlated?
df.filter(like='amount').corr()
# not such a high corr, will keep both features for now 
# engineering features, also to handle missing values
df.isna().any()
df.repaid_prv_loans = df.repaid_prv_loans.replace({1.0: 'repaid_prv_loans', 0.0: 'not_repay_prv_loans', np.NaN : 'first_time_loan'})

df.repaying_curr_loans = df.repaying_curr_loans.replace({1.0: 'repaying_curr_loans', 0.0: 'not_repaying_curr_loans', np.NaN : 'first_time_loan'})

df.drop('is_first_loan',1,inplace=True)
# will get_dummies in 2 parts

df = pd.get_dummies(df,prefix='',prefix_sep='',columns=['repaid_prv_loans','repaying_curr_loans'],drop_first=False)
df.drop('first_time_loan',1,inplace=True)
df.isnull().any()
# have null from loan_repaid, will anyways drop loan_repaid and loan_granted since target derived from it 



# prevents dataleakage 

df.drop(['loan_granted','loan_repaid'],1,inplace=True)
# adding new features

df['zero_cc_limit']=0

df.loc[df['total_credit_card_limit']==0,'zero_cc_limit']=1



df['zero_yearly_salary']=0  

df.loc[df['yearly_salary']==0,'zero_yearly_salary']=1
corr_mat = pd.get_dummies(df).corr()

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(corr_mat.round(decimals=2),square = False,annot =True,cmap='Spectral',ax=ax)

plt.show()
# zero_yearly_salary is perfectly inversely correlated to is_employed

# # seems like all those zeros for yearly salary are for is employed is false.
len(df[(df.is_employed==0) & (df.yearly_salary>0)])
# yep i was right. dropping

df.drop('zero_yearly_salary',1,inplace=True)
corr_mat = pd.get_dummies(df).corr()

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(corr_mat.round(decimals=2),square = False,annot =True,cmap='Spectral',ax=ax)

plt.show()
#regarding target + feature pair correlation,a lot of what saw before is seen here as well:

#    

#cc limit positively correlates with target. with it 

#increases so does target increase from -1 to 1. similar to what saw

#

#cc limit used last year is neg corr to target. similar to what we saw

#

#saving amount,checking amount, is employed,yearly salary, 

#

#dependent number is negatively correlated, but very weak. like saw before

#

#repaying current loans is also neg corr

#

#and emgergency loans are neg corr 

#

#age isn't correlated like saw before
#some other interesting correlating pairs:

#

#yearly salary is highly correlated to is employed. this makes logical sense

#

#not repaying current loans is highly correlated with repaid previous loans 

#

#repaying current loans is also correlated with repaid preivous loans
#to check:

#    

#features which do not contribute to other features, including target.

#would be interesting to check if it will contribute positivley to model

#

#age, loan_purpose_home, loan_purpose_investment, loan_purpose_other

#

#repaid_prv_loans only correlate to features it was created from. nothing to target 

#

#zero_cc_limit . not much significance to other 

#



# dummy variable trap

df.drop('not_repay_prv_loans',1,inplace=True)

df.drop('not_repaying_curr_loans',1,inplace=True)
df=pd.get_dummies(df,columns=['loan_purpose'],drop_first=True)
df_orig = df.copy(deep=True)
# not going to scale now as xgboost is indifferent to monotonous transformations 
# models 
scoring=['precision_macro', 'recall_macro', 'balanced_accuracy']





#scoring = {

    #'accuracy':'balanced_accuracy_score',

    #'confusion_matrix':'confusion_matrix',

    #'precision':'precision',

    #'recall':'recall',

    #'roc_auc':'roc_auc',

    ##'my_scorer':make_scorer(my_scorer, greater_is_better=True)

#}
def my_scorer(y_actual, y_pred):

    actual = np.sum(y_actual)

    print('actual:',actual)

    

    predicted = np.sum(y_pred)

    print('predicted:',predicted)

    

    diff = np.abs(actual-predicted) # assuming there is an equal cost for FP and FN

    print('diff',diff)

    

    return diff 
def get_best_model(models):

    min_val = models[min(models, key=lambda k: models[k][1])][1]

    for k in models:

        if models[k][1] == min_val:

            return k,models[k][0]
models={}
#feature selection using XGB 
# all features of df
X = df.drop('target',1)

y = df['target']



X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)
## finding best hyper params

#

#xgboost_params = {"max_depth" : list(range(3,10,1)), "learning_rate" : [.5,0.1,0.05,]}

#reg_xgb_grid = GridSearchCV(xgb.XGBClassifier(num_classes=3), xgboost_params,n_jobs=-1,cv=5,verbose=2)

#reg_xgb_grid.fit(X_train, y_train)

#reg_xgb=reg_xgb_grid.best_estimator_

#reg_xgb
#print('training accuracy',reg_xgb_grid.best_score_.round(2))
'''

best params:

max_depth=7

learning_rate=0.05

'''
reg_xgb = xgb.XGBClassifier(max_depth=7,learning_rate=0.05, num_class=3)
reg_xgb.fit(X_train,y_train)
y_pred = cross_val_predict(reg_xgb, X_train, y_train, cv=5,n_jobs= -1)

model_score = my_scorer(y_train,y_pred)
models['xgb_all']=[reg_xgb,model_score]
reg_xgb.feature_importances_

xgb.plot_importance(reg_xgb)
# testing to  dropping poor correlation features 
X = df.drop(['target','age','loan_purpose_home','loan_purpose_investment','loan_purpose_other','repaid_prv_loans','zero_cc_limit',],1)

y = df['target']



X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)
## finding best hyper params

#

#xgboost_params = {"max_depth" : list(range(3,10,1)), "learning_rate" : [.5,0.1,0.05,]}

#reg_xgb_grid = GridSearchCV(xgb.XGBClassifier(num_classes=3), xgboost_params,n_jobs=-1,cv=5,verbose=2)

#reg_xgb_grid.fit(X_train, y_train)

#reg_xgb=reg_xgb_grid.best_estimator_

#reg_xgb
#print('training accuracy',reg_xgb_grid.best_score_.round(2))
'''

best params:

max_depth=6

learning_rate=0.1

'''
reg_xgb = xgb.XGBClassifier(max_depth=6,num_class=3)
reg_xgb.fit(X_train,y_train)
y_pred = cross_val_predict(reg_xgb, X_train, y_train, cv=5,n_jobs= -1)

model_score = my_scorer(y_train,y_pred)
models['xgb_no_bad_corr']=[reg_xgb,model_score]
reg_xgb.feature_importances_

xgb.plot_importance(reg_xgb)
# testing to drop is employed as well as bad corr
X = df.drop(['target','age','loan_purpose_home','loan_purpose_investment','loan_purpose_other','repaid_prv_loans','zero_cc_limit','is_employed',],1)

y = df['target']



X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)
## finding best hyper params

#

#xgboost_params = {"max_depth" : list(range(3,8,1)), "learning_rate" : [0.1,0.05,]}

#reg_xgb_grid = GridSearchCV(xgb.XGBClassifier(num_classes=3), xgboost_params,n_jobs=-1,cv=5,verbose=2)

#reg_xgb_grid.fit(X_train, y_train)

#reg_xgb=reg_xgb_grid.best_estimator_

#reg_xgb
#print('training accuracy',reg_xgb_grid.best_score_.round(2))
'''

best params:

max_depth=6

learning_rate=0.1

'''
reg_xgb = xgb.XGBClassifier(max_depth=6,num_class=3)
reg_xgb.fit(X_train,y_train)
y_pred = cross_val_predict(reg_xgb, X_train, y_train, cv=5,n_jobs= -1)

model_score = my_scorer(y_train,y_pred)
models['xgb_no_bad_corr_no_isemployed']=[reg_xgb,model_score]
reg_xgb.feature_importances_

xgb.plot_importance(reg_xgb)
# testing to return drop is employed , dropping dependent_number
X = df.drop(['target','age','loan_purpose_home','loan_purpose_investment','loan_purpose_other','repaid_prv_loans','zero_cc_limit','dependent_number',],1)

y = df['target']



X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)
## finding best hyper params

#

#xgboost_params = {"max_depth" : list(range(3,8,1)), "learning_rate" : [0.1,0.05,]}

#reg_xgb_grid = GridSearchCV(xgb.XGBClassifier(num_classes=3), xgboost_params,n_jobs=-1,cv=5,verbose=2)

#reg_xgb_grid.fit(X_train, y_train)

#reg_xgb=reg_xgb_grid.best_estimator_

#reg_xgb
#print('training accuracy',reg_xgb_grid.best_score_.round(2))
'''

best params:

max_depth=6

learning_rate=0.1

'''
reg_xgb = xgb.XGBClassifier(max_depth=6,num_class=3)
reg_xgb.fit(X_train,y_train)
y_pred = cross_val_predict(reg_xgb, X_train, y_train, cv=5,n_jobs= -1)

model_score = my_scorer(y_train,y_pred)
models['xgb_no_bad_corr_no_depnum']=[reg_xgb,model_score]
reg_xgb.feature_importances_

xgb.plot_importance(reg_xgb)
#drop bad corr except for age 
X = df.drop(['target','age','loan_purpose_home','loan_purpose_investment','loan_purpose_other','repaid_prv_loans','zero_cc_limit',],1)

y = df['target']



X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)
## finding best hyper params

#

#xgboost_params = {"max_depth" : list(range(3,8,1)), "learning_rate" : [0.1,0.05,]}

#reg_xgb_grid = GridSearchCV(xgb.XGBClassifier(num_class=3), xgboost_params,n_jobs=-1,cv=5,verbose=2)

#reg_xgb_grid.fit(X_train, y_train)

#reg_xgb=reg_xgb_grid.best_estimator_

#reg_xgb
#print('training accuracy',reg_xgb_grid.best_score_.round(2))
'''

best params:

max_depth=6

learning_rate=0.1

'''
reg_xgb = xgb.XGBClassifier(max_depth=6,num_class=3)
reg_xgb.fit(X_train,y_train)
y_pred = cross_val_predict(reg_xgb, X_train, y_train, cv=5,n_jobs= -1)

model_score = my_scorer(y_train,y_pred)
models['xgb_no_bad_corr_no_depnum']=[reg_xgb,model_score]
reg_xgb.feature_importances_

xgb.plot_importance(reg_xgb)
# select features of best model 
models
_,model = get_best_model(models)
_
# wow!
xgb.plot_importance(model)
# it seems it's best to use features of dataset : all_features

# will now experiment with other classifiers 
# logistic regression
X = df.drop(['target'],1)

y = df['target']



X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)
## finding best hyper params

#

#reg_lg_params = {'penalty': ['none','l2'],'C':[.009,0.01,.09,1,5,10]}

#reg_lg_grid = GridSearchCV(LogisticRegression(), reg_lg_params,n_jobs=-1,cv=5,verbose=2)

#reg_lg_grid.fit(X_train, y_train)

#reg_lg=reg_lg_grid.best_estimator_

#reg_lg
#print('training accuracy',reg_lg_grid.best_score_.round(2))
'''

C=0.009

l2

'''
reg_lg = LogisticRegression(C=0.009,penalty='l2')
reg_lg.fit(X_train,y_train)
y_pred = cross_val_predict(reg_lg, X_train, y_train, cv=5,n_jobs= -1)

model_score = my_scorer(y_train,y_pred)
models['lg_all']=[reg_lg,model_score]
# svm
## finding best hyper params

#

#cl_svm_params = [

#  {'C': [1, 10, 100], 'kernel': ['linear']},

#  {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},

#]

#cl_svm_grid = GridSearchCV(svc(), cl_svm_params,n_jobs=-1,cv=5,verbose=2)

#cl_svm_grid.fit(X_train, y_train)

#cl_svm=cl_svm_grid.best_estimator_

#cl_svm
#print('training accuracy',reg_lg_grid.best_score_.round(2))
'''

took too long to optimize. will use default params for now 

'''
# will use ensembler to train on smaller samples of the data and will also use multithreading. svm vanilla is too slow to train on this dataset
#cl_svm = svc()
#cl_svm.fit(X_train,y_train)
#y_pred = cross_val_predict(cl_svm, X_train, y_train, cv=5,n_jobs= -1)

#model_score = my_scorer(y_train,y_pred)
#models['svm_all']=[cl_svm,model_score]
clf_svm_bag = BaggingClassifier(svc(C=1.0,

       cache_size=200,

       class_weight=None,

       coef0=0.0,

       decision_function_shape=None,

       degree=3,

       gamma='auto',

       kernel='rbf',

       max_iter=-1,

       probability=False,

       random_state=None,

       shrinking=True,

       tol=0.001,

       verbose=2,         

       ),n_jobs = -1)
#clf_svm_bag.fit(X_train,y_train)
'''

also svm bagged / ensembled took too long ... moving along and skipping svm 

'''
# random forest 
## finding best hyper params

#

#rf_params = {'n_estimators': [100,200],'max_features':['auto', 'sqrt'],'min_samples_split':[2, 10],

#            'min_samples_leaf': [ 2, 4]}#,'bootstrap':[True, False]}

#rf_grid = GridSearchCV(rf(), rf_params,n_jobs=-1,cv=5,verbose=2)

#rf_grid.fit(X_train, y_train)

#cl_rf=rf_grid.best_estimator_

#cl_rf
#print('training accuracy',rf_grid.best_score_.round(2))
'''

n_estimators=200

max_features='auto'

min_samples_leaf=4

min_samples_split=10

'''
cl_rf = rf(n_estimators=200,max_features='auto',min_samples_leaf=4,min_samples_split=10,n_jobs=-1)
cl_rf.fit(X_train,y_train)
y_pred = cross_val_predict(cl_rf, X_train, y_train, cv=5,n_jobs= -1)

model_score = my_scorer(y_train,y_pred)
models['rf_all']=[cl_rf,model_score]
# selecting best model based on best score 
models
_,model = get_best_model(models)
_
# testing performance on final model
y_pred = model.predict(X_test)
test_score = my_scorer(y_test,y_pred)
confusion_matrix(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred) * 100.0))

print("Precision (macro): %.2f%%" % (precision_score(y_test, y_pred,average='macro') * 100.0))

print("Precision (weighted): %.2f%%" % (precision_score(y_test, y_pred,average='weighted') * 100.0))

print("Recall (macro): %.2f%%" % (recall_score(y_test, y_pred,average='macro') * 100.0))

print("Recall (weighted): %.2f%%" % (recall_score(y_test, y_pred,average='weighted') * 100.0))

print("F1 (macro): %.2f%%" % (f1_score(y_test, y_pred,average='macro') * 100.0))

print("F1 (weighted): %.2f%%" % (f1_score(y_test, y_pred,average='weighted') * 100.0))



print()

print()

print(confusion_matrix(y_test, y_pred))

print()

print()

print(classification_report(y_test, y_pred))

from sklearn.metrics import plot_confusion_matrix
disp = plot_confusion_matrix(model, X_test, y_test,

                 

                                 cmap=plt.cm.Blues,

                                )

#print(disp.confusion_matrix)
# wanted to plot a beautiful auc curve and roc_auc_score but the sklearn library has an issue. even when following documentation 
auc_score = roc_auc_score(y_test,y_pred,multi_class='ovr')
plt.figure(figsize=(10,5))

fpr, tpr, thresold = roc_curve(y_test,y_pred)

auc_score = roc_auc_score(y_test,y_pred)



plt.title('ROCs' )

plt.plot(fpr, tpr, label=clf + '(AUC score: '+str(auc_score)+')')

plt.legend()

plt.xlabel('FPR', fontsize=10)

plt.ylabel('TPR', fontsize=10)



plt.plot([0, 1], [0, 1], 'k--',label='Random Guess 50/50 (AUC score:0.50)')

plt.legend()

plt.show()
# it seems like the training set and CV correctly reflected the test set in terms of perf
import pandas as pd

borrower_table = pd.read_csv("../input/bank-loan-data/borrower_table.csv")

loan_table = pd.read_csv("../input/bank-loan-data/loan_table.csv")
import pandas as pd

borrower_table = pd.read_csv("../input/bank-loan-data/borrower_table.csv")

loan_table = pd.read_csv("../input/bank-loan-data/loan_table.csv")