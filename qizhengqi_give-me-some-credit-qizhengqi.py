# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import math

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')

df_test = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-test.csv')

df_train.head()
df_train.info()
null_val_sums = df_train.isnull().sum()
pd.DataFrame({'Column': null_val_sums.index, 'Number of Null Values': null_val_sums.values, 'Proportions': null_val_sums.values/len(df_train)})
df_train = df_train.fillna(df_train.median())

print(df_train.isnull().sum())
sns.countplot(x='SeriousDlqin2yrs', data=df_train)

print('Default Rate: {}'.format(df_train['SeriousDlqin2yrs'].sum()/len(df_train)))
age_bins = [-math.inf, 25, 40, 50, 60, 70, math.inf]

dependent_bin = [-math.inf,2,4,6,8,10,math.inf]

dpd_bins = [-math.inf,1,2,3,4,5,6,7,8,9,math.inf]

df_train['bin_age'] = pd.cut(df_train['age'],bins=age_bins).astype(str)

df_train['bin_NumberOfDependents'] = pd.cut(df_train['NumberOfDependents'],bins=dependent_bin).astype(str)

df_train['bin_NumberOfTimes90DaysLate'] = pd.cut(df_train['NumberOfTimes90DaysLate'],bins=dpd_bins)

df_train['bin_NumberOfTime30-59DaysPastDueNotWorse'] = pd.cut(df_train['NumberOfTime30-59DaysPastDueNotWorse'], bins=dpd_bins)

df_train['bin_NumberOfTime60-89DaysPastDueNotWorse'] = pd.cut(df_train['NumberOfTime60-89DaysPastDueNotWorse'], bins=dpd_bins)





df_train['bin_RevolvingUtilizationOfUnsecuredLines'] = pd.qcut(df_train['RevolvingUtilizationOfUnsecuredLines'],q=5,duplicates='drop').astype(str)

df_train['bin_DebtRatio'] = pd.qcut(df_train['DebtRatio'],q=5,duplicates='drop').astype(str)

df_train['bin_MonthlyIncome'] = pd.qcut(df_train['MonthlyIncome'],q=5,duplicates='drop').astype(str)

df_train['bin_NumberOfOpenCreditLinesAndLoans'] = pd.qcut(df_train['NumberOfOpenCreditLinesAndLoans'],q=5,duplicates='drop').astype(str)

df_train['bin_NumberRealEstateLoansOrLines'] = pd.qcut(df_train['NumberRealEstateLoansOrLines'],q=5,duplicates='drop').astype(str)
bin_cols = [c for c in df_train.columns.values if c.startswith('bin_')]
df_train.head()
def cal_IV(df, feature, target):

    lst = []

    cols=['Variable', 'Value', 'All', 'Bad']

    for i in range(df[feature].nunique()):      

        val = list(df[feature].unique())[i]

        lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])

    data = pd.DataFrame(lst, columns=cols)

    data = data[data['Bad'] > 0]



    data['Share'] = data['All'] / data['All'].sum()

    data['Bad Rate'] = data['Bad'] / data['All']

    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())

    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()

    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data['IV'] = (data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])).sum()



    data = data.sort_values(by=['Variable', 'Value'], ascending=True)



    return data['IV'].values[0]
IV_list = []

for f in bin_cols:

    IV_list.append([f,cal_IV(df_train,f,'SeriousDlqin2yrs')])

IV_data = pd.DataFrame(IV_list, columns=['features','IV'])

IV_data
# We choose only those features with IV>0.1

feature_cols = ['RevolvingUtilizationOfUnsecuredLines','NumberOfTime30-59DaysPastDueNotWorse','age','NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse']

bin_cols = ['bin_RevolvingUtilizationOfUnsecuredLines','bin_NumberOfTime30-59DaysPastDueNotWorse','bin_age','bin_NumberOfTimes90DaysLate','bin_NumberOfTime60-89DaysPastDueNotWorse']
def cal_WOE(df,features,target):

    df_new = df

    for f in features:

        df_woe = df_new.groupby(f).agg({target:['sum','count']})

        df_woe.columns = list(map(''.join, df_woe.columns.values))

        df_woe = df_woe.reset_index()

        df_woe = df_woe.rename(columns = {target+'sum':'bad'})

        df_woe = df_woe.rename(columns = {target+'count':'all'})

        df_woe['good'] = df_woe['all']-df_woe['bad']

        df_woe = df_woe[[f,'good','bad']]

        df_woe['bad_rate'] = df_woe['bad']/df_woe['bad'].sum()

        df_woe['good_rate'] = df_woe['good']/df_woe['good'].sum()

        df_woe['woe'] = df_woe['bad_rate'].divide(df_woe['good_rate'],fill_value=1)

        df_woe.columns = [c if c==f else c+'_'+f for c in list(df_woe.columns.values)]

        df_new = df_new.merge(df_woe,on=f,how='left')

    return df_new
df_woe = cal_WOE(df_train,bin_cols,'SeriousDlqin2yrs')

woe_cols = [c for c in list(df_woe.columns.values) if 'woe' in c]

df_woe[woe_cols]
df_bin_to_woe = pd.DataFrame(columns = ['features','bin','woe'])

for f in feature_cols:

    b = 'bin_'+f

    w = 'woe_bin_'+f

    df = df_woe[[w,b]].drop_duplicates()

    df.columns = ['woe','bin']

    df['features'] = f

    df=df[['features','bin','woe']]

    df_bin_to_woe = pd.concat([df_bin_to_woe,df])

df_bin_to_woe
X_train, X_test, y_train, y_test = train_test_split(df_woe[woe_cols], df_woe['SeriousDlqin2yrs'], test_size=0.2, random_state=42)
model = LogisticRegression().fit(X_train,y_train)
model.score(X_test,y_test)
import sklearn.metrics as metrics

# calculate the fpr and tpr for all thresholds of the classification

probs = model.predict_proba(X_test)



preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
y_pred = model.predict(X_test)

metrics.confusion_matrix(y_test,y_pred)
model.coef_
theta_0 = 1/1

P_0 = 650

PDO = 50 # point of double
B = PDO/np.log(2)

A = P_0+B*np.log(theta_0)
def generate_scorecard(model_coef,binning_df,features,B):

    lst = []

    cols = ['Variable','Binning','Score']

    coef = model_coef[0]

    for i in range(len(features)):

        f = features[i]

        df = binning_df[binning_df['features']==f]

        for index,row in df.iterrows():

            lst.append([f,row['bin'],int(round(-coef[i]*row['woe']*B))])

    data = pd.DataFrame(lst, columns=cols)

    return data
score_card = generate_scorecard(model.coef_,df_bin_to_woe,feature_cols,B)

score_card
sort_scorecard = score_card.groupby('Variable').apply(lambda x: x.sort_values('Score', ascending=False))

sort_scorecard
# 本段代码的目的是把输入数据映射到分箱，并且选取相应的分值来计算最后的信用评分

def str_to_int(s):

    if s == '-inf':

        return -999999999.0

    elif s=='inf':

        return 999999999.0

    else:

        return float(s)

    

def map_value_to_bin(feature_value,feature_to_bin):

    for idx, row in feature_to_bin.iterrows():

        bins = str(row['Binning'])

        left_open = bins[0]=="("

        right_open = bins[-1]==")"

        binnings = bins[1:-1].split(',')

        in_range = True

        # check left bound

        if left_open:

            if feature_value<= str_to_int(binnings[0]):

                in_range = False   

        else:

            if feature_value< str_to_int(binnings[0]):

                in_range = False   

        #check right bound

        if right_open:

            if feature_value>= str_to_int(binnings[1]):

                in_range = False 

        else:

            if feature_value> str_to_int(binnings[1]):

                in_range = False   

        if in_range:

            return row['Binning']

    return null



def map_to_score(df,score_card):

    scored_columns = list(score_card['Variable'].unique())

    score = 0

    for col in scored_columns:

        feature_to_bin = score_card[score_card['Variable']==col]

        feature_value = df[col]

        selected_bin = map_value_to_bin(feature_value,feature_to_bin)

        selected_record_in_scorecard = feature_to_bin[feature_to_bin['Binning'] == selected_bin]

        score += selected_record_in_scorecard['Score'].iloc[0]

    return score  



def calculate_score_with_card(df,score_card,A):

    df['score'] = df.apply(map_to_score,args=(score_card,),axis=1)

    df['score'] = df['score']+A

    df['score'] = df['score'].astype(int)

    return df
good_sample = df_train[df_train['SeriousDlqin2yrs']==0].sample(5)

good_sample = good_sample[feature_cols]

bad_sample = df_train[df_train['SeriousDlqin2yrs']==1].sample(5)

bad_sample = bad_sample[feature_cols]
calculate_score_with_card(good_sample,score_card,A)
calculate_score_with_card(bad_sample,score_card,A)