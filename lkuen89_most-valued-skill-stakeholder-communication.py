import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../input/multipleChoiceResponses.csv')

questions = df.iloc[0,:]
answers = df.iloc[1:,:]


def tofloat(row, colname, replacement):
    try:
        row[colname] = float(row[colname])
    except:
        row[colname] = replacement
    return row

def tryfloat(row, columns):
    # try to make everything a float that does not resist with force
    for col in columns:
        try:
            row[col] = float(row[col])
        except:
            pass
    return row

def replace_by_middle(row, colname):
    # try to replace range by middle value. It gets kind of messy to deal with the different notations in these columns
    try:
        row[colname] = float(row[colname])
        return row
    except:
        pass
    if row[colname] == 'NaN':
        row[colname] = -999999
        return row
    try:
        row[colname] = float(row[colname].split('+')[0])
        return row
    except:
        pass 
    row[colname] = row[colname].replace(',000','')
    if row[colname] == 'I do not wish to disclose my approximate yearly compensation':
        row[colname] = -999999
        return row
    try:
        split = row[colname].split('-')
        row[colname] = (float(split[0])+float(split[1]))/2
    except:
        #print('Mapping ' + str(row[colname]) + ' to low value')
        row[colname] = -999999
    return row

# time tends to show up prominently, in feature importance. this does not look like a useful result
answers = answers.drop('Time from Start to Finish (seconds)', axis = 1)
tflambda = lambda x: tryfloat(x, answers.columns)
answers = answers.apply(tflambda, axis = 1)

for column in answers.columns:
    if '_OTHER_TEXT' in column:
        conversion = lambda x: tofloat(x,column,-999999)
        answers = answers.apply(conversion, axis=1)

for colname in ['Q2','Q8','Q9']:
    conversion = lambda x: replace_by_middle(x,colname)
    answers = answers.apply(conversion, axis=1)

answers = answers[answers['Q9'] > 0] # dismiss where no answer was provided

# remove were now answer is given
answers = answers[answers['Q9'] > 0]
# The income distribution is pretty skewed, we will cut it off at the price which is still reasonably far away from the median.
# This way the results are more meaningful for the biggest pack of people living around 50K/year
answers = answers[answers['Q9'] < 250]
# hist of incomes
plt.figure()
answers['Q9'].plot.hist()
df_train = pd.get_dummies(answers)
y = df_train['Q9']
X = df_train.drop('Q9', axis = 1)
n_train = int(X.shape[0] * 0.7)
X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
X_valid, y_valid = X.iloc[n_train:], y.iloc[n_train:]

m = lgb.LGBMRegressor(objective = 'regression_l1',
                      num_boost_round=1000,
                        learning_rate = 0.1,
                        num_leaves = 127,
                        max_depth = -1,
                        lambda_l1 = 0.0,
                        lambda_l2 = 1.0,
                        metric = 'l2',
                        seed = 42)  
m.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], 
            early_stopping_rounds=50)
def getQuestion(row):
    row['question'] = row['Feature'].split('_')[0]
    return row

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# sorted(zip(clf.feature_importances_, X.columns), reverse=True)
feature_imp = pd.DataFrame(sorted(zip(m.feature_importances_,X.columns)), columns=['Value','Feature'])
feature_imp['Value'] = feature_imp['Value']/feature_imp['Value'].sum() # normalize to 1
feature_imp = feature_imp.apply(getQuestion, axis = 1)
#feature_imp = feature_imp.drop('Feature', axis = 1)
#feature_imp = feature_imp.groupby(by='Feature', as_index=False).sum()
# group the values by Question
sorted_values = feature_imp.sort_values(by="Value", ascending=False)
# keep only the top 80% important
sorted_values['cum'] = sorted_values['Value'].cumsum()/sorted_values['Value'].sum()
sorted_values = sorted_values[sorted_values['Value'] > 0.005]

# save for later use
important_features = sorted_values['Feature'].copy()
#sorted_values = sorted_values.head(10)

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=sorted_values)
plt.title('LightGBM Question Importanct')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')
print('Correlation of Age and Experience: ' + str(answers['Q2'].corr(answers['Q8'])))
# Age 
sns.jointplot(np.log(answers['Q2']), np.log(answers['Q9']), kind='kde').set_axis_labels('log Age [years]', 'log Income [$1000]', fontsize=16)
# Working Experience im ML
sns.jointplot(np.log(answers['Q8']), np.log(answers['Q9']), kind='kde').set_axis_labels('log Experience in ML/DS [years]', 'log Income [$1000]', fontsize=16)

answers['Q12_Part_4_TEXT'].describe()
def setranking(row):
    if row['Q12_Part_4_TEXT'] < 0:
        row['Q12_ranking'] = 'No answer'
        return row
    row['Q12_ranking'] = 'Some answer'
    return row

answersq12 = answers.copy()
#split by quantiles
answersq12['Q12_ranking'] = 'NaN'
answersq12 = answersq12.apply(setranking, axis=1)
answersq12 = pd.DataFrame(answersq12.filter(['Q12_ranking','Q9'],axis=1).groupby(by='Q12_ranking', as_index=False).median())

#sns.jointplot(answersq12['Q12_Part_4_TEXT'], answersq12['Q9'], kind='kde') # make the col strictly positive

plt.figure(figsize=(20, 10))
sns.barplot(x="Q9", y="Q12_ranking", data=answersq12.sort_values(by='Q9', ascending=False))
plt.title('Median Income in Q12 groups')
plt.tight_layout()
plt.show()
plt.savefig('q12_on_income.png')
def setranking(row):
    q35 = [row['Q35_Part_1'], row['Q35_Part_2'], row['Q35_Part_3'], row['Q35_Part_4'], row['Q35_Part_5'], row['Q35_Part_6']]
    if q35[0] == max(q35):
        row['q35_group'] = 'Self taught'
    if q35[1] == max(q35):
        row['q35_group'] = 'Online Courses'
    if q35[2] == max(q35):
        row['q35_group'] = 'Work'
    if q35[3] == max(q35):
        row['q35_group'] = 'University'
    if q35[4] == max(q35):
        row['q35_group'] = 'Kaggle'
    if q35[5] == max(q35):
        row['q35_group'] = 'Other'
    return row

answersq35 = answers.copy()
#split by quantiles
answersq35['q35_group'] = 'Not specified'
answersq35 = answersq35.apply(setranking, axis=1)
answersq35 = pd.DataFrame(answersq35.filter(['q35_group','Q9'],axis=1).groupby(by='q35_group', as_index=False).median())

#sns.jointplot(answersq12['Q12_Part_4_TEXT'], answersq12['Q9'], kind='kde') # make the col strictly positive

plt.figure(figsize=(20, 10))
sns.barplot(x="Q9", y="q35_group", data=answersq35.sort_values(by="Q9", ascending=False))
plt.title('Median Income in Q35 groups')
plt.tight_layout()
plt.show()
plt.savefig('q35_on_income.png')
def setranking(row):
    q34 = [row['Q34_Part_1'], row['Q34_Part_2'], row['Q34_Part_3'], row['Q34_Part_4'], row['Q34_Part_5'], row['Q34_Part_6']]
    if q34[0] == max(q34):
        row['q34_group'] = 'Gathering data'
    if q34[1] == max(q34):
        row['q34_group'] = 'Cleaning data'
    if q34[2] == max(q34):
        row['q34_group'] = 'Visualizing data'
    if q34[3] == max(q34):
        row['q34_group'] = 'Model building'
    if q34[4] == max(q34):
        row['q34_group'] = 'Putting Model to Production'
    if q34[5] == max(q34):
        row['q34_group'] = 'Stakeholder communication'
    return row

answersq34 = answers.copy()
#split by quantiles
answersq34['q34_group'] = 'Not specified'
answersq34 = answersq34.apply(setranking, axis=1)
answersq34 = pd.DataFrame(answersq34.filter(['q34_group','Q9'],axis=1).groupby(by='q34_group', as_index=False).median())

#sns.jointplot(answersq12['Q12_Part_4_TEXT'], answersq12['Q9'], kind='kde') # make the col strictly positive

plt.figure(figsize=(20, 10))
sns.barplot(x="Q9", y="q34_group", data=answersq34.sort_values(by="Q9", ascending=False))
plt.title('Median Income in Q34 groups')
plt.tight_layout()
plt.show()
plt.savefig('q34_on_income.png')