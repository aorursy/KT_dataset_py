# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/HR_comma_sep.csv')

data.head()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = data['satisfaction_level']

y = data['average_montly_hours']

z = data['last_evaluation']

c = data['number_project']

_ = ax.scatter(xs=x, ys=y, zs=z, c=c)

_ = ax.set_xlabel('Satisfaction Level')

_ = ax.set_ylabel('Average Monthly Hours')

_ = ax.set_zlabel('Last Evaluation')

_ = plt.title('Plot 1: Multivariate Visualization with Number of Projects by Color')

plt.show()
pd.options.mode.chained_assignment = None

data['workhorses'] = 0

data['workhorses'][(data['number_project'] >= 6) & (data['average_montly_hours'] > 200) & (data['satisfaction_level'] < 0.130) & (data['last_evaluation'] > 0.8)] = 1

print("We can identify {} employees who fit the 'workhorse' description according to this analysis.".format(len(data[data['workhorses'] == 1])))



workhorsedf = data[data['workhorses'] == 1]



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = workhorsedf['satisfaction_level']

y = workhorsedf['average_montly_hours']

z = workhorsedf['last_evaluation']

_ = ax.scatter(xs=x, ys=y, zs=z, c='blue')

_ = ax.set_xlabel('Satisfaction Level')

_ = ax.set_ylabel('Average Monthly Hours')

_ = ax.set_zlabel('Last Evaluation')

_ = plt.title('Plot 1a: Multivariate Visualization of Company Workhorses')

plt.show()



data['disengaged'] = 0

data['disengaged'][(data['number_project'] <= 2) & (data['average_montly_hours'] <= 170) & (data['average_montly_hours'] > 110) & (data['satisfaction_level'] < 0.50) & (data['satisfaction_level'] > 0.20) & (data['last_evaluation'] < 0.50) & (data['last_evaluation'] > 0.41)] = 1

print("We can identify {} employees who fit the 'disengaged' description according to this analysis.".format(len(data[data['disengaged'] == 1])))



disengageddf = data[data['disengaged'] == 1]



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = disengageddf['satisfaction_level']

y = disengageddf['average_montly_hours']

z = disengageddf['last_evaluation']

_ = ax.scatter(xs=x, ys=y, zs=z, c='green')

_ = ax.set_xlabel('Satisfaction Level')

_ = ax.set_ylabel('Average Monthly Hours')

_ = ax.set_zlabel('Last Evaluation')

_ = plt.title('Plot 1b: Multivariate Visualization of Disengaged Employees')

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = data['satisfaction_level']

y = data['average_montly_hours']

z = data['last_evaluation']

c = data['time_spend_company']

_ = ax.scatter(xs=x, ys=y, zs=z, c=c)

_ = ax.set_xlabel('Satisfaction Level')

_ = ax.set_ylabel('Average Monthly Hours')

_ = ax.set_zlabel('Last Evaluation')

_ = plt.title('Plot 2: Multivariate Visualization with Tenure at the Company by Color')

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = data['satisfaction_level']

y = data['average_montly_hours']

z = data['last_evaluation']

c = data['left']

_ = ax.scatter(xs=x, ys=y, zs=z, c=c)

_ = ax.set_xlabel('Satisfaction Level')

_ = ax.set_ylabel('Average Monthly Hours')

_ = ax.set_zlabel('Last Evaluation')

_ = plt.title('Plot 3: Multivariate Visualization with Attrition by Color (black if left)')

plt.show()
data['veterans'] = 0

data['veterans'][(data['number_project'] >= 3) & (data['average_montly_hours'] > 200) & (data['satisfaction_level'] > 0.65) & (data['satisfaction_level'] < .93) & (data['last_evaluation'] > 0.78)] = 1

print("We can identify {} employees who fit the 'veteran' description according to this analysis.".format(len(data[data['veterans'] == 1])))



veteransdf = data[data['veterans'] == 1]



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = veteransdf['satisfaction_level']

y = veteransdf['average_montly_hours']

z = veteransdf['last_evaluation']

c = veteransdf['left']

_ = ax.scatter(xs=x, ys=y, zs=z, c=c)

_ = ax.set_xlabel('Satisfaction Level')

_ = ax.set_ylabel('Average Monthly Hours')

_ = ax.set_zlabel('Last Evaluation')

_ = plt.title('Plot 1a: Multivariate Visualization of Company Workhorses')

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = data['satisfaction_level']

y = data['average_montly_hours']

z = data['last_evaluation']

c = data['promotion_last_5years']

_ = ax.scatter(xs=x, ys=y, zs=z, c=c)

_ = ax.set_xlabel('Satisfaction Level')

_ = ax.set_ylabel('Average Monthly Hours')

_ = ax.set_zlabel('Last Evaluation')

_ = plt.title('Plot 4: Multivariate Visualization with Promotions within the Last 5 Years by Color')

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = data['satisfaction_level']

y = data['average_montly_hours']

z = data['last_evaluation']

c = data['Work_accident']

_ = ax.scatter(xs=x, ys=y, zs=z, c=c)

_ = ax.set_xlabel('Satisfaction Level')

_ = ax.set_ylabel('Average Monthly Hours')

_ = ax.set_zlabel('Last Evaluation')

_ = plt.title('Plot 5: Multivariate Visualization with Work Accidents by Color')

plt.show()
_ = sns.distplot(data['time_spend_company'])

_ = plt.title('Plot 6: Histogram of Tenure at Company')

_ = plt.xlabel('Number of Years at Company')

_ = plt.ylabel('Employees')

plt.show()
data['salaries'] = 1

data['salaries'][data['salary'] == 'medium'] = 2

data['salaries'][data['salary'] == 'high'] = 3



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = data['satisfaction_level']

y = data['average_montly_hours']

z = data['last_evaluation']

c = data['salaries']

_ = ax.scatter(xs=x, ys=y, zs=z, c=c)

_ = ax.set_xlabel('Satisfaction Level')

_ = ax.set_ylabel('Average Monthly Hours')

_ = ax.set_zlabel('Last Evaluation')

_ = plt.title('Plot 7: Multivariate Visualization with Salaries by Color')

plt.show()
_ = sns.barplot(x='time_spend_company', y='salaries', data=data, hue='workhorses')

_ = plt.title('Workhorse Salaries and Tenure')

_ = plt.xlabel('Tenure')

_ = plt.ylabel('Average Salary')

plt.show()



_ = sns.barplot(x='time_spend_company', y='salaries', data=data, hue='disengaged')

_ = plt.title('Disengaged Employee Salaries and Tenure')

_ = plt.xlabel('Tenure')

_ = plt.ylabel('Average Salary') 

plt.show()



_ = sns.barplot(x='time_spend_company', y='salaries', data=data, hue='veterans')

_ = plt.title('Veteran Salaries and Tenure')

_ = plt.xlabel('Tenure')

_ = plt.ylabel('Average Salary')

plt.show()
_ = sns.violinplot(x='time_spend_company', y='salaries', data=data)

plt.show()



_ = sns.barplot(x='time_spend_company', y='last_evaluation', hue='salary', hue_order=['low', 'medium', 'high'], data=data)

_ = plt.xlabel('Tenure (in years)')

_ = plt.ylabel('Average of Last Evaluation Score')

_ = plt.title('Average Compensation by Tenure and Performance')

_ = plt.legend(loc=2)

plt.show()



sales = data[data['sales'] == 'sales']

_ = sns.barplot(x='time_spend_company', y='last_evaluation', hue='salary', hue_order=['low', 'medium', 'high'], data=sales)

_ = plt.xlabel('Tenure (in years)')

_ = plt.ylabel('Average of Last Evaluation Score')

_ = plt.title('Average Compensation Among Sales Employees by Tenure and Performance')

_ = plt.legend(loc=2)

plt.show()



corr = np.corrcoef(x=data['last_evaluation'], y=data['salaries'])

print('The correlation between evaluated performance and employee salaries is {}.'.format(corr))
_ = sns.barplot(x='time_spend_company', y='salaries', data=data, hue='left')

_ = plt.title('Salaries and Tenure by Retention/Loss')

_ = plt.xlabel('Tenure')

_ = plt.ylabel('Average Salary')

plt.show()
data['seniors'] = 0

data['seniors'][data['time_spend_company'] > 6] = 1

print("There are {} 'seniors' at this firm.".format(len(data[data['seniors'] == 1])))
data = data.rename(columns={'sales':'department'})
data['dep'] = 1

data['dep'][data['department'] == 'accounting'] = 2

data['dep'][data['department'] == 'hr'] = 3

data['dep'][data['department'] == 'technical'] = 4

data['dep'][data['department'] == 'support'] = 5

data['dep'][data['department'] == 'management'] = 6

data['dep'][data['department'] == 'IT'] = 7

data['dep'][data['department'] == 'product_mng'] = 8

data['dep'][data['department'] == 'RandD'] = 9



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = data['satisfaction_level']

y = data['average_montly_hours']

z = data['last_evaluation']

c = data['dep']

_ = ax.scatter(xs=x, ys=y, zs=z, c=c)

_ = ax.set_xlabel('Satisfaction Level')

_ = ax.set_ylabel('Average Monthly Hours')

_ = ax.set_zlabel('Last Evaluation')

_ = plt.title('Plot 7: Multivariate Visualization with Salaries by Color')

plt.show()
_ = sns.countplot(x='department', data=data)

_ = plt.xlabel('Department')

_ = plt.ylabel('Number of Employees')

_ = plt.title('Employees Assigned per Department')

plt.show()
fp = sns.factorplot(x='number_project', y='satisfaction_level', col='left', row='department', kind='bar', data=data)

_ = fp.set_axis_labels('Number of Projects', 'Satisfaction Level')

_ = fp.set_xticklabels(['2','3','4','5','6','7'])

plt.show()
data['tenure_vs_hours'] = data['time_spend_company'] / data['average_montly_hours']

_ = sns.distplot(data['tenure_vs_hours'])

plt.show()



x = np.corrcoef(x=data['tenure_vs_hours'], y=data['left'])

y = np.corrcoef(x=data['tenure_vs_hours'], y=data['satisfaction_level'])

print(x, y)
data['tenure_vs_projects'] = data['time_spend_company'] / data['number_project']

_ = sns.distplot(data['tenure_vs_projects'])

plt.show()

x = np.corrcoef(x=data['tenure_vs_projects'], y=data['left'])

y = np.corrcoef(x=data['tenure_vs_projects'], y=data['satisfaction_level'])

print(x, y)
def make_feat(df, new_feat_name, feat1, feat2, feat3, feat4):

    df[new_feat_name] = df[feat1] / df[feat2]

    _ = sns.distplot(df[new_feat_name])

    plt.show()

    x = np.corrcoef(x=df[new_feat_name], y=df[feat3])

    y = np.corrcoef(x=df[new_feat_name], y=df[feat4])

    print(x,y)

    return df[new_feat_name]



data['tenure_vs_evaluation'] = make_feat(data, 'tenure_vs_evaluation', 'time_spend_company', 'last_evaluation', 'left', 'satisfaction_level')

data['tenure_vs_salaries'] = make_feat(data, 'tenure_vs_salaries', 'salaries', 'time_spend_company', 'left', 'satisfaction_level')

data['projects_vs_eval'] = make_feat(data, 'projects_vs_eval', 'number_project', 'last_evaluation', 'left', 'satisfaction_level')

data['projects_vs_salary'] = make_feat(data, 'projects_vs_salary', 'number_project', 'salaries', 'left', 'satisfaction_level')

data['salaries_vs_evaluation'] = make_feat(data, 'salaries_vs_evaluation', 'salaries', 'last_evaluation', 'left', 'satisfaction_level')

data['projects_v_eval_vs_salaries'] = make_feat(data, 'projects_v_eval_vs_salaries', 'projects_vs_eval', 'salaries', 'left', 'satisfaction_level')

data['tvs_vs_tve'] = make_feat(data, 'tvs_vs_tve', 'tenure_vs_salaries', 'tenure_vs_evaluation', 'left', 'satisfaction_level')

data['tvp_vs_pvs'] = make_feat(data, 'tvp_vs_pvs', 'tenure_vs_projects', 'projects_vs_salary', 'left', 'satisfaction_level')

data['tvs_vs_tvp'] = make_feat(data, 'tvs_vs_tvp', 'tenure_vs_salaries', 'tenure_vs_projects', 'left', 'satisfaction_level')        
del data['department']

del data['salary']
from sklearn import preprocessing



x = data

scaler = preprocessing.scale(x)

cols = x.columns

data1 = pd.DataFrame(scaler, columns=cols, index=data.index)

data1['left'] = data['left']
from sklearn.decomposition import PCA

X = data

Y = X['left']

del X['left']
pca = PCA(n_components=24)

X = pca.fit_transform(X)

var = pca.explained_variance_ratio_

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

_ = plt.plot(var1)

_ = plt.xlabel('Number of Components')

_ = plt.ylabel('Percent of Explained Variance')

_ = plt.title('Primary Component Breakdown')

plt.show()

print(var1)
pca = PCA(n_components=8)

X = pca.fit_transform(X)

print(X.shape)
def outcomes_score(test_df, outcome_array):

    """Summarize the true/false positives/negatives identified by my model"""

    

    compare = pd.DataFrame(index=test_df.index)

    compare['test'] = test_df

    compare['prediction'] = outcome_array



    # Compute total and percentage of true positives



    compare['True Positive'] = 0

    compare['True Positive'][(compare['test'] == 1) & (compare['prediction'] == 1)] = 1

    truepospercent = np.round(np.sum(compare['True Positive']) / len(compare.index) * 100, decimals=2)

    truepostotal = np.sum(compare['True Positive'])



    # Compute total and percentage of true negatives



    compare['True Negative'] = 0

    compare['True Negative'][(compare['test'] == 0) & (compare['prediction'] == 0)] = 1

    truenegpercent = np.round(np.sum(compare['True Negative']) / len(compare.index) * 100, decimals=2)

    truenegtotal = np.sum(compare['True Negative'])



    # Compute total and percentage of true negatives



    compare['False Positive'] = 0

    compare['False Positive'][(compare['test'] == 0) & (compare['prediction'] == 1)] = 1

    falsepospercent = np.round(np.sum(compare['False Positive']) / len(compare.index) * 100, decimals=2)

    falsepostotal = np.sum(compare['False Positive'])



    # Compute total and percentage of false negatives



    compare['False Negative'] = 0

    compare['False Negative'][(compare['test'] == 1) & (compare['prediction'] == 0)] = 1

    falsenegpercent = np.round(np.sum(compare['False Negative']) / len(compare.index) * 100, decimals=2)

    falsenegtotal = np.sum(compare['False Negative'])



    print('There are {}, or {}%, true positives.'.format(truepostotal, truepospercent))

    print('There are {}, or {}%, true negatives.'.format(truenegtotal, truenegpercent))

    print("Congratulations! You have correctly identified {}, or {}%, of the observed outcomes.".format(truepostotal + truenegtotal, truepospercent + truenegpercent))

    print('There are {}, or {}%, false positives.'.format(falsepostotal, falsepospercent))

    print('There are {}, or {}%, false negatives.'.format(falsenegtotal, falsenegpercent))

    print("Bummer! You incorrectly identified {}, or {}%, of the observed outcomes.".format(falsenegtotal + falsepostotal, falsepospercent + falsenegpercent))

    

def bottomline_score(test_df, outcome_array):

    """Summarize the true/false positives/negatives identified by my model"""

    

    discard_train, verify_test, dscore_train, vscore_test = train_test_split(data, Y, test_size=0.33, random_state=42)

    

    compare = pd.DataFrame(verify_test, columns=data.columns)

    compare['test'] = test_df

    compare['prediction'] = outcome_array

    compare['left'] = Y

    

    compare['estimated_salary'] = 0

    compare['estimated_salary'][compare['salaries'] == 1] = 30000

    compare['estimated_salary'][compare['salaries'] == 2] = 60000

    compare['estimated_salary'][compare['salaries'] == 3] = 90000



    # Compute total and percentage of true positives



    compare['True Positive'] = 0

    compare['True Positive'][(compare['test'] == 1) & (compare['prediction'] == 1)] = 1

    truepospercent = np.sum(compare['True Positive']) / len(compare.index) * 100

    truepostotal = np.sum(compare['True Positive'])



    # Compute total and percentage of true negatives



    compare['True Negative'] = 0

    compare['True Negative'][(compare['test'] == 0) & (compare['prediction'] == 0)] = 1

    truenegpercent = np.sum(compare['True Negative']) / len(compare.index) * 100

    truenegtotal = np.sum(compare['True Negative'])



    # Compute total and percentage of true negatives



    compare['False Positive'] = 0

    compare['False Positive'][(compare['test'] == 0) & (compare['prediction'] == 1)] = 1

    falsepospercent = np.sum(compare['False Positive']) / len(compare.index) * 100

    falsepostotal = np.sum(compare['False Positive'])



    # Compute total and percentage of false negatives



    compare['False Negative'] = 0

    compare['False Negative'][(compare['test'] == 1) & (compare['prediction'] == 0)] = 1

    falsenegpercent = np.sum(compare['False Negative']) / len(compare.index) * 100

    falsenegtotal = np.sum(compare['False Negative'])



    compare['projected_cost'] = 0

    compare['projected_cost'][(compare['salaries'] == 1) & (compare['True Positive'] + compare['False Positive'] == 1)] = compare['estimated_salary'] * .16

    compare['projected_cost'][(compare['salaries'] == 2) & (compare['True Positive'] + compare['False Positive'] == 1)] = compare['estimated_salary'] * .2

    compare['projected_cost'][(compare['salaries'] == 3) & (compare['True Positive'] + compare['False Positive'] == 1)] = compare['estimated_salary'] * 2.13

    

    compare['retained'] = 0



    np.random.seed(50)

    

    nums = []



    for i in range(len(compare)):

        num = np.random.randint(10)

        nums.append(num)

    

    compare['randint'] = nums

    compare['retained'][(compare['randint'] <= 5) & (compare['True Positive'] == 1)] = 1

    

    compare['actual_cost'] = compare['projected_cost'] * compare['retained']

    

    compare['retain_program_cost'] = 0

    compare['retain_program_cost'][compare['True Positive'] + compare['False Positive'] == 1] = compare['projected_cost'] * .25



    projected_cost = np.sum(compare.projected_cost)

    model_cost = np.sum(compare.actual_cost) - np.sum(compare.retain_program_cost)

    savings = projected_cost - model_cost - np.sum(compare.retain_program_cost)

    benefit = projected_cost - model_cost

    employees_retained = np.count_nonzero(compare.retained)

    ROI = np.round(benefit / np.sum(compare.retain_program_cost), decimals=2)

    cost_per_retention = np.round(np.sum(compare.retain_program_cost) / employees_retained, decimals=2)    

    

    print("Using this model will save the firm ${} at a cost of ${}, for an ROI of {}%.".format(savings, np.sum(compare.retain_program_cost), ROI))

    print("The firm will retain {} employees at an average cost of ${} each.".format(employees_retained, cost_per_retention))
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)



my_forest = RandomForestClassifier(n_estimators=100, random_state=42)

my_forest.fit(X_train, y_train)

forest = my_forest.predict(X_test)

print(outcomes_score(y_test, forest))

print(bottomline_score(y_test, forest))
from sklearn.ensemble import GradientBoostingClassifier



gradboost = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

prediction = gradboost.predict(X_test)

print(outcomes_score(y_test, prediction))

print(bottomline_score(y_test, prediction))
from sklearn.ensemble import ExtraTreesClassifier



extratrees = ExtraTreesClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

prediction = extratrees.predict(X_test)

print(outcomes_score(y_test, prediction))

print(bottomline_score(y_test, prediction))
from sklearn.svm import SVC



gm = SVC(random_state=42).fit(X_train, y_train)

prediction = gm.predict(X_test)

print(outcomes_score(y_test, prediction))

print(bottomline_score(y_test, prediction))