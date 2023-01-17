from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.feature_selection import SelectKBest, chi2, f_classif # For the dimensionality reduction

# For the tree visualization
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO

# To enable plotting graphs in Jupyter notebook
%matplotlib inline 
#print(pd.options.display.max_rows) # default is 60 
#print(pd.options.display.max_columns) # default is 20
#print(pd.options.display.max_colwidth) # default is 50

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns",100)
pd.set_option("display.max_colwidth", 500)

pd.set_option('precision', 4)

xlsx = pd.read_excel('/kaggle/input/hr-analytics-case-study/data_dictionary.xlsx')
general = pd.read_csv('/kaggle/input/hr-analytics-case-study/general_data.csv')
emp_survey = pd.read_csv('/kaggle/input/hr-analytics-case-study/employee_survey_data.csv')
mgr_survey = pd.read_csv('/kaggle/input/hr-analytics-case-study/manager_survey_data.csv')
intime = pd.read_csv('/kaggle/input/hr-analytics-case-study/in_time.csv')
outtime = pd.read_csv('/kaggle/input/hr-analytics-case-study/out_time.csv')

print('general data shape = ', general.shape)
print('employee survey data shape = ', emp_survey.shape)
print('manager survey data shape = ', mgr_survey.shape)
print('in time data shape = ', intime.shape)
print('out time data shape = ', outtime.shape)
print('excel data shape = ', xlsx.shape)
xlsx
general.head()
emp_survey.head()
mgr_survey.head()
intime.head()
outtime.head()
# change column name 'Unnamed: 0' to 'EmployeeID' in the in-time and out-time datasets.
intime.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace=True)
outtime.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace=True)
general.set_index('EmployeeID', inplace=True)
emp_survey.set_index('EmployeeID', inplace=True)
mgr_survey.set_index('EmployeeID', inplace=True)
intime.set_index('EmployeeID', inplace=True)
outtime.set_index('EmployeeID', inplace=True)
main_df = pd.concat([general, emp_survey, mgr_survey], axis = 1)
main_df.head()
dtypes_df = pd.DataFrame(main_df.dtypes, columns=['Type'])
dtypes_df['Null'] = main_df.isnull().sum()
dtypes_df['N-Unique'] = main_df.nunique()
dtypes_df['Unique'] = [main_df[col].unique() if main_df[col].dtype == 'object' else [] for col in main_df]
dtypes_df
#intime = intime.apply(pd.to_datetime)
#outtime = outtime.apply(pd.to_datetime)
#main_df['WorkingHours'] = (outtime - intime).mean(axis=1)
#main_df['WorkingHours'] = main_df['WorkingHours'] / np.timedelta64(1, 's') # Convert Timedelta units to seconds (float64)
#main_df['Overtime'] = main_df['WorkingHours'] - 28800 # (8 working hours per day * 3600)
#main_df.head()
intime = intime.apply(pd.to_datetime)
outtime = outtime.apply(pd.to_datetime)

time = outtime - intime

# Convert Timedelta units to seconds (float64)
time = time / np.timedelta64(1, 's')

# convert Time-stamp to seconds (float64)
intime = intime.applymap(lambda x: 3600*x.hour + 60*x.minute + x.second)
outtime = outtime.applymap(lambda x: 3600*x.hour + 60*x.minute + x.second)
# Compute Mean and Std of working hours per employee in seconds (float64)
main_df['in_mean'] = intime.mean(axis=1)
main_df['out_mean'] = outtime.mean(axis=1)

main_df['in_std'] = intime.std(axis=1)
main_df['out_std'] = outtime.std(axis=1)

main_df['time_mean'] = time.mean(axis=1)
main_df['time_std'] = time.std(axis=1)

# Compute 'overtime_mean' column in seconds (float64)
main_df['overtime_mean'] = main_df['time_mean'] - 28800 # (8 working hours a day * 3600) 

main_df.head()
# Compute Ratio of NaN for specific employee
#intime.iloc[4,].isnull().sum() / intime.iloc[4,].count()
# Compute Ratio of NaN to not-NaN values
main_df['ratio_NaN_time'] = (intime.isnull().sum(axis=1)).divide(intime.count(axis=1)) # count() does not include NA values.
main_df['ratio_NaN_time']

sns.boxplot(x="Attrition", y="time_mean", data=main_df);
sns.boxplot(x="Attrition", y="overtime_mean", data=main_df);
sns.boxplot(x="Attrition", y="ratio_NaN_time", data=main_df);
# The usual way to test for a NaN is to see if it's equal to itself:
def isNaN(my_val):
    return my_val != my_val
#------------------------------------
def intime_to_cat(my_int):
    """
    10:30 - after --> Late (2)
    09:30 - 10:29 --> Normal (1)
    before 09:30 --> Early (0)
    """
    if isNaN(my_int):
        return np.nan
    time_str = datetime.fromtimestamp(my_int).strftime('%H:%M')
    hh, mm = map(int, time_str.split(':'))
    if (hh >= 11) or (hh == 10 and mm >= 30):
        return 2 # 'InLate'
    elif (hh >= 10) or (hh == 9 and mm >= 30):
        return 1 # 'InNormal'
    else:
        return 0 # 'InEarly'
#------------------------------------    
def outtime_to_cat(my_int):
    """
    18:30 - after --> Late (2)
    17:30 - 18:29 --> Normal (1)
    before 17:30 --> Early (0)
    """
    if isNaN(my_int):
        return np.nan
    time_str = datetime.fromtimestamp(my_int).strftime('%H:%M')
    hh, mm = map(int, time_str.split(':'))
    if (hh >= 19) or (hh == 18 and mm >= 30):
        return 2 # 'OutLate'
    elif (hh >= 18) or (hh == 17 and mm >= 30):
        return 1 # 'OutNormal'
    else:
        return 0 # 'OutEarly'
#------------------------------------
def time_to_cat(my_int):
    """
    8:00 or more --> Over-Time (2)
    7:00 - 7:59 --> Normal Time (1)
    until 6:59 --> Short Time (0)
    """
    if isNaN(my_int):
        return np.nan
    hours = int(datetime.fromtimestamp(my_int).strftime('%H'))
    if hours >= 8:
        return 2 # 'OverTime'
    elif hours >= 7:
        return 1 # 'NormalTime'
    else:
        return 0 # 'ShortTime'
#------------------------------------    
def int_to_date_str(my_int): # translates seconds to date string
    if isNaN(my_int):
        return np.nan
    return datetime.fromtimestamp(my_int).strftime('%H:%M:%S')
#time = outtime - intime

# Convert Timedelta units to seconds (float64)
#time = time / np.timedelta64(1, 's')

# Convert Time-stamp to seconds (float64)
#intime = intime.applymap(lambda x: 3600*x.hour + 60*x.minute + x.second)
#outtime = outtime.applymap(lambda x: 3600*x.hour + 60*x.minute + x.second)
intime_cat = intime.applymap(intime_to_cat)

outtime_cat = outtime.applymap(outtime_to_cat)

time_cat = time.applymap(time_to_cat)
time_cat.head()
main_df['time_cat_mean'] = time_cat.mean(axis=1)
main_df.head()
intime.median(axis=1).hist(bins=100, figsize=(30,10));
plt.title('MEDIAN IN TIME');
outtime.median(axis=1).hist(bins=100, figsize=(30,10), color="g");
plt.title('MEDIAN OUT TIME');
intime.std(axis=1).hist(bins=100, figsize=(30,10));
plt.title('STD IN TIME');
outtime.std(axis=1).hist(bins=100, figsize=(30,10), color="g");
plt.title('STD OUT TIME');
empID = 5 # empID starts from 1
time.iloc[2-1,].plot(figsize=(30,10))
plt.figure(figsize=(30,8));

plt.subplot(3,1,1);
intime_cat.iloc[empID-1,].plot(figsize=(18,20));
plt.title("In time categories")

plt.subplot(3,1,2);
outtime_cat.iloc[empID-1,].plot(figsize=(18,20), color='g');
plt.title("Out time categories")

plt.subplot(3,1,3);
time_cat.iloc[empID-1,].plot(figsize=(18,20), color='r');
plt.title("Total Working time categories")
print("********* All-employees Mean In-Times *********")
print(intime.mean().apply(int_to_date_str))
print("\nMean in-time:", int_to_date_str(intime.mean().mean()))

print("\n********* All-employees Std In-Times *********")
print(intime.std().apply(int_to_date_str))
print("\nMean Std in-time:", int_to_date_str(intime.std().mean()))
print("********* All-employees Mean Out-Times *********")
print(outtime.mean().apply(int_to_date_str))
print("\nMean out-time:", int_to_date_str(outtime.mean().mean()))

print("\n********* All-employees Std Out-Times *********")
print(outtime.std().apply(int_to_date_str))
print("\nMean Std out-time:", int_to_date_str(outtime.std().mean()))
# check missing values count
null_columns = main_df.columns[main_df.isnull().any()]
main_df[null_columns].isnull().sum()
plt.figure(figsize=(25,8))

plt.subplot(1,5,1)
main_df['NumCompaniesWorked'].plot(kind='density', color='teal');
plt.title('Density Plot Of Number Of \nCompanies Worked')

plt.subplot(1,5,2)
main_df['TotalWorkingYears'].plot(kind='density', color='blue');
plt.title('Density Plot Of \nTotal Working Years')

plt.subplot(1,5,3)
main_df['EnvironmentSatisfaction'].plot(kind='density', color='teal');
plt.title('Density Plot Of \nEnvironment Satisfaction')

plt.subplot(1,5,4)
main_df['JobSatisfaction'].plot(kind='density', color='blue');
plt.title('Density Plot Of \nJob Satisfaction')

plt.subplot(1,5,5)
main_df['WorkLifeBalance'].plot(kind='density', color='green');
plt.title('Density Plot Of \nWork Life Balance')
#null_col = ['NumCompaniesWorked', 'TotalWorkingYears', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
#for i in null_col:
#    main_data[i] = main_data[i].fillna(main_data[i].median())
# drop rows with missing values
main_df.dropna(inplace=True)
# check again to confirm there are no more missing values
null_columns = main_df.columns[main_df.isnull().any()]
main_df[null_columns].isnull().sum()
# descriptive statistics. We use .T for Transposition 
main_df.describe(include='all').T
# There are columns like EmployeeCount, Over18, StandardHours that have only 1 value, hence we drop them 
# 'EmployeeCount' is always 1
# 'Over18' is always 'Y'
# 'StandardHours' is always 8
main_df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)
breakpoint_df = main_df.copy()
#main_df = breakpoint_df.copy()
# Attrition Ratio Pie Diagram

attrition_value_counts = main_df['Attrition'].value_counts()

print(attrition_value_counts)
pct_attrition = (len(main_df[main_df['Attrition']=='Yes'])/len(main_df))*100
print('Rate of Attrition for Entire Company: {:.2f}%'.format(pct_attrition))

plt.pie(attrition_value_counts, labels=['Not Attrited', 'Attrited']);
sns.pairplot(main_df[['Age','MonthlyIncome','DistanceFromHome','Attrition']],hue = 'Attrition');
num_main_df = main_df.select_dtypes(include=['number'])
plt.figure(figsize=(16,10))
sns.heatmap(num_main_df.corr(), annot=True);
main_df.hist(figsize=(20, 15));
main_df['YearsAtCompany'].hist(figsize=(20, 15), bins=100);
main_df['YearsSinceLastPromotion'].hist(figsize=(20, 15), bins=100, color='k');
sns.boxplot(x="Attrition", y="Age", data=main_df);
main_df[main_df['Attrition']=='Yes'].groupby('Age')['Attrition'].count()\
 .plot(figsize=(20,10), title='Age wise Attrition');
main_df.groupby('YearsAtCompany')['Attrition'].apply(lambda s : (s=='Yes').sum()/len(s)).plot(figsize=(30, 10), linestyle='-.', marker="d");
# maybe throw away outlier's data
main_df.groupby('YearsSinceLastPromotion')['Attrition'].apply(lambda s : (s=='Yes').sum()/len(s)).plot(figsize=(30, 10),color='r', linestyle='-.', marker="d");
# maybe throw away outlier's data
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)
sns.distplot(main_df['Age'], color='green');
plt.xlim(10,70)
plt.title('Age Distribution')

plt.subplot(1,3,2)
main_df['MaritalStatus'].value_counts().plot(kind='bar', color='lightblue');
plt.xticks(rotation=0)
plt.title('Marital Status Distribution')

plt.subplot(1,3,3)
main_df['Gender'].value_counts().plot(kind='bar', color='lightpink');
plt.xticks(rotation=0)
plt.title('Gender Distribution');
plt.figure(figsize=(16,10))

plt.subplot(2,3,4)
main_df['Department'].value_counts().plot(kind='bar', color='lightblue');
plt.xticks(rotation=0)
plt.title('Department Distribution');

plt.subplot(2,3,5)
main_df['JobRole'].value_counts().plot(kind='bar', color='lightblue');
plt.title('Job Role Distribution');

plt.subplot(2,3,6)
main_df['EducationField'].value_counts().plot(kind='bar', color='lightblue');
plt.title('Education Field Distribution');
#main_df['AgeGroupCategory'] = pd.cut(main_df['Age'], bins=[0,26,35,120], labels=['Junior', 'Busy', 'Senior'])
main_df['AgeGroupRange'] = pd.cut(main_df['Age'], range(10, 70, 10))

main_df.drop('Age', axis=1, inplace=True)
#def Age(dataframe):
#    dataframe.loc[dataframe['Age'] <= 30, 'AgeGroup'] = 1
#    dataframe.loc[(dataframe['Age'] > 30) & (dataframe['Age'] <= 40), 'AgeGroup'] = 2
#    dataframe.loc[(dataframe['Age'] > 40) & (dataframe['Age'] <= 50), 'AgeGroup'] = 3
#    dataframe.loc[(dataframe['Age'] > 50) & (dataframe['Age'] <= 60), 'AgeGroup'] = 4
#    dataframe.loc[(dataframe['Age'] > 60), 'AgeGroup'] = 5
#    return dataframe

#Age(main_df); 

#main_df.drop(['Age'], axis=1, inplace=True)
main_df.head()
print(main_df.AgeGroupRange.unique())
print(main_df.BusinessTravel.unique())
print(main_df.Department.unique())
print(main_df.Gender.unique())
print(main_df.MaritalStatus.unique())
print(main_df.EducationField.unique())
print(main_df.JobRole.unique())
main_df.JobRole.value_counts()
graphs = ['AgeGroupRange', 'MaritalStatus', 'Gender', 'Department', 'BusinessTravel', 'JobRole', 'EducationField']
plt.figure(figsize=(20,15))
for index, item in enumerate(graphs):
    plt.subplot(3,3,index+1)
    ax = sns.countplot(x=item, hue='Attrition', data=main_df, palette='husl')
    if index+1>3: plt.xticks(rotation=90)
    index = int(len(ax.patches)/2)
    for left,right in zip(ax.patches[:index], ax.patches[index:]):
        left_height = left.get_height()
        right_height = right.get_height()
        total = left_height + right_height
        ax.text(left.get_x() + left.get_width()/2., left_height + 20, '{:.1%}'.format(left_height/total), ha="center")
        ax.text(right.get_x() + right.get_width()/2., right_height + 20, '{:.1%}'.format(right_height/total), ha="center")
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,7))

sns.countplot(ax=ax1, x='Attrition', data=main_df, hue='JobLevel');
sns.countplot(ax=ax2, x='Attrition', data=main_df, hue='Gender');

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,7))

sns.countplot(ax=ax1, x='Attrition', data=main_df, hue='MaritalStatus');
sns.countplot(ax=ax2, x='Attrition', data=main_df, hue='AgeGroupRange');
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,7))

sns.countplot(ax=ax1, x='Attrition', data=main_df, hue='Department');
sns.countplot(ax=ax2, x='Attrition', data=main_df, hue='BusinessTravel');
# Dropping features: 'EducationField' (6 categories), 'JobRole' (9 categories). 
# The are 2 reasons why i decided to drop these columns:
# 1. They have a lot of unique values.
# 2. They don't seem as important or interesing as other features. 
main_df.drop(['EducationField', 'JobRole'], axis=1, inplace=True)
# Compute average of scores from survey data files. Each score is ‘low’=1, ‘medium’=2, ‘high’=3, ‘very high’=4 or equivalent scale.
main_df['SurveyAverageScore'] = (main_df['EnvironmentSatisfaction'] + main_df['JobSatisfaction'] + main_df['WorkLifeBalance'] +
                                 main_df['JobInvolvement'] + main_df['PerformanceRating']) / 5
main_df.drop(['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'JobInvolvement', 'PerformanceRating'], axis=1, inplace=True)
X = main_df.drop(['Attrition'], axis=1)
y = main_df['Attrition']
#y.replace({'Yes':1, 'No':0}, inplace=True)
X = pd.get_dummies(X)
X.head()
#encoder = LabelBinarizer() # sparse_output=True 
#X['BusinessTravel'] = encoder.fit_transform(X['BusinessTravel'])
#X['Department'] = encoder.fit_transform(X['Department'])    
#X['EducationField'] = encoder.fit_transform(X['EducationField'])    
#X['Gender'] = encoder.fit_transform(X['Gender'])    
#X['JobRole'] = encoder.fit_transform(X['JobRole'])    
#X['MaritalStatus'] = encoder.fit_transform(X['MaritalStatus'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)
def visualize_tree(model, max_depth=5, width=800):
    dot_data = StringIO()  
    export_graphviz(model, out_file=dot_data, feature_names=X_train.columns, max_depth=max_depth, 
                    leaves_parallel=True, filled=True, class_names=model.classes_)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
    return Image(graph.create_png(), width=width) 

def print_dot_text(model, max_depth=5):
    """The output of this function can be copied to http://www.webgraphviz.com/"""
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, feature_names=X_train.columns, max_depth=max_depth,
                   leaves_parallel=True, filled=True, class_names=model.classes_)
    dot_text = dot_data.getvalue()
    print(dot_text)

param_grid = {'max_depth': [3, 7, 12], 'min_samples_leaf': [5, 15, 40], 'min_samples_split': [5, 10]} 
gs_cv = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=5, scoring='roc_auc')

gs_cv.fit(X_train, y_train)

df_results_train = pd.DataFrame(gs_cv.cv_results_)[['param_max_depth', 'param_min_samples_leaf', 'param_min_samples_split', 'mean_test_score']]

df_results_train
print("Train: Tuned Decision Tree Parameters: {}".format(gs_cv.best_params_))
print("Train: Best score is {:.3f}".format(gs_cv.best_score_))

scores_mean = df_results_train['mean_test_score'].mean()
scores_std = df_results_train['mean_test_score'].std()

print("Train: Mean {:.3f}, STD {:.3f}".format(scores_mean, scores_std))
importances = list(zip(gs_cv.best_estimator_.feature_importances_, X_train.columns))
sorted(importances, key = lambda x: x[0], reverse=True)
visualize_tree(gs_cv.best_estimator_)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, min_samples_split=10) # n_estimators default is 100. (Changed from 10 to 100 in sklearn version 0.22.)

k = 10

cv_results = cross_val_score(rf_clf, X_train, y_train, cv=k, scoring='roc_auc')

print("'roc_auc' Scores : " + (k * "{:.3f} ").format(*cv_results))
print("Mean {:.3f}, STD {:.3f}".format(cv_results.mean(), cv_results.std()))
importances = list(zip(rf_clf.feature_importances_, X_train.columns))
sorted(importances, key = lambda x: x[0], reverse=True)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, min_samples_split=5) # n_estimators default is 100. (Changed from 10 to 100 in sklearn version 0.22.)

rf_clf.fit(X_train, y_train)

# Compute predicted probabilities: y_pred_prob
y_pred_prob = rf_clf.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label='Yes')

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label='Yes')

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
from scipy.stats import randint
# randint(a,b) creates a new random variable, that has a discrete uniform distribution with possible outcomes a, ..., b-1.

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

tree = DecisionTreeClassifier()

tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X_train, y_train) 

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

model_params_dict = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params': {
            'gamma': ['scale', 'auto'], 
            'C': [1, 10, 20],
            'kernel': ['rbf']
        } 
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': { 
            'n_estimators': [10, 100]
        } 
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': { 
            'C': [1, 5, 10]
        } 
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [2, 5]
        }
        
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'min_samples_leaf': [1, 3, 5]
        }   
    }
}
scores = []
for model_name, model_params in model_params_dict.items():
  clf = GridSearchCV(model_params['model'], model_params['params'], cv=5)
  #clf = RandomizedSearchCV(model_params['model'], model_params['params'], cv=5, n_iter=2)
  clf.fit(X_train, y_train)
  scores.append({
      'model': model_name,
      'best_score': clf.best_score_,
      'best_params': clf.best_params_
  })

df_results = pd.DataFrame(scores, columns = ['model', 'best_score', 'best_params'])
df_results