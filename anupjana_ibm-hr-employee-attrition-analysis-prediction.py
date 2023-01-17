# Loading all the required libraries
%matplotlib inline
import os
import pandas as pd
from pandas import ExcelFile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Settings
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize'] = (16, 4)

pd.options.display.max_columns = 500
# Loading IBM HR Employee Attrition dataset
hr_file_name = '../input/WA_Fn-UseC_-HR-Employee-Attrition.csv'

emp_data_org = pd.read_csv(hr_file_name)

print('Dataset dimension: {} rows, {} columns'.format(emp_data_org.shape[0], emp_data_org.shape[1]))
# Metadata of IBM HR Employee Attrition dataset
emp_data_org.info()
# Basic statistics of numerical features
emp_data_org.describe()
# Basic statistics of categorical features
emp_data_org.describe(include=[np.object])
attrition_freq = emp_data_org[['Attrition']].apply(lambda x: x.value_counts())
attrition_freq['frequency_percent'] = round((100 * attrition_freq / attrition_freq.sum()),2)

print(attrition_freq)

# Attrition distribution bar plot
plot = attrition_freq[['frequency_percent']].plot(kind="bar");
plot.set_title("Attrition Distribution", fontsize=20);
plot.grid(color='lightgray', alpha=0.5);
null_feat_df = pd.DataFrame()
null_feat_df['Null Count'] = emp_data_org.isnull().sum().sort_values(ascending=False)
null_feat_df['Null Pct'] = null_feat_df['Null Count'] / float(len(emp_data_org))

null_feat_df = null_feat_df[null_feat_df['Null Pct'] > 0]

total_null_feats = null_feat_df.shape[0]
null_feat_names = null_feat_df.index
print('Total number of features having null values: ', total_null_feats)
del null_feat_df
emp_viz_df = emp_data_org.copy() # Copy cleaned dataset for EDA & feature changes

# Let's add 2 features for EDA: Employee left and not left
emp_viz_df['Attrition_Yes'] = emp_viz_df['Attrition'].map({'Yes':1, 'No':0}) # 1 means Employee Left
emp_viz_df['Attrition_No'] = emp_viz_df['Attrition'].map({'Yes':0, 'No':1}) # 1 means Employee didnt leave

# Let's look into the new dataset and identify features for which plots needs to be build for categorical features
emp_viz_df.head()
cat_col_names = emp_viz_df.select_dtypes(include=[np.object]).columns.tolist() # Get categorical feature names
cat_col_names
def generate_frequency_graph(col_name):
    
    # Plotting of Employee Attrition against feature
    temp_grp = emp_viz_df.groupby(col_name).agg('sum')[['Attrition_Yes', 'Attrition_No']]
    temp_grp['Percentage Attrition'] =  temp_grp['Attrition_Yes'] / (temp_grp['Attrition_Yes'] + temp_grp['Attrition_No']) * 100
    print(temp_grp)
    emp_viz_df.groupby(col_name).agg('sum')[['Attrition_Yes', 'Attrition_No']].plot(kind='bar', stacked=False, color=['red', 'green'])
    plt.xlabel(col_name)
    plt.ylabel('Attrition');
# Plotting of Employee Attrition against Business Travel feature
generate_frequency_graph('BusinessTravel')
# We can see Employee Attrition rate of Travel Frequently is high, then Travel Rarely & then Non-Travel
# Plotting of Employee Attrition against Department feature
generate_frequency_graph('Department')
# We can see Employee Attrition rate for Sales department is high then HR and finally Research & Development
# Plotting of Employee Attrition against Gender feature
generate_frequency_graph('Gender')
# We can see Employee Attrition rate for Male is higher than Female
# Plotting of Employee Attrition against MaritalStatus feature
generate_frequency_graph('MaritalStatus')
# We can see Employee Attrition rate for Single is higher than any other.
# Plotting of Employee Attrition against MaritalStatus feature
generate_frequency_graph('OverTime')
# It is obivious that employee who are doing overtime has higher attrition rate
# Plotting of Employee Attrition against Education feature
# 1-Below College, 2-College, 3-Bachelor, 4-Master, 5-Doctor
generate_frequency_graph('Education')
# We can see Employee Attrition rate is high for Below College & Bachelor
# Plotting of Employee Attrition against Education feature
# 1-Low, 2-Medium, 3-High, 4-Very High
generate_frequency_graph('JobSatisfaction')
# Attrition is directly proportional to JobSatisfaction
# Plotting of Employee Attrition against Education feature
# 1-Bad, 2-Good, 3-Better, 4-Best
generate_frequency_graph('WorkLifeBalance')
# Attrition rate of Bad work life balance is high, however attrition rate of Best is higher than Good
fig = plt.figure(figsize=(16,4))

# Histogram Plot for Employee Age
plt.subplot(1,2,1)
plt.hist(emp_viz_df['Age'], bins=15, color='green')
plt.title('Distribution of Employee Age')
plt.xlabel("Employee Age")

# Histogram Plot for employee Monthly Income
plt.subplot(1,2,2)
plt.hist(emp_viz_df['MonthlyIncome'], bins=30, color='red')
plt.title('Distribution of Employee Monthly Income')
plt.xlabel("Employee Monthly Income")

fig.tight_layout();
fig = plt.figure(figsize=(16,4))

# Histogram Plot for Employee Age
plt.subplot(1,2,1)
plt.hist(np.log1p(emp_viz_df['Age']), bins=20, color='green')
plt.title('Distribution of Employee Age')
plt.xlabel("Log of Employee Age")

# Histogram Plot for employee Monthly Income
plt.subplot(1,2,2)
plt.hist(np.log1p(emp_viz_df['MonthlyIncome']), bins=15, color='red')
plt.title('Distribution of Employee Monthly Income')
plt.xlabel("Log of Employee Monthly Income")

fig.tight_layout();
print('Skewness in employee Age feature: ', emp_viz_df['Age'].skew())
print('Skewness in employee Monthly Income feature: ', emp_viz_df['MonthlyIncome'].skew())
emp_proc_df = emp_data_org.copy() # Copy cleaned dataset for feature engineering

emp_proc_df['TenurePerJob'] = 0

for i in range(0, len(emp_proc_df)):
    if emp_proc_df.loc[i,'NumCompaniesWorked'] > 0:
        emp_proc_df.loc[i,'TenurePerJob'] = emp_proc_df.loc[i,'TotalWorkingYears'] / emp_proc_df.loc[i,'NumCompaniesWorked']

emp_proc_df['YearWithoutChange1'] = emp_proc_df['YearsInCurrentRole'] - emp_proc_df['YearsSinceLastPromotion']
emp_proc_df['YearWithoutChange2'] = emp_proc_df['TotalWorkingYears'] - emp_proc_df['YearsSinceLastPromotion']

monthly_income_median = np.median(emp_proc_df['MonthlyIncome'])
emp_proc_df['CompRatioOverall'] = emp_proc_df['MonthlyIncome'] / monthly_income_median

print('Dataset dimension: {} rows, {} columns'.format(emp_proc_df.shape[0], emp_proc_df.shape[1]))
# Features to remove
feat_to_remove = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']

emp_proc_df.drop( feat_to_remove , axis = 1, inplace = True )
print('Dataset dimension: {} rows, {} columns'.format(emp_proc_df.shape[0], emp_proc_df.shape[1]))
full_col_names = emp_proc_df.columns.tolist()
num_col_names = emp_proc_df.select_dtypes(include=[np.int64,np.float64]).columns.tolist() # Get numerical feature names

# Preparing list of ordered categorical features
num_cat_col_names = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
                     'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance', 'StockOptionLevel']

target = ['Attrition']

num_col_names = list(set(num_col_names) - set(num_cat_col_names)) # Numerical features w/o Ordered Categorical features
cat_col_names = list(set(full_col_names) - set(num_col_names) - set(target)) # Categorical & Ordered Categorical features

print('Total number of numerical features: ', len(num_col_names))
print('Total number of categorical & ordered categorical features: ', len(cat_col_names))

cat_emp_df = emp_proc_df[cat_col_names]
num_emp_df = emp_proc_df[num_col_names]
# Let's create dummy variables for each categorical attribute for training our calssification model
for col in num_col_names:
    if num_emp_df[col].skew() > 0.80:
        num_emp_df[col] = np.log1p(num_emp_df[col])

num_emp_df.head()
# Let's create dummy variables for each categorical attribute for training our calssification model
for col in cat_col_names:
    col_dummies = pd.get_dummies(cat_emp_df[col], prefix=col)
    cat_emp_df = pd.concat([cat_emp_df, col_dummies], axis=1)

# Use the pandas apply method to numerically encode our attrition target variable
attrition_target = emp_proc_df['Attrition'].map({'Yes':1, 'No':0})

# Drop categorical feature for which dummy variables have been created
cat_emp_df.drop(cat_col_names, axis=1, inplace=True)

cat_emp_df.head()
num_corr_df = num_emp_df[['MonthlyIncome', 'CompRatioOverall', 'YearWithoutChange1', 'DistanceFromHome']]
corr_df = pd.concat([num_corr_df, attrition_target], axis=1)
corr = corr_df.corr()

plt.figure(figsize = (10, 8))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.axes_style("white")
#sns.heatmap(data=corr, annot=True, mask=mask, square=True, linewidths=.5, vmin=-1, vmax=1, cmap="YlGnBu")
sns.heatmap(data=corr, annot=True, square=True, linewidths=.5, vmin=-1, vmax=1, cmap="YlGnBu")
plt.show()
# Another way to check for correlation between attributes is to use Pandas’ scatter_matrix function,

from pandas.plotting import scatter_matrix

scatter_matrix(corr_df, figsize=(16, 10));
# Concat the two dataframes together columnwise
final_emp_df = pd.concat([num_emp_df, cat_emp_df], axis=1)

print('Dataset dimension after treating categorical features with dummy variables: {} rows, {} columns'.format(final_emp_df.shape[0], final_emp_df.shape[1]))
final_emp_df.head()
# Import the train_test_split method
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Split data into train and test sets as well as for validation and testing
X_train, X_val, y_train, y_val = train_test_split(final_emp_df, attrition_target,
                                                  test_size= 0.30, random_state=42);

print("Stratified Sampling: ", len(X_train), "train set +", len(X_val), "validation set")

# Stratified Splitting
#split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
#for train_index, test_index in split.split(emp_data_proc, emp_data_proc['Gender']):
#    strat_train_set = emp_data_proc.loc[train_index]
#    strat_test_set = emp_data_proc.loc[test_index]
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
def gen_model_performance(actual_target, pred_target):
    model_conf_matrix = confusion_matrix(actual_target, pred_target)
    model_roc_score = roc_auc_score(actual_target, pred_target)
    model_accuracy = accuracy_score(actual_target, pred_target) * 100.0
    
    TP = model_conf_matrix[1][1]; TN = model_conf_matrix[0][0]; 
    FP = model_conf_matrix[0][1]; FN = model_conf_matrix[1][0];
    sensitivity = TP / float(TP + FN) * 100.0; specificity = TN / float(TN + FP) * 100.0;
    precision = TP / float(TP + FP) * 100.0;
    
    return sensitivity, specificity, model_accuracy, precision, model_roc_score
def evaluate_model_score(X, y, scoring='accuracy'):
    
    logreg_model = LogisticRegression(random_state=0)
    logreg_cv_model = LogisticRegressionCV()
    rfc_model = RandomForestClassifier()
    extrees_model = ExtraTreesClassifier()
    gboost_model = GradientBoostingClassifier()
    dt_model = DecisionTreeClassifier()
    aboost_model = AdaBoostClassifier()
    gnb_model = GaussianNB()

    models = [logreg_model, logreg_cv_model, dt_model, rfc_model, 
              extrees_model, gboost_model, aboost_model, gnb_model]
    
    model_results = pd.DataFrame(columns = ["Model", "Accuracy", "Precision", "CV Score",
                                            "Sensitivity","Specificity","ROC Score"])
    
    for model in models:
        model.fit(X, y,)
        y_pred = model.predict(X)
        score = cross_val_score(model, X, y, cv=5, scoring=scoring)
        
        sensitivity, specificity, accuracy, precision, roc_score = gen_model_performance(y, y_pred)
    
        scores = cross_val_score(model, X, y, cv=5)
    
        model_results = model_results.append({"Model": model.__class__.__name__,
                              "Accuracy": accuracy, "Precision": precision,
                              "CV Score": scores.mean()*100.0,
                              "Sensitivity": sensitivity, "Specificity": specificity,
                              "ROC Score": roc_score}, ignore_index=True)
    return model_results
model_results = evaluate_model_score(X_train, y_train)

model_results
rfc_model = RandomForestClassifier();

refclasscol = X_train.columns

# fit random forest classifier on the training set
rfc_model.fit(X_train, y_train);

# extract important features
score = np.round(rfc_model.feature_importances_, 3)
importances = pd.DataFrame({'feature':refclasscol, 'importance':score})
importances = importances.sort_values('importance', ascending=False).set_index('feature')

# random forest classifier parameters used for feature importances
print(rfc_model)

# plot importances
#importances.plot.bar();
high_imp_df = importances[importances.importance>=0.015]
high_imp_df.plot.bar();
del high_imp_df
mid_imp_df = importances[importances.importance<=0.015]
mid_imp_df = mid_imp_df[mid_imp_df.importance>=0.0050]
mid_imp_df.plot.bar();
del mid_imp_df
selection = SelectFromModel(rfc_model, threshold = 0.002, prefit=True)

X_train_select = selection.transform(X_train)
X_val_select = selection.transform(X_val)

print('Train dataset dimension before Feature Selection: {} rows, {} columns'.format(X_train.shape[0], X_train.shape[1]))
print('Train dataset dimension after Feature Selection: {} rows, {} columns'.format(X_train_select.shape[0], X_train_select.shape[1]))
model_results = evaluate_model_score(X_train_select, y_train)

model_results
final_rfc_model = RandomForestClassifier()
final_rf_scores = cross_val_score(final_rfc_model, X_train_select, y_train, cv=5)

final_rfc_model.fit(X_train_select, y_train)
y_trn_pred = final_rfc_model.predict(X_train_select)
sensitivity, specificity, accuracy, precision, roc_score = gen_model_performance(y_train, y_trn_pred)

print("Train Accuracy: %.2f%%, Precision: %.2f%%, CV Mean Score=%.2f%%, Sensitivity=%.2f%%, Specificity=%.2f%%" % 
      (accuracy, precision, final_rf_scores.mean()*100.0, sensitivity, specificity))
print('***************************************************************************************\n')

y_val_pred = final_rfc_model.predict(X_val_select)
sensitivity, specificity, accuracy, precision, roc_score = gen_model_performance(y_val, y_val_pred)

print("Validation Accuracy: %.2f%%, Precision: %.2f%%, Sensitivity=%.2f%%, Specificity=%.2f%%" % 
      (accuracy, precision, sensitivity, specificity))
print('***************************************************************************************\n')
from imblearn.over_sampling import SMOTE

oversampler=SMOTE(random_state=0)
X_train_smote, y_train_smote = oversampler.fit_sample(X_train_select, y_train)
model_results = evaluate_model_score(X_train_smote, y_train_smote)

model_results
final_rfc_model = RandomForestClassifier()
final_rf_scores = cross_val_score(final_rfc_model, X_train_smote, y_train_smote, cv=5)

final_rfc_model.fit(X_train_smote, y_train_smote)
y_trn_pred = final_rfc_model.predict(X_train_smote)
sensitivity, specificity, accuracy, precision, roc_score = gen_model_performance(y_train_smote, y_trn_pred)

print("Train Accuracy: %.2f%%, Precision: %.2f%%, CV Mean Score=%.2f%%, Sensitivity=%.2f%%, Specificity=%.2f%%" % 
      (accuracy, precision, final_rf_scores.mean()*100.0, sensitivity, specificity))
print('***************************************************************************************\n')

y_val_pred = final_rfc_model.predict(X_val_select)
sensitivity, specificity, accuracy, precision, roc_score = gen_model_performance(y_val, y_val_pred)

print("Validation Accuracy: %.2f%%, Precision: %.2f%%, Sensitivity=%.2f%%, Specificity=%.2f%%" % 
      (accuracy, precision, sensitivity, specificity))
print('***************************************************************************************\n')
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# Hyperparameters Tuning for Random Forest
rfc_param_grid = {
                 'max_depth' : [None, 4, 6, 8],
                 'n_estimators': [10, 30, 50],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 }
rfc_best_model = RandomForestClassifier()
rfc_cross_val = StratifiedKFold(n_splits=5)

rfc_grid_search = GridSearchCV(rfc_best_model,
                               scoring='accuracy',
                               param_grid=rfc_param_grid,
                               cv=rfc_cross_val,
                               verbose=1)

rfc_grid_search.fit(X_train_smote, y_train_smote)
rfc_model = rfc_grid_search
rfc_parameters = rfc_grid_search.best_params_

rfc_best_model = RandomForestClassifier(**rfc_parameters)

scores = cross_val_score(rfc_best_model, X_train_smote, y_train_smote, cv=5, scoring='accuracy')
    
print('Cross-validation of : {0}'.format(rfc_best_model.__class__))
print('After Hyperparameters tuning CV score = {0}'.format(np.mean(scores) * 100.0))
print('Best score: {}'.format(rfc_grid_search.best_score_))
print('Best parameters: {}'.format(rfc_grid_search.best_params_))

rfc_best_model.fit(X_train_smote, y_train_smote)
y_trn_pred = rfc_best_model.predict(X_train_smote)
sensitivity, specificity, accuracy, precision, roc_score = gen_model_performance(y_train_smote, y_trn_pred)

print("\nTrain Accuracy: %.2f%%, Precision: %.2f%%, CV Mean Score=%.2f%%, Sensitivity=%.2f%%, Specificity=%.2f%%" % 
      (accuracy, precision, scores.mean()*100.0, sensitivity, specificity))
print('***************************************************************************************\n')
y_val_pred = rfc_best_model.predict(X_val_select)
sensitivity, specificity, accuracy, precision, roc_score = gen_model_performance(y_val, y_val_pred)

print("Validation Accuracy: %.2f%%, Precision: %.2f%%, Sensitivity=%.2f%%, Specificity=%.2f%%" % 
      (accuracy, precision, sensitivity, specificity))
print('***************************************************************************************\n')