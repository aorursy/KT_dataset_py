import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score,recall_score, precision_recall_curve
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype, is_object_dtype
from sklearn.model_selection import train_test_split,  GridSearchCV, StratifiedKFold, KFold, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, confusion_matrix, make_scorer
## Helper functions

## display all columns
def display_all(df):
    "Function to display trauncated columns/rows while displaying"
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

## Fix missing values
def fix_missing(df, col, name, na_dict):
    """Repalce missing values in a column when the column is 
    numerical type -> replace with median
    object tupe -> replace with mode
    Also reates an extra column indicating if that row is missing from the data or not
    """
    if pd.isnull(col).sum() or (name in na_dict):
        df[name+'_na'] = pd.isnull(col)
        if is_numeric_dtype(col):
            filler = na_dict[name] if name in na_dict else col.median()
        else:
            filler = na_dict[name] if name in na_dict else col.mode()[0]
        df[name] = col.fillna(filler)
        na_dict[name] = filler
    return na_dict

def countPlot(data, ax1, ax2, title, rotation=0, hue=None):
    "count plot the discrete type features"
    chart = sns.countplot(data, ax=axes[ax1,ax2] , hue=hue)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=rotation)
    chart.set_title(title)
    
def warn(*args, **kwargs):
    "Suppres warnings"
    pass
import warnings
warnings.warn = warn
HR_train = pd.read_csv('../input/avhranalytics/train_jqd04QH.csv')
display_all(HR_train.head())
## Check for missing value percentage in the training data data
HR_train.isnull().sum()/len(HR_train)*100
## Check for data types
HR_train.dtypes
# How many rows and cloumns in the data set
print('Number of rows: ', HR_train.shape[0])
print('Number of columns: ', HR_train.shape[1])
## Percentage of students looking for job change
HR_train.target.value_counts()/HR_train.shape[0] *100
## Statistical analysis of numerical columns
HR_train.describe()
# Mean calculation of numerical columns grouped by target (looking for job or not)
HR_train.groupby('target').mean()
# Create a correlation matrix.
corr = HR_train.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.title('Heatmap of Correlation Matrix')
corr
# Set up the matplotlib figure
f, axes = plt.subplots(ncols=2, figsize=(15, 4))

# Graph Training hours
sns.distplot(HR_train.training_hours, kde=False, color="g", ax=axes[0]).set_title('Training hours Distribution')
axes[0].set_ylabel('Student Count')

# Graph city development index
sns.distplot(HR_train.city_development_index, kde=False, color="r", ax=axes[1]).set_title('City development index Distribution')
axes[1].set_ylabel('Student Count')
# Set up the matplotlib figure
f, axes = plt.subplots(ncols=3,nrows = 2, figsize=(20, 12))
countPlot(HR_train.last_new_job, 0, 0, 'Year diffence in last & current job')
countPlot(HR_train.relevent_experience, 0, 1, 'Relevant Experience Distribution')
countPlot(HR_train.enrolled_university, 0, 2, 'Enrolled in university')
countPlot(HR_train.education_level, 1, 0, 'Educational level')
countPlot(HR_train.major_discipline, 1, 1, 'Major discipline Distribution', rotation=45)
countPlot(HR_train.company_type, 1, 2, 'Company type Distribution', rotation=45)
f, axes = plt.subplots(ncols=2,nrows = 2, figsize=(20, 12))
countPlot(HR_train.company_type, 0, 0, 'Company type Distribution', 0,  HR_train.target)
countPlot(HR_train.last_new_job, 0, 1, 'Years difference in last & current job ', 0,  HR_train.target)
countPlot(HR_train.relevent_experience, 1, 0, 'Relevant experience', 0,  HR_train.target)
countPlot(HR_train.experience, 1, 1, 'Total years of experience', 0,  HR_train.target)
print('newbies:', HR_train[HR_train.experience.isin (['1','2','3', '<1', '4', '5'])].shape[0]), 
print('experienced:', HR_train[HR_train.experience.isin (['15','16','17', '18', '19', '20', '>20'])].shape[0])
pd.pivot_table(HR_train, values = 'target', index = 'major_discipline', columns = 'relevent_experience',aggfunc ='mean').plot.bar(figsize=(15,6))
plt.title('Average percent of students looking for a change grouped by discipline of study and relevant experience')
pd.pivot_table(HR_train, values = 'target', index = 'last_new_job', columns = 'company_type',aggfunc ='mean').plot.bar(figsize=(15,6))
plt.title('Average percent of students looking for a change grouped by company type and year difference between last and current job')
pd.pivot_table(HR_train, values = 'target', index = 'experience', columns = 'relevent_experience',aggfunc ='mean').plot.bar(figsize=(15,6))
plt.title('Average percent of students looking for a change grouped by years of experience and relevant experience')
change = HR_train[HR_train['target']==1]
change = pd.DataFrame(change.experience.value_counts()).reset_index()
total = pd.DataFrame(HR_train.experience.value_counts()).reset_index()
merge = pd.merge(change, total, how='inner', on='index')
merge = merge.rename(columns={"experience_x":'change', "experience_y":'total', "index":'experience' })

sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 7))
# plot Total cases by experience
sns.set_color_codes("pastel")
sns.barplot(x="total", y='experience', data=merge,
            label="Total", color="b")
# plot total students who are looking for a change
sns.set_color_codes("muted")
sns.barplot(x="change", y="experience", data=merge,
            label="Looking for change", color="r")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set( ylabel="Experience", title='Students per experience category',
       xlabel="# of Students")
sns.despine(left=True, bottom=True)
#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(HR_train.loc[(HR_train['target'] == 0),'training_hours'] , color='b',shade=True, label='not looking for change')
ax=sns.kdeplot(HR_train.loc[(HR_train['target'] == 1),'training_hours'] , color='r',shade=True, label='looking for change')
plt.title('Training hours Distribution - Looking for job or not')
# Replace missing values: We will replace the numerical columns with median and object column with mode.
missing_col = ['gender', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size',
              'company_type', 'last_new_job']
for col in missing_col:
    fix_missing(HR_train, HR_train[col], col, {})
# Replace ordinal categorical variables with numerics with ordered maintained
experience_mapping = {'Has relevent experience': 2, 'No relevent experience': 1}
enroll_mapping = {'no_enrollment': 1, 'Full time course': 3, 'Part time course': 2}
edu_mapping = {'Graduate': 3, 'Masters': 4, 'High School': 2, 'Phd': 5, 'Primary School': 1}
company_mapping = {'100-500': 4, '<10': 1, '50-99': 3, '5000-9999': 7, '10000+': 8, '1000-4999': 6, '500-999': 5, '10/49': 2}

HR_train['relevent_experience'] = HR_train['relevent_experience'].map(experience_mapping)
HR_train['enrolled_university'] = HR_train['enrolled_university'].map(enroll_mapping)
HR_train['education_level'] = HR_train['education_level'].map(edu_mapping)
HR_train['company_size'] = HR_train['company_size'].map(company_mapping)

# Replace with numbers in experience and last new job variables whenever necessary
HR_train.last_new_job = HR_train.last_new_job.replace({">4": 5, "never": 0}).astype('int')
HR_train.experience = HR_train.experience.replace({"<1": 0, ">20": 21}).astype('int')
# Replace cardinal variables with application of get dummies and rejoin the data after conersion
cat_var = ['company_type','major_discipline','gender','city']
num_var = [var for var in HR_train.columns if var not in cat_var ]
categorical_data = pd.get_dummies(HR_train[cat_var], drop_first=True)
numerical_data = HR_train[num_var]
HR_train = pd.concat([categorical_data,numerical_data], axis=1)
# new_HR_df.head()
# Remove ID column (not to be used in modelling) and target column
ID = HR_train.enrollee_id
target  = HR_train.target
HR_train = HR_train.drop(['enrollee_id', 'target'], axis = 1)
HR_dev, HR_val, target_dev, target_val = train_test_split(HR_train, target, test_size =0.2, random_state =1, stratify=target)
# percentage of each class
target.value_counts()/len(HR_train)*100
# Upsample minority class
HR_dev_up, target_dev_up = resample(HR_dev[target_dev == 1],
                                target_dev[target_dev == 1],
                                replace=True,
                                n_samples=HR_dev[target_dev == 0].shape[0],
                                random_state=1)

HR_dev_up = np.concatenate((HR_dev[target_dev == 0], HR_dev_up))
target_dev_up = np.concatenate((target_dev[target_dev == 0], target_dev_up))


# Upsample using SMOTE
sm = SMOTE(random_state=12)
HR_dev_sm, target_dev_sm = sm.fit_sample(HR_dev, target_dev)


# Downsample majority class
HR_dev_dn, target_dev_dn = resample(HR_dev[target_dev == 0],
                                target_dev[target_dev == 0],
                                replace=True,
                                n_samples=HR_dev[target_dev == 1].shape[0],
                                random_state=1)
HR_dev_dn = np.concatenate((HR_dev[target_dev == 1], HR_dev_dn))
target_dev_dn = np.concatenate((target_dev[target_dev == 1], target_dev_dn))


print("Original shape:", HR_dev.shape, target_dev.shape)
print("Upsampled shape:", HR_dev_up.shape, target_dev_up.shape)
print ("SMOTE sample shape:", HR_dev_sm.shape, target_dev_sm.shape)
print("Downsampled shape:", HR_dev_dn.shape, target_dev_dn.shape)
# Create the Original, Upsampled, and Downsampled training sets
methods_data = {"Original": (HR_dev, target_dev),
                "Upsampled": (HR_dev_up, target_dev_up),
                "SMOTE":(HR_dev_sm, target_dev_sm),
                "Downsampled": (HR_dev_dn, target_dev_dn)}

# Loop through each type of training sets and apply 5-Fold CV using Logistic Regression
# By default in cross_val_score StratifiedCV is used
for method in methods_data.keys():
    lr_results = cross_val_score(LogisticRegression(), methods_data[method][0], methods_data[method][1], cv=10, scoring='f1')
    print(f"The best F1 Score for {method} data:")
    print (lr_results.mean())
 
cross_val_score(LogisticRegression(class_weight='balanced'), HR_dev, target_dev, cv=10, scoring='f1').mean() 
target_baseline = np.array([0.13])
print('Baseline: AUC of validation set', roc_auc_score(target_val, target_baseline.repeat(len(target_val)))) 
print('Baseline: AUC of development set',roc_auc_score(target_dev, target_baseline.repeat(len(target_dev))))
lr = LogisticRegression()

# Fit the model to the Upsampling data
lr = lr.fit(HR_dev_sm, target_dev_sm)

print ("---Logistic Regression Model---")
lr_auc = roc_auc_score(target_val, lr.predict(HR_val))

print ("Logistic Regression AUC = %2.2f" % lr_auc)

#lr2 = lr.fit(x_train_sm, y_train_sm)
print(classification_report(target_val, lr.predict(HR_val)))
mdl = RandomForestClassifier(n_jobs=-1)

# Define grid parameters to search for best parameters
param_grid = {
    'min_samples_split': [3, 7,  11],
    'min_samples_leaf' : [3, 7, 11],
    'max_depth'        : [3,  7,  11] 
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'roc_auc_score' : make_scorer(roc_auc_score)
}

# Grid search method define
def grid_search_wrapper(refit_score='precision_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(mdl, param_grid, scoring=scorers, refit=refit_score,
                           cv=None, return_train_score=True, n_jobs=-1)
    grid_search.fit(HR_dev_sm, target_dev_sm)

    # make the predictions
    y_pred = grid_search.predict(HR_val)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the Validation data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(target_val, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search
# Check for best parameters using refit score as precision score
grid_search_mdl = grid_search_wrapper(refit_score='precision_score')
# Train the randomforest model using the best paramets from grid search
mdl = RandomForestClassifier(n_estimators = 800, n_jobs = -1,min_samples_leaf=3,max_features='auto', random_state=11,
                                      criterion='gini',  max_depth = 11, min_samples_split = 7)
mdl.fit(HR_dev_sm, target_dev_sm)
rf_roc_auc = roc_auc_score(target_val, mdl.predict_proba(HR_val)[:,1])
print('RF: AUC of validation set',roc_auc_score(target_val, mdl.predict_proba(HR_val)[:,1]))
print('RF: AUC of development set',roc_auc_score(target_dev_sm, mdl.predict_proba(HR_dev_sm)[:,1]))
errxgb = []
y_pred_tot_xgb = []

fold = StratifiedKFold(n_splits=5)
i = 1
for train_index, test_index in fold.split(HR_dev_sm, target_dev_sm):
    x_train, x_val = HR_dev_sm.iloc[train_index], HR_dev_sm.iloc[test_index]
    y_train, y_val = target_dev_sm[train_index], target_dev_sm[test_index]
    m = XGBClassifier(boosting_type='gbdt',
                      max_depth=5,
                      learning_rate=0.05,
                      n_estimators=3000,
                      random_state=194)
    m.fit(x_train, y_train,
          eval_set=[(x_train,y_train),(x_val, y_val)],
          early_stopping_rounds=400,
          eval_metric='auc',
          verbose=0)
    pred_y = m.predict_proba(x_val)[:,-1]
    print(i, " err_xgb: ", roc_auc_score(y_val,pred_y))
    errxgb.append(roc_auc_score(y_val,pred_y))
    pred_test = m.predict_proba(HR_val)[:,-1]
    i = i + 1
    y_pred_tot_xgb.append(pred_test)
xgb_auc = roc_auc_score(target_val,  np.mean(y_pred_tot_xgb, 0))
print('XGB: Mean Cross validation AUC score:', np.mean(errxgb,0))
print('XGB: AUC on Validation set:',roc_auc_score(target_val, np.mean(y_pred_tot_xgb, 0)))
# Create ROC Graph
fpr, tpr, thresholds = roc_curve(target_val, lr.predict_proba(HR_val)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(target_val, mdl.predict_proba(HR_val)[:,1])
xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(target_val, np.mean(y_pred_tot_xgb, 0))

plt.figure()

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % lr_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)

# Plot XGB ROC
plt.plot(xgb_fpr, xgb_tpr, label='Extreme Gradient Boosting Classifier (area = %0.2f)' % xgb_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()
# Get Feature Importances  (few top features)
feature_importances = pd.DataFrame(mdl.feature_importances_,
                                   index = HR_val.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances = feature_importances.reset_index()
feature_importances[0:12]
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 7))

# Plot the Feature Importance (for first 12 features)
sns.set_color_codes("pastel")
sns.barplot(x="importance", y='index', data=feature_importances[0:12],
            label="Total", color="b")
# Apply Random Noise to data set
X_train_rnoise = pd.DataFrame(HR_dev)
X_train_rnoise['RANDOM_NOISE'] = np.random.normal(0, 1, X_train_rnoise.shape[0])

# Fit Random Forest to DataSet
rf_random = RandomForestClassifier(n_estimators = 800, n_jobs = -1,min_samples_leaf=3,max_features='auto', random_state=11,
                                      criterion='gini',  max_depth = 7, min_samples_split = 7)
rf_random = rf_random.fit(X_train_rnoise, target_dev)

# Get Feature Importances
feature_importances_random = pd.DataFrame(rf_random.feature_importances_, index = X_train_rnoise.columns,columns=['importance']).sort_values('importance', ascending=False)
feature_importances_random = feature_importances_random.reset_index()

# Create Seaborn PLot
sns.set(style="whitegrid")
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 7))

clrs = ['red' ]

# Plot the Feature Importance
sns.barplot(x="importance", y='index', data=feature_importances_random[0:12],
            label="Total",  palette=clrs)
class pedict_hr_model:
    # init methods
    def __init__(self, data):
        self.data = data
        
    # replace missing values
    def impute_missing(self):
        missing_col = ['gender', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size',
              'company_type', 'last_new_job']
        for col in missing_col:
            fix_missing(self.data, self.data[col], col, {})
        return self.data
###############################################################    
def convert_cat_vars(data):
    "Replace ordinal categorical variables with numerics with ordered maintained"
     
    experience_mapping = {'Has relevent experience': 2, 'No relevent experience': 1}
    enroll_mapping = {'no_enrollment': 1, 'Full time course': 3, 'Part time course': 2}
    edu_mapping = {'Graduate': 3, 'Masters': 4, 'High School': 2, 'Phd': 5, 'Primary School': 1}
    company_mapping = {'100-500': 4, '<10': 1, '50-99': 3, '5000-9999': 7, '10000+': 8, '1000-4999': 6, '500-999': 5, '10/49': 2}

    data['relevent_experience'] = data['relevent_experience'].map(experience_mapping)
    data['enrolled_university'] = data['enrolled_university'].map(enroll_mapping)
    data['education_level'] = data['education_level'].map(edu_mapping)
    data['company_size'] = data['company_size'].map(company_mapping)

    "Replace with numbers in experience and last new job variables whenever necessary"
    data['last_new_job'] = data['last_new_job'].replace({">4": 5, "never": 0}).astype('int')
    data['experience'] = data['experience'].replace({"<1": 0, ">20": 21}).astype('int')
    
    "perform get dummies for cardinal variables"
    cat_var = ['company_type','major_discipline','gender','city']
    num_var = [var for var in data.columns if var not in cat_var ]
    categorical_data = pd.get_dummies(data[cat_var], drop_first=True)
    numerical_data = data[num_var]
    data = pd.concat([categorical_data,numerical_data], axis=1)
    return data

def train_model(data):
    "train the model"
    target  = data.target
    data = data.drop(['enrollee_id', 'target'], axis = 1)
    sm = SMOTE(random_state=12)
    HR_sm, target_sm = sm.fit_sample(data, target)
    mdl = RandomForestClassifier(n_estimators = 800, n_jobs = -1,min_samples_leaf=3,max_features='auto', random_state=11,
                                      criterion='gini', max_depth = 11, min_samples_split = 7)
    mdl.fit(data, target)
        # save the model to disk
    pickle.dump(mdl, open('HR_model.sav', 'wb'))
# Read the file, train the model and save
HR_train = pd.read_csv('../input/avhranalytics/train_jqd04QH.csv')
HR_trn = pedict_hr_model(HR_train)
HR_trn = HR_trn.impute_missing()
HR_trn = convert_cat_vars(HR_trn)
train_model(HR_trn)
def predict_HR_proba(test):
    "Predict on the new dataset"
    HR_test = pedict_hr_model(test)
    HR_test = HR_test.impute_missing()
    HR_test = convert_cat_vars(HR_test)
    HR_model = pickle.load(open('HR_model.sav', 'rb'))
    enrollee_id = HR_test.enrollee_id
    HR_test = HR_test.drop(['enrollee_id'], axis = 1)
    pred_proba = HR_model.predict_proba(HR_test)[:,1]
    output = pd.DataFrame(enrollee_id)
    output['target'] = pred_proba
    return output
# Read the new data file and predict the probability of job outlook by students
HR_test = pd.read_csv('../input/avhranalytics/test_KaymcHn.csv')
output = predict_HR_proba(HR_test)
output.to_csv('output.csv', index = False)
output.head()
