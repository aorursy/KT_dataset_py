# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, roc_curve, roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from xgboost import XGBClassifier
from xgboost import plot_importance
from hashlib import sha256
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

# Plot settings
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
sns.set()
# Load applications data
train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
train.head()
# Print info about each column in the train dataset
for col in train:
    print(col)
    Nnan = train[col].isnull().sum()
    print('Number empty: ', Nnan)
    print('Percent empty: ', 100*Nnan/train.shape[0])
    print(train[col].describe())
    if train[col].dtype==object:
        print('Categories and Count:')
        print(train[col].value_counts().to_string(header=None))
    print()
# Print info about each column in the test dataset
for col in test:
    print(col)
    Nnan = test[col].isnull().sum()
    print('Number empty: ', Nnan)
    print('Percent empty: ', 100*Nnan/test.shape[0])
    print(test[col].describe())
    if test[col].dtype==object:
        print('Categories and Count:')
        print(test[col].value_counts().to_string(header=None))
    print()
# Show target distribution
train['TARGET'].value_counts()
for col in test:
    if test[col].dtype==object:
        print(col)
        print('Num Unique in Train:', train[col].nunique())
        print('Num Unique in Test: ', test[col].nunique())
        print('Unique in Train:', sorted([str(e) for e in train[col].unique().tolist()]))
        print('Unique in Test: ', sorted([str(e) for e in test[col].unique().tolist()]))
        print()
# Merge test and train into all application data
train_o = train.copy()
train['Test'] = False
test['Test'] = True
test['TARGET'] = np.nan
app = train.append(test, ignore_index=True)
# Remove entries with gender = XNA
app = app[app['CODE_GENDER'] != 'XNA']
# Remove entries with income type = maternity leave
app = app[app['NAME_INCOME_TYPE'] != 'Maternity leave']
# Remove entries with unknown family status
app = app[app['NAME_FAMILY_STATUS'] != 'Unknown']
app['DAYS_EMPLOYED'].hist()
plt.xlabel('DAYS_EMPLOYED')
plt.ylabel('Count')
plt.show()
# Show distribution of reasonable values
app.loc[app['DAYS_EMPLOYED']<200000, 'DAYS_EMPLOYED'].hist()
plt.xlabel('DAYS_EMPLOYED (which are less than 200,000)')
plt.ylabel('Count')
plt.show()
# Show all unique outlier values
app.loc[app['DAYS_EMPLOYED']>200000, 'DAYS_EMPLOYED'].unique()
# Set unreasonable values to nan
app['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
app['PROPORTION_LIFE_EMPLOYED'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']
app['INCOME_TO_CREDIT_RATIO'] = app['AMT_INCOME_TOTAL'] / app['AMT_CREDIT'] 
app['INCOME_TO_ANNUITY_RATIO'] = app['AMT_INCOME_TOTAL'] / app['AMT_ANNUITY']
app['INCOME_TO_ANNUITY_RATIO_BY_AGE'] = app['INCOME_TO_ANNUITY_RATIO'] * app['DAYS_BIRTH']
app['CREDIT_TO_ANNUITY_RATIO'] = app['AMT_CREDIT'] / app['AMT_ANNUITY']
app['CREDIT_TO_ANNUITY_RATIO_BY_AGE'] = app['CREDIT_TO_ANNUITY_RATIO'] * app['DAYS_BIRTH']
app['INCOME_TO_FAMILYSIZE_RATIO'] = app['AMT_INCOME_TOTAL'] / app['CNT_FAM_MEMBERS']
# Create map from categories to polar projection
DOW_map = {
    'MONDAY':    0,
    'TUESDAY':   1,
    'WEDNESDAY': 2,
    'THURSDAY':  3,
    'FRIDAY':    4,
    'SATURDAY':  5,
    'SUNDAY':    6,
}
DOW_map1 = {k: np.cos(2*np.pi*v/7.0) for k, v in DOW_map.items()}
DOW_map2 = {k: np.sin(2*np.pi*v/7.0) for k, v in DOW_map.items()}

# Show encoding of days of week -> circle
days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
tt = np.linspace(0, 2*np.pi, 200)
xx = np.cos(tt)
yy = np.sin(tt)
plt.plot(xx,yy)
plt.gca().axis('equal')
plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.title('2D Projection of days of the week')
for day in days:
    plt.text(DOW_map1[day], DOW_map2[day], day, ha='center')
plt.show()
# WEEKDAY_APPR_PROCESS_START to polar coords
col = 'WEEKDAY_APPR_PROCESS_START'
app[col+'_1'] = app[col].map(DOW_map1)
app[col+'_2'] = app[col].map(DOW_map2)
app.drop(columns=col, inplace=True)
# Add indicator columns for empty values
for col in app:
    if col!='Test' and col!='TARGET':
        app_null = app[col].isnull()
        if app_null.sum()>0:
            app[col+'_ISNULL'] = app_null
# Label encoder
le = LabelEncoder()

# Label encode binary fearures in training set
for col in app: 
    if col!='Test' and col!='TARGET' and app[col].dtype==object and app[col].nunique()==2:
        if col+'_ISNULL' in app.columns: #missing values here?
            app.loc[app[col+'_ISNULL'], col] = 'NaN'
        app[col] = le.fit_transform(app[col])
        if col+'_ISNULL' in app.columns: #re-remove missing vals
            app.loc[app[col+'_ISNULL'], col] = np.nan
# Get categorical features to encode
cat_features = []
for col in app: 
    if col!='Test' and col!='TARGET' and app[col].dtype==object and app[col].nunique()>2:
        cat_features.append(col)

# One-hot encode categorical features in train set
app = pd.get_dummies(app, columns=cat_features)
# Hash columns
hashes = dict()
for col in app:
    hashes[col] = sha256(app[col].values).hexdigest()
    
# Get list of duplicate column lists
Ncol = app.shape[1] #number of columns
dup_list = []
dup_labels = -np.ones(Ncol)
for i1 in range(Ncol):
    if dup_labels[i1]<0: #if not already merged,
        col1 = app.columns[i1]
        t_dup = [] #list of duplicates matching col1
        for i2 in range(i1+1, Ncol):
            col2 = app.columns[i2]
            if ( dup_labels[i2]<0 #not already merged
                 and hashes[col1]==hashes[col2] #hashes match
                 and app[col1].equals(app[col2])): #cols are equal
                #then this is actually a duplicate
                t_dup.append(col2)
                dup_labels[i2] = i1
        if len(t_dup)>0: #duplicates of col1 were found!
            t_dup.append(col1)
            dup_list.append(t_dup)
        
# Merge duplicate columns
for iM in range(len(dup_list)):
    new_name = 'Merged'+str(iM)
    app[new_name] = app[dup_list[iM][0]].copy()
    app.drop(columns=dup_list[iM], inplace=True)
    print('Merged', dup_list[iM], 'into', new_name)
# Split data back into test + train
train = app.loc[~app['Test'], :]
test = app.loc[app['Test'], :]

# Make SK_ID_CURR the index
train.set_index('SK_ID_CURR', inplace=True)
test.set_index('SK_ID_CURR', inplace=True)

# Ensure all data is stored as floats
train = train.astype(np.float32)
test = test.astype(np.float32)

# Target labels
train_y = train['TARGET']

# Remove test/train indicator column and target column
train.drop(columns=['Test', 'TARGET'], inplace=True)
test.drop(columns=['Test', 'TARGET'], inplace=True)

# Classification pipeline
xgb_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', XGBClassifier())
])

# Cross-validated AUROC
auroc_scorer = make_scorer(roc_auc_score, needs_proba=True)
scores = cross_val_score(xgb_pipeline, train, train_y, 
                         cv=3, scoring=auroc_scorer)
print('Mean AUROC:', scores.mean())

# Fit to training data
xgb_fit = xgb_pipeline.fit(train, train_y)

# Predict default probabilities of test data
test_pred = xgb_fit.predict_proba(test)

# Save predictions to file
df_out = pd.DataFrame()
df_out['SK_ID_CURR'] = test.index
df_out['TARGET'] = test_pred[:,1]
df_out.to_csv('xgboost_baseline.csv', index=False)
# Predict probabilities for the training data
train_pred = cross_val_predict(xgb_pipeline, 
                               train, 
                               y=train_y,
                               method='predict_proba')
train_pred = train_pred[:,1] #only want p(default)

# Show calibration curve
fraction_of_positives, mean_predicted_value = \
    calibration_curve(train_y, train_pred, n_bins=10)
plt.figure()
plt.plot([0, 1], [0, 1], 'k:', 
         label='Perfectly Calibrated')
plt.plot(mean_predicted_value, 
         fraction_of_positives, 's-',
         label='XGBoost Predictions')
plt.legend()
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration curve for baseline XGBoost model')
plt.show()
# Classification pipeline w/ isotonic calibration
calib_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', CalibratedClassifierCV(
                        base_estimator=XGBClassifier(),
                        method='isotonic'))
])

# Classification pipeline w/ sigmoid calibration
sig_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', CalibratedClassifierCV(
                        base_estimator=XGBClassifier(),
                        method='sigmoid'))
])

# Predict probabilities w/ isotonic calibration
calib_pred = cross_val_predict(calib_pipeline, 
                               train, 
                               y=train_y,
                               method='predict_proba')
calib_pred = calib_pred[:,1] #only want p(default)

# Predict probabilities w/ sigmoid calibration
sig_pred = cross_val_predict(sig_pipeline, 
                             train, 
                             y=train_y,
                             method='predict_proba')
sig_pred = sig_pred[:,1] #only want p(default)

# Show calibration curve
fop_calib, mpv_calib = \
    calibration_curve(train_y, calib_pred, n_bins=10)
fop_sig, mpv_sig = \
    calibration_curve(train_y, sig_pred, n_bins=10)
plt.figure()
plt.plot([0, 1], [0, 1], 'k:', 
         label='Perfectly Calibrated')
plt.plot(mean_predicted_value, 
         fraction_of_positives, 's-',
         label='XGBoost Predictions')
plt.plot(mpv_calib, fop_calib, 's-',
         label='Calibrated Predictions - isotonic')
plt.plot(mpv_sig, fop_sig, 's-',
         label='Calibrated Predictions - sigmoid')
plt.legend()
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration curve for Calibrated XGBoost model')
plt.show()

# Cross-validated AUROC for isotonic
print('Mean AUROC with isotonic calibration:', 
      roc_auc_score(train_y, calib_pred))

# Cross-validated AUROC for sigmoid
print('Mean AUROC with sigmoid calibration:',
      roc_auc_score(train_y, sig_pred))
# Fit to the training data
calib_fit = calib_pipeline.fit(train, train_y)

# Predict default probabilities of the test data
test_pred = calib_fit.predict_proba(test)

# Save predictions to file
df_out = pd.DataFrame()
df_out['SK_ID_CURR'] = test.index
df_out['TARGET'] = test_pred[:,1]
df_out.to_csv('xgboost_calibrated.csv', index=False)
# Show distribution of target variable
sns.countplot(x='TARGET', data=app)
plt.title('Number of applicants who had trouble repaying')
plt.show()
# A sampler that doesn't re-sample!
class DummySampler(object):
    def sample(self, X, y):
        return X, y
    def fit(self, X, y):
        return self
    def fit_sample(self, X, y):
        return self.sample(X, y)
    
# List of samplers to test
samplers = [
    ['Oversampling', RandomOverSampler()], 
    ['Undersampling', RandomUnderSampler()], 
    ['SMOTE', SMOTE()],
    ['No resampling', DummySampler()]
]

# Preprocessing pipeline
pre_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('imputer', SimpleImputer(strategy='median'))
])

# Classifier
classifier = CalibratedClassifierCV(
                        base_estimator=XGBClassifier(),
                        method='isotonic')

# Compute AUROC and plot ROC for each type of sampler
plt.figure()
auroc_scorer = make_scorer(roc_auc_score, needs_proba=True)
cv = StratifiedKFold(n_splits=3)
for name, sampler in samplers:
    
    # Make the sampling and classification pipeline
    pipeline = make_pipeline(sampler, calib_pipeline)

    # Cross-validated predictions on training set
    probas = np.zeros(train.shape[0]) # to store predicted probabilities
    for tr, te in cv.split(train, train_y):
        test_pre = pre_pipeline.fit_transform(train.iloc[te])  #preprocess test fold
        train_pre = pre_pipeline.fit_transform(train.iloc[tr]) #preprocess training fold
        train_s, train_y_s = sampler.fit_sample(train_pre, train_y.iloc[tr]) #resample train fold
        probas_ = classifier.fit(train_s, train_y_s).predict_proba(test_pre) #predict test fold
        probas[te] = probas_[:,1]
    
    # Print AUROC value
    print(name, 'AUROC:', roc_auc_score(train_y, probas))
    
    # Plot ROC curve for this sampler
    fpr, tpr, threshs = roc_curve(train_y, probas)
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], label='Chance')
plt.legend()
plt.show()
# Fit XGBoost model on the training data
train_pre = pre_pipeline.fit_transform(train) #preprocess training data
model = XGBClassifier()
model.fit(train, train_y)

# Show feature importances
plt.figure(figsize=(6, 15))
plot_importance(model, height=0.5, ax=plt.gca())
plt.show()
# Show default probability by gender
plt.figure()
sns.barplot(x='CODE_GENDER', y="TARGET", data=train_o)
plt.show()