import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline



import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)
# read datasets 
train = pd.read_csv('../input/loan-prediction-practice-av-competition/train_csv.csv')  
test = pd.read_csv('../input/loan-prediction-practice-av-competition/test.csv.csv')
# drop duplicate rows, if any - no duplicates found
print(train.shape)
train =train.drop_duplicates(subset = None, keep = 'first')
print(train.shape)

print(test.shape)
test =test.drop_duplicates(subset = None, keep = 'first')
print(test.shape)
train.head()
# check for missing values in test + train dataset
print("TRAIN DATA\n")
print(train.isnull().sum())
print("")
print("TEST DATA\n")
print(test.isnull().sum())
#combine train and test for missing value treatment (not the most optimal way, unless test/train are from same source)

train['source'] = 'train'
test['source'] = 'test'
test['Loan_Status'] = np.nan # emply column for test dataset (we have to predict this)

frames = [test,train]
combined = pd.concat(frames, sort = False)
# missing values as % of nrows
per_missing = round((combined.drop('Loan_Status',axis =1).isnull().sum()/combined.drop('Loan_Status',axis = 1).isnull().count())*100,1)
print('% Missing Values for each column :')
per_missing.sort_values(ascending = False)
# replace non-numeric with mode, replace numeric with median

#median 
combined.loc[:,'Credit_History'].fillna(combined.loc[:,'Credit_History'].median(),inplace = True)
combined.loc[:,'LoanAmount'].fillna(combined.loc[:,'LoanAmount'].median(),inplace = True)
combined.loc[:,'Loan_Amount_Term'].fillna(combined.loc[:,'Loan_Amount_Term'].median(),inplace = True)

#mode
combined.loc[:,'Self_Employed'].fillna(combined.loc[:,'Self_Employed'].mode()[0],inplace = True)
combined.loc[:,'Dependents'].fillna(combined.loc[:,'Dependents'].mode()[0],inplace = True)
combined.loc[:,'Gender'].fillna(combined.loc[:,'Gender'].mode()[0],inplace = True)
combined.loc[:,'Married'].fillna(combined.loc[:,'Married'].mode()[0],inplace = True)


per_missing = round((combined.drop('Loan_Status',axis =1).isnull().sum()/combined.drop('Loan_Status',axis = 1).isnull().count())*100,1)
print('% Missing Values for each column :')
per_missing.sort_values(ascending = False)
# separate train from combined for visualization
train = combined.loc[combined.source == 'train'].copy() #add copy() to create copy instead of view

train.shape
# gender
eda_gender_proportion = round(pd.crosstab(train.loc[:,'Gender'],train.loc[:,'Loan_Status']).apply(lambda r: r/r.sum(), axis=1)*100,1)

# married
eda_married_proportion = round(pd.crosstab(train.loc[:,'Married'],train.loc[:,'Loan_Status']).apply(lambda r: r/r.sum(), axis=1)*100,1)

# dependents
eda_dependents_proportion = round(pd.crosstab(train.loc[:,'Dependents'],train.loc[:,'Loan_Status']).apply(lambda r: r/r.sum(), axis=1)*100,1)

# education
eda_education_proportion = round(pd.crosstab(train.loc[:,'Education'],train.loc[:,'Loan_Status']).apply(lambda r: r/r.sum(), axis=1)*100,1)

# employment
eda_self_employed_proportion = round(pd.crosstab(train.loc[:,'Self_Employed'],train.loc[:,'Loan_Status']).apply(lambda r: r/r.sum(), axis=1)*100,1)

# property
eda_self_property_proportion = round(pd.crosstab(train.loc[:,'Property_Area'],train.loc[:,'Loan_Status']).apply(lambda r: r/r.sum(), axis=1)*100,1)

# credit history
eda_credit_proportion = round(pd.crosstab(train.loc[:,'Credit_History'],train.loc[:,'Loan_Status']).apply(lambda r: r/r.sum(), axis=1)*100,1)
fig, axes = plt.subplots(3,3)
fig.set_size_inches(15, 15)

p1 = eda_gender_proportion.plot.bar(stacked=True,ax = axes[0,0])
p2 = eda_married_proportion.plot.bar(stacked=True,ax = axes[0,1])
p3 = eda_dependents_proportion.plot.bar(stacked=True,ax = axes[0,2])
p4 = eda_education_proportion.plot.bar(stacked=True,ax = axes[1,0])
p5 = eda_self_employed_proportion.plot.bar(stacked=True,ax = axes[1,1])
p6 = eda_self_property_proportion.plot.bar(stacked=True,ax = axes[1,2])
p7 = eda_credit_proportion.plot.bar(stacked=True,ax = axes[2,0])


p1.get_legend().remove()
p2.get_legend().remove()
#p3.get_legend().remove() #modify p3 legend to mark loan_status outside the graph
p4.get_legend().remove()
p5.get_legend().remove()
p6.get_legend().remove()
p7.get_legend().remove()

plt.subplots_adjust(hspace = 0.5)
p3.legend(loc='center left', bbox_to_anchor=(1,0.8),title = 'Loan Status') #modifying p3 legend

fig.delaxes(axes[2,1]) #deleting additional empty graph areas 
fig.delaxes(axes[2,2])
#create bins - bucket continuous variables, plot by Loan_Status 

# ApplicantIncome 

# train.loc[:,'ApplicantIncome'].max() #- 81000
# train.loc[:,'ApplicantIncome'].min() #- 150
# train.loc[:,'ApplicantIncome'].mean() #- 5403
# train.loc[:,'ApplicantIncome'].median() #- 3813

bins_applicant=[0,3000,6000,9000,81000]
group_applicant=['Low','Medium','High','Very High']
train['ApplicantIncome_bin']=pd.cut(train['ApplicantIncome'],bins_applicant,labels=group_applicant,include_lowest =True)


# print(train.groupby('ApplicantIncome_bin')['Loan_ID'].count())
# print(train.groupby('ApplicantIncome_bin')['ApplicantIncome'].mean())


# # CoapplicantIncome 

# train.loc[:,'CoapplicantIncome'].max() #- 41667
# train.loc[:,'CoapplicantIncome'].min() #- 0
# train.loc[:,'CoapplicantIncome'].mean() #- 1621
# train.loc[:,'CoapplicantIncome'].median() #- 1189

bins_coapplicant=[0,1000,2000,4000,41667]
group_coapplicant=['Low','Medium','High','Very High']
train['CoapplicantIncome_bin']=pd.cut(train.loc[:,'CoapplicantIncome'],bins_coapplicant,labels=group_coapplicant,include_lowest = True)


# print(train.groupby('CoapplicantIncome_bin')['Loan_ID'].count())
# print(train.groupby('CoapplicantIncome_bin')['CoapplicantIncome'].mean())

# # LoanAmount

# train.loc[:,'LoanAmount'].max() #- 700
# train.loc[:,'LoanAmount'].min() #- 9
# train.loc[:,'LoanAmount'].mean() #- 146
# train.loc[:,'LoanAmount'].median() #- 128

bins_loan_amt=[0,100,200,700]
group_loan_amt=['Low','Medium','High']
train['LoanAmount_bin']=pd.cut(train.loc[:,'LoanAmount'],bins_loan_amt,labels=group_loan_amt,include_lowest = True)


# print(train.groupby('LoanAmount_bin')['Loan_ID'].count())
# print(train.groupby('LoanAmount_bin')['LoanAmount'].mean())


# Loan_Amount_Term

# train.loc[:,'Loan_Amount_Term'].max() #- 480
# train.loc[:,'Loan_Amount_Term'].min() #- 12
# train.loc[:,'Loan_Amount_Term'].mean() #- 342
# train.loc[:,'Loan_Amount_Term'].median() #- 360

bins_loan_term=[0,300,480]
group_loan_term=['<300 months ','> 300 months']
train['Loan_Amount_Term_bin']=pd.cut(train.loc[:,'Loan_Amount_Term'],bins_loan_term,labels=group_loan_term,include_lowest = True)


# print(train.groupby('Loan_Amount_Term_bin')['Loan_ID'].count())
# print(train.groupby('Loan_Amount_Term_bin')['Loan_Amount_Term'].mean())

eda_applicant_proportion = round(pd.crosstab(train.loc[:,'ApplicantIncome_bin'],train.loc[:,'Loan_Status']).apply(lambda r: r/r.sum(), axis=1)*100,1)
eda_coapplicant_proportion = round(pd.crosstab(train.loc[:,'CoapplicantIncome_bin'],train.loc[:,'Loan_Status']).apply(lambda r: r/r.sum(), axis=1)*100,1)
eda_loan_amt_proportion = round(pd.crosstab(train.loc[:,'LoanAmount_bin'],train.loc[:,'Loan_Status']).apply(lambda r: r/r.sum(), axis=1)*100,1)
eda_loan_term_proportion = round(pd.crosstab(train.loc[:,'Loan_Amount_Term_bin'],train.loc[:,'Loan_Status']).apply(lambda r: r/r.sum(), axis=1)*100,1)

# print(eda_applicant_proportion)
# print(eda_coapplicant_proportion)
# print(eda_loan_amt_proportion)
# print(eda_loan_term_proportion)


fig, axes = plt.subplots(2,2)
fig.set_size_inches(15, 10)

p1 = eda_applicant_proportion.plot.bar(stacked=True,ax = axes[0,0])
p2 = eda_coapplicant_proportion.plot.bar(stacked=True,ax = axes[0,1])
p3 = eda_loan_amt_proportion.plot.bar(stacked=True,ax = axes[1,0])
p4 = eda_loan_term_proportion.plot.bar(stacked=True,ax = axes[1,1])


p1.get_legend().remove()
p2.get_legend().remove() #modify p2 legend to mark loan_status outside the graph
p3.get_legend().remove() 
p4.get_legend().remove()

plt.subplots_adjust(hspace = 0.5)
p2.legend(loc='center left', bbox_to_anchor=(1,0.8),title = 'Loan Status') #modifying p2 legend
# check data skewness 

fig, axes = plt.subplots(2,2)
fig.set_size_inches(15, 10)

cp1 = sns.distplot(train.loc[:,'ApplicantIncome'],ax = axes[0,0],kde = False)
cp2 = sns.distplot(train.loc[:,'CoapplicantIncome'],ax = axes[0,1],kde = False)
cp3 = sns.distplot(train.loc[:,'LoanAmount'],ax = axes[1,0],kde = False)
cp4 = sns.distplot(train.loc[:,'Loan_Amount_Term'], ax = axes[1,1],kde = False)

plt.subplots_adjust(hspace = 0.5)
# CoapplicantIncome has a lot of zeroes, check proportion
print('Proportion of entries where Coapplicant Income is zero:',round(train.loc[:,'CoapplicantIncome'][train.CoapplicantIncome==0].count()/train.loc[:,'CoapplicantIncome'].count()*100,1),'%')
# Feature 1 
#Total Household Income = Applicant Income + CoApplicant Income 
combined['TotalIncome'] = combined['ApplicantIncome'] + combined['CoapplicantIncome']
# Feature 2
# EMI : [P x R x (1+R)^N]/[(1+R)^N-1] ; assume 8% p.a rate of interest 

P = combined.loc[:,'LoanAmount']
R = 0.08
N = combined.loc[:,'Loan_Amount_Term']

combined['EMI'] = (P*R*(1+R)**N)/((1+R)**(N-1))

combined['EMI'].head()
# Feature 3
# Loan Amount to Income Ratio
combined['Loan_to_income'] = combined.loc[:,'LoanAmount']/combined.loc[:,'TotalIncome']
# drop extra columns 
combined = combined.drop(['Loan_ID','ApplicantIncome','CoapplicantIncome','Loan_Amount_Term'],axis = 1)
# convert Loan_Status, Source into numeric before applying encoding

combined['Loan_Status'].replace('Y',1,inplace = True)
combined['Loan_Status'].replace('N',0,inplace = True)

combined['source'].replace('train',0,inplace = True)
combined['source'].replace('test',1,inplace = True)

combined.info()
# one hot encoding 
combined_onehot = pd.get_dummies(combined)

# we only need n-1 columns ; for example, if there are two genders, M & F, if Gender_Male column is 1, we don't need another Gender_Female column as it will always be zero for this record
# drop redundant one hot encoded columns 

combined_onehot = combined_onehot.drop(['Gender_Male','Married_No','Dependents_3+','Education_Not Graduate','Self_Employed_Yes','Property_Area_Urban'],axis = 1)

combined_onehot.head()
# log transform numeric variables

# Loan Amount 
combined_onehot['LoanAmount_log'] = np.log(1+combined_onehot.loc[:,'LoanAmount'])

# Total Income 
combined_onehot['TotalIncome_log'] = np.log(1+combined_onehot.loc[:,'TotalIncome'])

# EMI 
combined_onehot['EMI_log'] = np.log(1+combined_onehot.loc[:,'EMI'])

# Loan To Income 
combined_onehot['Loan_to_income_log'] = np.log(1+combined_onehot.loc[:,'Loan_to_income'])

# drop orignal non-transformed variables
combined_onehot = combined_onehot.drop(['LoanAmount','TotalIncome','EMI','Loan_to_income'],axis = 1)
# scaling data
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

features_to_be_scaled = ['LoanAmount_log','TotalIncome_log','EMI_log','Loan_to_income_log']

combined_onehot.loc[:,features_to_be_scaled] = scaler.fit_transform(combined_onehot.loc[:,features_to_be_scaled])

print(combined_onehot.loc[:,features_to_be_scaled].mean()) # should be 0 if standard scalar, robust scalar ignores outliers for the calulations
print(combined_onehot.loc[:,features_to_be_scaled].std()) # should be 1, if standard scalar, robust scalar ignores outliers for the calulations
# separate test and train before modeling 


train_model = combined_onehot[combined_onehot['source'] == 0].copy()
train_model = train_model.drop(['source'],axis = 1).copy()

test_model = combined_onehot[combined_onehot['source'] == 1].copy()
test_model = test_model.drop(['source','Loan_Status'],axis = 1).copy()

# store feature matrix and response vector in two different datasets 
train_features = train_model.loc[:,train_model.columns != 'Loan_Status'].copy()
train_labels = np.ravel(train_model.loc[:,train_model.columns == 'Loan_Status'].copy()) #ravel() converts df y to flattened array
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

# store feature matrix and response vector in two different datasets 
train_features = train_model.loc[:,train_model.columns != 'Loan_Status'].copy()
train_labels = np.ravel(train_model.loc[:,train_model.columns == 'Loan_Status'].copy()) #ravel() converts df y to flattened array


X_train, X_validate, y_train, y_validate = train_test_split(train_features, train_labels, test_size=0.2, random_state=3)

# check classification scores of logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_validate)
y_pred_proba = logreg.predict_proba(X_validate)[:, 1]
[fpr, tpr, thr] = roc_curve(y_validate, y_pred_proba)

print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_validate, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_validate, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

# output file
pred_test = logreg.predict(test_model)
pred_test_df = pd.DataFrame()
pred_test_df['Loan_ID'] = test['Loan_ID']
pred_test_df['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(pred_test)
pred_test_df[['Loan_ID','Loan_Status']].to_csv('new_simple_logistic.csv',index=False)
from sklearn.model_selection import GridSearchCV

X = train_features
y = train_labels

param_grid = {'C': np.arange(1e-05, 3, 0.1)}
scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}

gs = GridSearchCV(LogisticRegression(), return_train_score=True,
                  param_grid=param_grid, scoring=scoring, cv=10, refit='Accuracy')

gs.fit(X, y)
results = gs.cv_results_

print('='*20)
print("best params: " + str(gs.best_estimator_))
print("best params: " + str(gs.best_params_))
print('best score:', gs.best_score_)
print('='*20)

plt.figure(figsize=(10, 10))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)

plt.xlabel("Inverse of regularization strength: C")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, param_grid['C'].max()) 
ax.set_ylim(0.35, 0.95)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_C'].data, dtype=float)

for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']): 
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = -results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = -results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else results['mean_test_%s' % scorer][best_index]
        
    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()
# output file
pred_test = gs.predict(test_model)
pred_test_df = pd.DataFrame()
pred_test_df['Loan_ID'] = test['Loan_ID']
pred_test_df['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(pred_test)
pred_test_df[['Loan_ID','Loan_Status']].to_csv('new_GSCV_logistic.csv',index=False)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

gs = gs.fit(train_features, train_labels)


print(gs.best_score_)
print(gs.best_params_)


print(gs.best_score_)
print(gs.best_params_)


# train rf model using the best_params_ from gridsearch

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='entropy', 
                             n_estimators=50,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(train_features, train_labels)
print("%.4f" % rf.oob_score_)
pd.concat((pd.DataFrame(train_features.columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
# output file
pred_test = rf.predict(test_model)
pred_test_df = pd.DataFrame()
pred_test_df['Loan_ID'] = test['Loan_ID']
pred_test_df['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(pred_test)
pred_test_df[['Loan_ID','Loan_Status']].to_csv('new_rf_GSCV.csv',index=False)
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer


param_test = {
    'n_estimators': [10,50,100],
    'max_depth': [3,5,7],
    'min_child_weight': [1,3],
    'gamma':[i/10.0 for i in range(0,5)],
    'subsample':[0.5,0.75,1],
    'colsample_bytree':[0.5,0.75,1],
    'learning_rate': [0.01, 0.05, 0.1]
}

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

gs = GridSearchCV(estimator = XGBClassifier(), 
                       param_grid = param_test, 
                       scoring=scoring,
                       iid=False,
                       cv=5, 
                       verbose = 1, 
                       refit='Accuracy',
                       n_jobs = -1)

gs.fit(train_features, train_labels)
print(gs.best_score_)
print(gs.best_params_)
# train xBG model using the best_params_ from gridsearch

# split into train and validate
X_train, X_validate, y_train, y_validate = train_test_split(train_features, train_labels, test_size=0.2, random_state=3,stratify = y)

# train xBG on train
from xgboost import XGBClassifier

xGB = XGBClassifier(colsample_bytree=0.75, 
                             gamma=0.0,
                             learning_rate=0.05,
                             max_depth=3,
                             min_child_weight=3,
                             n_estimators=10,
                             subsample = 1,
                             random_state=1,
                             n_jobs=-1)

xGB.fit(X_train, y_train)

# predict validate

y_pred = xGB.predict(X_validate)
y_pred_proba = xGB.predict_proba(X_validate)[:, 1]

[fpr, tpr, thr] = roc_curve(y_validate, y_pred_proba)
print('Train/Test split results:')
print(xGB.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_validate, y_pred))
print(xGB.__class__.__name__+" log_loss is %2.3f" % log_loss(y_validate, y_pred_proba))
print(xGB.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
# output file
pred_test = xGB.predict(test_model)
pred_test_df = pd.DataFrame()
pred_test_df['Loan_ID'] = test['Loan_ID']
pred_test_df['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(pred_test)
pred_test_df[['Loan_ID','Loan_Status']].to_csv('new_xGB_GSCV.csv',index=False)
# convert Loan_Status, Source into numeric before applying encoding

combined['Loan_Status'].replace('Y',1,inplace = True)
combined['Loan_Status'].replace('N',0,inplace = True)

combined['source'].replace('train',0,inplace = True)
combined['source'].replace('test',1,inplace = True)

combined.info()
# separate test and train before modeling 

train_model = combined[combined['source'] == 0].copy()
train_model = train_model.drop(['source'],axis = 1).copy()

test_model = combined[combined['source'] == 1].copy()
test_model = test_model.drop(['source','Loan_Status'],axis = 1).copy()

cols_mean_encoded = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
train_mean_encoded = train_model
test_mean_encoded = test_model

for col in cols_mean_encoded:
    mean = train_mean_encoded.groupby(col)['Loan_Status'].mean()
    train_mean_encoded['mean_encoded_'+col] =train_mean_encoded.loc[:,col].map(mean)
    test_mean_encoded['mean_encoded_'+col] =test_mean_encoded.loc[:,col].map(mean)

# drop original columns 

train_mean_encoded = train_mean_encoded.drop(cols_mean_encoded,axis = 1)
test_mean_encoded = test_mean_encoded.drop(cols_mean_encoded,axis = 1)

# store feature matrix and response vector in two different datasets 
train_features = train_mean_encoded.loc[:,train_mean_encoded.columns != 'Loan_Status'].copy()
train_labels = np.ravel(train_mean_encoded.loc[:,train_mean_encoded.columns == 'Loan_Status'].copy()) #ravel() converts df y to flattened array
# train, validate split
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(train_features, train_labels, test_size=0.2, random_state=3)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

gs = gs.fit(train_features, train_labels)


print(gs.best_score_)
print(gs.best_params_)

# train rf model using the best_params_ from gridsearch

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=100,
                             min_samples_split=12,
                             min_samples_leaf=5,
                             max_features='auto',
                             oob_score=True,
                             n_jobs=-1)
rf.fit(train_features, train_labels)
print("%.4f" % rf.oob_score_)
pd.concat((pd.DataFrame(train_features.columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
# output file
pred_test = rf.predict(test_mean_encoded)
pred_test_df = pd.DataFrame()
pred_test_df['Loan_ID'] = test['Loan_ID']
pred_test_df['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(pred_test)
pred_test_df[['Loan_ID','Loan_Status']].to_csv('new_rf_GSCV_encoded.csv',index=False)
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, roc_curve


param_test = {
    'n_estimators': [10,50,100],
    'max_depth': [3,5,7],
    'min_samples_split': [50,100,500],
    'max_features':['sqrt'],
    'subsample':[0.8],
    'colsample_bytree':[0.5,1],
    'learning_rate': [0.05, 0.1],
    'random_state': [1]
}

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

gs = GridSearchCV(estimator = XGBClassifier(), 
                       param_grid = param_test, 
                       scoring=scoring,
                       iid=False,
                       cv=5, 
                       verbose = 1, 
                       refit='Accuracy',
                       n_jobs = -1)

gs.fit(train_features, train_labels)
print(gs.best_score_)
print(gs.best_params_)
# train xBG model using the best_params_ from gridsearch
from sklearn.metrics import make_scorer, accuracy_score, roc_curve,log_loss,auc
# split into train and validate
X_train, X_validate, y_train, y_validate = train_test_split(train_features, train_labels, test_size=0.2, random_state=3,stratify = y)

# train xBG on train
xGB = XGBClassifier(colsample_bytree=1, 
                             gamma=0.0,
                             learning_rate=0.01,
                             max_depth=3,
                             min_child_weight=3,
                             n_estimators=50,
                             subsample = 1,
                             n_jobs=-1)

xGB.fit(X_train, y_train)

# predict validate

y_pred = xGB.predict(X_validate)
y_pred_proba = xGB.predict_proba(X_validate)[:, 1]

[fpr, tpr, thr] = roc_curve(y_validate, y_pred_proba)
print('Train/Test split results:')
print(xGB.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_validate, y_pred))
print(xGB.__class__.__name__+" log_loss is %2.3f" % log_loss(y_validate, y_pred_proba))
print(xGB.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
# output file
pred_test = xGB.predict(test_mean_encoded)
pred_test_df = pd.DataFrame()
pred_test_df['Loan_ID'] = test['Loan_ID']
pred_test_df['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(pred_test)
pred_test_df[['Loan_ID','Loan_Status']].to_csv('new_xGB_GSCV_encoded.csv',index=False)