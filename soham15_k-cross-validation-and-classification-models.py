import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
df1=pd.read_csv("../input/preprocessed_loans.csv")
df1=df1.iloc[:,[1,2,4,5,6,7,10,11,12,13,14,15,17,19,20,22,24,25,26,29,30,31,33,34,35,36,37,38,39,40,41,42,61,63,65,66,67,68,70,72,73,74,75,76,78]]
df1['fico_mean']=(df1['last_fico_range_low']+df1['last_fico_range_high'])/2

df1['fico_mean'].head()

df1.drop(['last_fico_range_low','last_fico_range_high'],axis=1,inplace=True)

df1.head()
df1=df1.dropna()

len(df1)
df1["loan_status"].unique()
ls=sns.countplot(x="loan_status",data=df1)

for item in ls.get_xticklabels():

    item.set_rotation(90)
df_new=df1.loc[df1["loan_status"]!='Current',:]

ls1=sns.countplot(x="loan_status",data=df_new)

for item in ls1.get_xticklabels():

    item.set_rotation(90)
G_b=lambda x: "Bad Loan" if x=='Default' else ("Bad Loan" if x=='Charged Off' else ("Bad Loan" if x== 'Does not meet the credit policy. Status:Charged Off' else"Good Loan"))

df_new["Status"]=df_new["loan_status"].apply(G_b)

df_new["Status"]
df_new=df_new.drop("loan_status",axis=1)

df_new.head(3)
#plotting histogram to decide no. of bins

sns.distplot(df_new['fico_mean'],bins=5)
f_b=lambda x: "0-500" if x<=500  else("701-900" if x>700 else "501-700")

df_new["fico_bin"]=df_new["fico_mean"].apply(f_b)

df_new["fico_bin"]
df_new.drop("fico_mean",axis=1,inplace=True)
corr=df_new.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
df1_num=df_new.loc[:,['funded_amnt', 'annual_inc', 'dti','delinq_2yrs','out_prncp','recoveries','last_pymnt_amnt',

 'tot_coll_amt',

 'tot_cur_bal',

 'bc_util',

 'chargeoff_within_12_mths',

 'delinq_amnt',

 'mort_acc',

 'total_bc_limit',

 'total_il_high_credit_limit',

 'term',

 'int_rate',

 'revol_util']]
corr2=df1_num.corr()

sns.heatmap(corr2,xticklabels=corr2.columns,yticklabels=corr2.columns)
colnames_numerics_only = df_new.select_dtypes(include=[np.number]).columns.tolist()

colnames_numerics_only
updated_numeric_columns=['funded_amnt', 'annual_inc', 'dti','delinq_2yrs','out_prncp','recoveries','last_pymnt_amnt',

 'tot_coll_amt',

 'tot_cur_bal',

 'bc_util',

 'chargeoff_within_12_mths',

 'delinq_amnt',

 'mort_acc',

 'total_bc_limit',

 'total_il_high_credit_limit',

 'term',

 'int_rate',

 'revol_util']
from sklearn.preprocessing import StandardScaler

# Separating out the features

x = df_new.loc[:, updated_numeric_columns].values



# Standardizing the features

x = StandardScaler().fit_transform(x)
df2=pd.DataFrame(x,columns=updated_numeric_columns)
df_new=df_new.drop(colnames_numerics_only,axis=1)
df=pd.concat([df_new.reset_index(drop=True),df2],axis=1) 
df.shape
#dividing the dataset into training and hold-out dataset

from sklearn.model_selection import train_test_split

loan_train,loan_holdout=train_test_split(df,test_size=0.2)
#Seperating out the predictors and target variable of train data

#Label encoding for categorical variables; the categorical variables need to be label encoded to implement the classification models

from sklearn import preprocessing



for column in loan_train.columns:

    if loan_train[column].dtype == type(object):

        le = preprocessing.LabelEncoder()

        loan_train[column] = le.fit_transform(loan_train[column])

features=loan_train.columns.drop("Status")

target=["Status"]
#feautures and target

loan_train_features=loan_train[features]

loan_train_target=loan_train[target]
#k-cross validation only with train data

from sklearn.model_selection import StratifiedKFold

folds=StratifiedKFold(n_splits=5)

folds
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=15, random_state=42)

rnd_clf.fit(loan_train_features, loan_train_target)



features = loan_train_features

importances = rnd_clf.feature_importances_



plt.title('Feature Importances')

feat_importances = pd.Series(importances, index=loan_train_features.columns)

feat_importances.plot(kind='barh')
# Checking the accuracy score with 3-cross validation and taking all the columns 

from sklearn.model_selection import cross_val_score

cross_val_score(RandomForestClassifier(),loan_train_features,loan_train_target,cv=3)
#Random forest with only significant variables

cross_val_score(RandomForestClassifier(),loan_train_features.loc[:,['recoveries','last_pymnt_amnt','fico_bin','out_prncp','int_rate','term','sub_grade']],loan_train_target,cv=3)
#Logistic regression model

from sklearn.linear_model import LogisticRegression

cross_val_score(LogisticRegression(),loan_train_features,loan_train_target,cv=5)

#XGBoost Classification



rf_col=['recoveries','last_pymnt_amnt','fico_bin','out_prncp','int_rate','term','sub_grade']



from xgboost.sklearn import XGBClassifier



results = cross_val_score(XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.01,  

                      colsample_bytree = 0.8,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=15, 

                      reg_alpha = 0.3,

                      max_depth=4, 

                      gamma=1), loan_train[rf_col], loan_train_target, cv=3)

results
#Running Random Forest model on the whole of train dataset and validating using holdout dataset

from sklearn import preprocessing





#Label encoding the categorical columns

for column in loan_holdout.columns:

    if loan_holdout[column].dtype == type(object):

        le2 = preprocessing.LabelEncoder()

        loan_holdout[column] = le2.fit_transform(loan_holdout[column])



target2=["Status"]

rf_col=['recoveries','last_pymnt_amnt','fico_bin','out_prncp','int_rate','term','sub_grade']



loan_holdout_features=loan_holdout[rf_col]

loan_holdout_target=loan_holdout[target2]





model_final = RandomForestClassifier(n_estimators=15, random_state=42)

model_final.fit(loan_train_features[rf_col], loan_train_target)
#Confusion matrix



from sklearn.metrics import confusion_matrix

test_predictions=model_final.predict(loan_holdout_features)

test_conf_matrix=confusion_matrix(loan_holdout_target,test_predictions)

cm = pd.DataFrame(test_conf_matrix,columns=model_final.classes_,index=model_final.classes_)

cm
#0 corresponds to Bad Loan and 1 corresponds to Good Loan



cm_acc_num = cm.iloc[0,0]+cm.iloc[1,1]

cm_acc_denum = cm.loc[0,0]+cm.iloc[1,1]+cm.loc[0,1]+cm.iloc[1,0]

cm_accuracy = cm_acc_num/cm_acc_denum

cm_accuracy
cm_specificity=cm.iloc[0,0]/(cm.iloc[0,0]+cm.iloc[1,0])

cm_specificity