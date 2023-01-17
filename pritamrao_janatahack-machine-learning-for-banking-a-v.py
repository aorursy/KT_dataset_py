import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

import sklearn
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
train_df = pd.read_csv("../input/jantahack/train_fNxu4vz.csv")
test_df = pd.read_csv("../input/jantahack/test_fjtUOL8.csv")
subm = pd.read_csv("../input/jantahack/sample_submission_HSqiq1Q.csv")

train = train_df.copy()
test = test_df.copy()
train_df.info()
## Checking for missing values in terms of percentage
(train_df.isnull().sum()/len(train_df))*100
(test_df.isnull().sum()/len(train_df))*100

## Test Data also has same pattern for missing data
train_df["Interest_Rate"].value_counts().plot(kind="bar");

## Number of samples of all the three categories are not too far from one another i.e We are not dealing with imbalanced
## classification.
train_df["Home_Owner"].value_counts().plot(kind="bar");
train_df["Length_Employed"].value_counts().plot(kind="bar");

# No specific trend binning them into two or three groups might be helpful
train_df["Purpose_Of_Loan"].value_counts().plot(kind="bar");
sns.distplot(train_df["Annual_Income"]);

# The data is highly skewed applying log transformation may work 
## Applying log transformation
sns.distplot(np.log(train_df["Annual_Income"]));

# This is better as it looks like Gaussian curve 
sns.distplot(train_df["Debt_To_Income"]);

# Most of the debt to income values lies betwwen 10 to 30
train_df["Inquiries_Last_6Mo"].value_counts().plot(kind="bar");

# Most applicant haven't even enquired about the loan
# Binning categories >4 into one will reduce categories
sns.distplot(train_df["Months_Since_Deliquency"]);
## Converting loan amt as numerical column
train_df["Loan_Amount_Requested"]= train_df["Loan_Amount_Requested"].str.replace('\D+','').astype(int)
test_df["Loan_Amount_Requested"]= test_df["Loan_Amount_Requested"].str.replace('\D+','').astype(int)
sns.distplot(train_df["Loan_Amount_Requested"]);

# Loan amount req has multiple peaks.I guess there might be some categories in
# which one can apply loan like between 10k to 20k which is inducing such pattern
plt.figure(figsize=(20,10))
temp = train_df.drop("Loan_ID",axis=1)
corr = temp.corr()
sns.heatmap(corr,vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True,annot=True);
train = pd.read_csv("../input/jantahack/train_fNxu4vz.csv")
test = pd.read_csv("../input/jantahack/test_fjtUOL8.csv")
subm = pd.read_csv("../input/jantahack/sample_submission_HSqiq1Q.csv")


train_df = train.copy()
test_df = test.copy()


test_df["Interest_Rate"] = 21 # dummy value
# Joining train and test data as it will be easy to pre-process them once
joined_df = pd.concat([train_df,test_df],axis=0)
joined_df = joined_df.reset_index(drop=True)
print(f'Og Train shape {train_df.shape} \nOg Test Shape {test_df.shape} \nCombined Df {joined_df.shape}')


## Dropping loan id
joined_df.drop("Loan_ID",1,inplace=True)

## Coverting to int 
joined_df["Loan_Amount_Requested"]= joined_df["Loan_Amount_Requested"].str.replace('\D+','').astype(int)


## Because of  XGBOOST specifically since column names don't contains '<,>,]'
joined_df["Length_Employed"] = joined_df["Length_Employed"].replace(to_replace='10+ years',value="more_than_10yr")
joined_df["Length_Employed"] = joined_df["Length_Employed"].replace(to_replace='< 1 year',value="less_than_1yr")



## Numerical and cat cols
numerical = [col for col in joined_df.columns if joined_df[col].dtype!='object']
categorical = [col for col in joined_df.columns if joined_df[col].dtype =='object']
numerical.remove("Interest_Rate")


## Imuting missing values Using mode for categorical and median for numerical as it is more robust than mean
joined_df["Length_Employed"].fillna(joined_df["Length_Employed"].mode()[0],inplace=True)
joined_df["Home_Owner"].fillna(joined_df["Home_Owner"].mode()[0],inplace=True)
joined_df["Annual_Income"].fillna(joined_df["Annual_Income"].median(),inplace=True)
joined_df["Months_Since_Deliquency"].fillna(joined_df["Months_Since_Deliquency"].median(),inplace=True)



## Standard Scaling for numerical columns
scaler = StandardScaler()
numerical_data = pd.DataFrame(scaler.fit_transform(joined_df[numerical]))
numerical_data.columns = numerical


##  One Hot Encoding for categorical columns
joined_df = pd.concat([ 
            numerical_data,
            pd.get_dummies(joined_df['Length_Employed'],drop_first = True),
            pd.get_dummies(joined_df['Home_Owner'],drop_first = True),
            pd.get_dummies(joined_df['Income_Verified'],drop_first = True),
            pd.get_dummies(joined_df['Purpose_Of_Loan'],drop_first = True),
            pd.get_dummies(joined_df['Gender'],drop_first = True),
            joined_df["Interest_Rate"]
            
            ],axis=1)


## Let's separate train and test 

train_df = joined_df[:164309]
test_df = joined_df[164309:]

test_df.reset_index(drop=True,inplace=True) # since indexes were not starting from zero
test_df.drop(["Interest_Rate"],axis=1,inplace=True)
print(f'Final Train shape {train_df.shape} \nFinal Test Shape {test_df.shape}')


X_train,y_train = train_df.drop("Interest_Rate",1),train_df.loc[:,"Interest_Rate"]
print(f'Training shape {X_train.shape,y_train.shape}')
def cross_val_evaluate(model,X,y,cv,scoring,verbose,model_name):
    weighted_f1s = cross_val_score(model,X,y,cv=cv,scoring=scoring,verbose=verbose,n_jobs=-1)
    mean_weighted_f1 = round(np.sum(weighted_f1s)/cv,5)
    print(f" -----------------------{model_name}-------------------------------")
    print(f" weightedF1 for folds = {weighted_f1s}\n And Mean weighted_f1 on cv = {mean_weighted_f1}\n\n")
log = LogisticRegression()
cross_val_evaluate(log,X_train,y_train,4,"f1_weighted",4,"LOG")

knn = KNeighborsClassifier()
cross_val_evaluate(knn,X_train,y_train,4,"f1_weighted",4,"KNN")

gnb = GaussianNB()
cross_val_evaluate(gnb,X_train,y_train,4,"f1_weighted",4,"GNB")

dtc = DecisionTreeClassifier()
cross_val_evaluate(dtc,X_train,y_train,4,"f1_weighted",4,"DTC")

rfc = RandomForestClassifier()
cross_val_evaluate(rfc,X_train,y_train,4,"f1_weighted",4,"RFC")

xtc = ExtraTreesClassifier()
cross_val_evaluate(xtc,X_train,y_train,4,"f1_weighted",4,"XTC")

gbc = GradientBoostingClassifier()
cross_val_evaluate(gbc,X_train,y_train,4,"f1_weighted",4,"GBC")

xgb = XGBClassifier()
cross_val_evaluate(xgb,X_train,y_train,4,"f1_weighted",4,"XGB")

## Data Creation

train = pd.read_csv("../input/jantahack/train_fNxu4vz.csv")
test = pd.read_csv("../input/jantahack/test_fjtUOL8.csv")
subm = pd.read_csv("../input/jantahack/sample_submission_HSqiq1Q.csv")


train_df = train.copy()
test_df = test.copy()


test_df["Interest_Rate"] = 21 # dummy value
# Joining train and test data as it will be easy to pre-process them once
joined_df = pd.concat([train_df,test_df],axis=0)
joined_df = joined_df.reset_index(drop=True)
print(f'Og Train shape {train_df.shape} \nOg Test Shape {test_df.shape} \nCombined Df {joined_df.shape}')


## Dropping loan id
joined_df.drop("Loan_ID",1,inplace=True)
print(f'Shape after dropping loan id {joined_df.shape}')

## Coverting to int 
joined_df["Loan_Amount_Requested"]= joined_df["Loan_Amount_Requested"].str.replace('\D+','').astype(int)


## Because of  XGBOOST specifically since column names don't contains '<,>,]'
joined_df["Length_Employed"] = joined_df["Length_Employed"].replace(to_replace='10+ years',value="more_than_10yr")
joined_df["Length_Employed"] = joined_df["Length_Employed"].replace(to_replace='< 1 year',value="less_than_1yr")



## What is the loan amount w.r.t Loan amount requested by applicant ?

joined_df["LA_per_AI"] = np.round(joined_df["Loan_Amount_Requested"]/joined_df["Annual_Income"],4)
sns.distplot(joined_df["LA_per_AI"]);

# I thought the loan amount requested by an applicant w.r.t his/her annual income might be a useful ratio 
# to decide in which category of Interest rate an applicant falls.

# Because the smaller this ratio would be less time an applicant will need to repay the amount and vice versa .
# Hence charge interest rate accordingly  
##  creating no of close account
joined_df["no_close_accnts"] = joined_df["Total_Accounts"]-joined_df["Number_Open_Accounts"]

# How many total accounts does an applicant have w.r.t total accounts conveys a message that does this person has more
# closed account then open which is not a good sign for the banks.Because the more credit score , the more payment history
# and reliabilty is better when applying loans to decide interest rate
joined_df["Open_per_Total"] = round(joined_df["Number_Open_Accounts"]/joined_df["Total_Accounts"],4)

# Looking at the distribution of data
sns.distplot(joined_df["Open_per_Total"]);
## Income_split_per_accnt communicates amount an applicant can split into his various accounts .
## Eg if this value is very small then it conveys that this person has more accounts w.r.t his income which is not a good sign

joined_df["Income_split_per_accnt"] = round(joined_df["Annual_Income"]/(joined_df["Number_Open_Accounts"]+1),4)
sns.distplot(joined_df["Income_split_per_accnt"]);

# The data looks quite skewed hence taking log.
joined_df["Income_split_per_accnt"] = np.log(joined_df["Income_split_per_accnt"])
sns.distplot(joined_df["Income_split_per_accnt"]);

# This looks more like gaussian curve 
## Standard Scaling for numerical columns

## Numerical and cat cols
numerical = [col for col in joined_df.columns if joined_df[col].dtype!='object']
categorical = [col for col in joined_df.columns if joined_df[col].dtype =='object']
numerical.remove("Interest_Rate")

scaler = StandardScaler()
numerical_data = pd.DataFrame(scaler.fit_transform(joined_df[numerical]))
numerical_data.columns = numerical


##  One Hot Encoding for categorical columns
joined_df = pd.concat([ 
            numerical_data,
            pd.get_dummies(joined_df['Length_Employed'],drop_first = True),
            pd.get_dummies(joined_df['Home_Owner'],drop_first = True),
            pd.get_dummies(joined_df['Income_Verified'],drop_first = True),
            pd.get_dummies(joined_df['Purpose_Of_Loan'],drop_first = True),
            pd.get_dummies(joined_df['Gender'],drop_first = True),
            joined_df["Interest_Rate"]
            
            ],axis=1)


## Let's separate train and test 

train_df = joined_df[:164309]
test_df = joined_df[164309:]

test_df.reset_index(drop=True,inplace=True) # since indexes were not starting from zero
test_df.drop(["Interest_Rate"],axis=1,inplace=True)
print(f'Final Train shape {train_df.shape} \nFinal Test Shape {test_df.shape}')

## Creating a list of categorical columns which 

columns = train["Purpose_Of_Loan"].value_counts().index.tolist()+train["Length_Employed"].value_counts().index.tolist()
columns.extend(["more_than_10yr","less_than_1yr"])
columns.remove('10+ years')
columns.remove('< 1 year')
print(*columns,sep=", ")
print(f'Originally shape of train = {train_df.shape} and shape of test = {test_df.shape}')
for col in columns:
    if col in train_df.columns.to_list() and train_df[col].value_counts()[1] < 5000:
        train_df.drop(col,1,inplace=True)
        test_df.drop(col,1,inplace=True)
        
print(f'After shape of train = {train_df.shape} and shape of test = {test_df.shape}')
X_train,y_train = train_df.drop("Interest_Rate",1),train_df.loc[:,"Interest_Rate"]
print(f'Training shape {X_train.shape,y_train.shape}')

xgb = XGBClassifier(
                learning_rate=0.2,
                n_estimators=150,
                subsample=1,
                colsample_bytree=1,
                objective='multi:softprob', 
                n_jobs=-1,
                scale_pos_weight=None, 
                verbosity=3,
                max_depth=6,
                min_child_weight =14,
                gamma = 0.3
               )

xgb.fit(X_train,y_train)
preds = xgb.predict(test_df)
subm["Interest_Rate"] = preds
subm.to_csv("./xgb.csv",index=False)
plt.figure(figsize=(16,8))
plt.bar(X_train.columns.to_list(), xgb.feature_importances_);
plt.xticks(rotation=90);
