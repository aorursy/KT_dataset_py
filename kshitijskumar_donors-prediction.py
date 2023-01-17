### reading data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
raw_train=pd.read_csv("../input/donorsprediction/Raw_Data_for_train_test.csv")
raw_test=pd.read_csv("../input/donorsprediction/Predict_donor.csv")
raw_train.head()
train_df=raw_train.copy()
train_df
train_df.drop("TARGET_D",axis=1,inplace=True)
train_df
train_df.columns
train_df=train_df[['CONTROL_NUMBER', 'MONTHS_SINCE_ORIGIN', 'DONOR_AGE',
       'IN_HOUSE', 'URBANICITY', 'SES', 'CLUSTER_CODE', 'HOME_OWNER',
       'DONOR_GENDER', 'INCOME_GROUP', 'PUBLISHED_PHONE', 'OVERLAY_SOURCE',
       'MOR_HIT_RATE', 'WEALTH_RATING', 'MEDIAN_HOME_VALUE',
       'MEDIAN_HOUSEHOLD_INCOME', 'PCT_OWNER_OCCUPIED', 'PER_CAPITA_INCOME',
       'PCT_ATTRIBUTE1', 'PCT_ATTRIBUTE2', 'PCT_ATTRIBUTE3', 'PCT_ATTRIBUTE4',
       'PEP_STAR', 'RECENT_STAR_STATUS', 'RECENCY_STATUS_96NK',
       'FREQUENCY_STATUS_97NK', 'RECENT_RESPONSE_PROP', 'RECENT_AVG_GIFT_AMT',
       'RECENT_CARD_RESPONSE_PROP', 'RECENT_AVG_CARD_GIFT_AMT',
       'RECENT_RESPONSE_COUNT', 'RECENT_CARD_RESPONSE_COUNT',
       'MONTHS_SINCE_LAST_PROM_RESP', 'LIFETIME_CARD_PROM', 'LIFETIME_PROM',
       'LIFETIME_GIFT_AMOUNT', 'LIFETIME_GIFT_COUNT', 'LIFETIME_AVG_GIFT_AMT',
       'LIFETIME_GIFT_RANGE', 'LIFETIME_MAX_GIFT_AMT', 'LIFETIME_MIN_GIFT_AMT',
       'LAST_GIFT_AMT', 'CARD_PROM_12', 'NUMBER_PROM_12',
       'MONTHS_SINCE_LAST_GIFT', 'MONTHS_SINCE_FIRST_GIFT', 'FILE_AVG_GIFT',
       'FILE_CARD_GIFT','TARGET_B']]

train_df
train_df.info()
### Dealing with missing values
train_df["DONOR_AGE"]
train_df["DONOR_AGE"].fillna(np.around(train_df["DONOR_AGE"].mean()),inplace=True)
def income_group(df):
    null_income_group_df=df[df["INCOME_GROUP"].isnull()]
    not_null_income_group_df=df[df["INCOME_GROUP"].notnull()]
    for groups in df["URBANICITY"].unique():
        i=null_income_group_df[null_income_group_df["URBANICITY"]==groups].index
        val=not_null_income_group_df[not_null_income_group_df["URBANICITY"]==groups]["INCOME_GROUP"].mode()[0]
        df["INCOME_GROUP"][i]=val
        
    return df
train_df=income_group(train_df)
train_df.info()
### creating a copy of the train_df for a checkpoint
train2_df=train_df.copy()
train2_df["WEALTH_RATING"].value_counts()
def wealth_rating(df):
    df["WEALTH_RATING"].replace({0:"low",1:"low",2:"low",3:"med",4:"med",5:"med",6:"med",7:"upper",8:"upper",9:"upper"},inplace=True)
    null_wealth_rating_df=df[df["WEALTH_RATING"].isnull()]
    not_null_wealth_rating_df=df[df["WEALTH_RATING"].notnull()]
    
    for groups in df["INCOME_GROUP"].unique():
        i=null_wealth_rating_df[null_wealth_rating_df["INCOME_GROUP"]==groups].index
        val=not_null_wealth_rating_df[not_null_wealth_rating_df["INCOME_GROUP"]==groups]["WEALTH_RATING"].mode()[0]
        df["WEALTH_RATING"][i]=val
        
    return df
train2_df=wealth_rating(train2_df)
train2_df.info()
def last_prom_resp(df):
    null_lpr_df=df[df["MONTHS_SINCE_LAST_PROM_RESP"].isnull()]
    not_null_lpr_df=df[df["MONTHS_SINCE_LAST_PROM_RESP"].notnull()]
    for groups in df["MONTHS_SINCE_LAST_GIFT"].unique():
        i=null_lpr_df[null_lpr_df["MONTHS_SINCE_LAST_GIFT"]==groups].index
        val=not_null_lpr_df[not_null_lpr_df["MONTHS_SINCE_LAST_GIFT"]==groups]["MONTHS_SINCE_LAST_PROM_RESP"].mean()
        df["MONTHS_SINCE_LAST_PROM_RESP"][i]=val
        
    return df
train3_df=last_prom_resp(train2_df)
train3_df.info()
cleaned_train=train3_df.copy()
### cleaning test data
test_df=raw_test.copy()
test_df.head()
test_df.info()
test_df["DONOR_AGE"].fillna(np.around(test_df["DONOR_AGE"].mean()),inplace=True)
test_df=income_group(test_df)
test_df=wealth_rating(test_df)
test_df=last_prom_resp(test_df)
test_df.info()
cleaned_test=test_df.copy()
### working on categorical values
categorical_cols=["IN_HOUSE","URBANICITY","SES","HOME_OWNER","DONOR_GENDER",
                    "INCOME_GROUP","PUBLISHED_PHONE","OVERLAY_SOURCE","WEALTH_RATING",
                    "PEP_STAR","RECENT_STAR_STATUS","RECENCY_STATUS_96NK","CLUSTER_CODE"]
replacing_dict={".":0}
for i in range(54):
    if i<16:
        replacing_dict[str(i)]=0
    elif i<31:
        replacing_dict[str(i)]=1
    elif i<45:
        replacing_dict[str(i)]=2
    else:
        replacing_dict[str(i)]=3
cleaned_train["CLUSTER_CODE"].replace(replacing_dict,inplace=True)
cleaned_test["CLUSTER_CODE"].replace(replacing_dict,inplace=True)
cleaned_train["CLUSTER_CODE"].value_counts()
### dummy variables and standardizing
from sklearn.preprocessing import StandardScaler
def dummy_and_standard(df):
    
    dummy_df=pd.get_dummies(df[categorical_cols],drop_first=True)
    scaler=StandardScaler()
    scaler.fit(df.drop(categorical_cols,axis=1))
    scaled_variables=scaler.transform(df.drop(categorical_cols,axis=1))
    scaled_df=pd.DataFrame(data=scaled_variables,columns=df.drop(categorical_cols,axis=1).columns)
    final_df=pd.concat([scaled_df,dummy_df],axis=1)
    
    return final_df
final_train=dummy_and_standard(cleaned_train.drop(["TARGET_B","CONTROL_NUMBER"],axis=1))
final_train=pd.concat([final_train,cleaned_train["TARGET_B"]],axis=1)
final_train
final_test=dummy_and_standard(cleaned_test.drop(["CONTROL_NUMBER"],axis=1))
final_test
### correlation between dependent and independent variables
final_train.corr()["TARGET_B"]
independent_features=["FREQUENCY_STATUS_97NK","RECENT_RESPONSE_COUNT","RECENT_CARD_RESPONSE_COUNT","RECENT_RESPONSE_PROP",
                     "FILE_CARD_GIFT","PEP_STAR","RECENT_CARD_RESPONSE_PROP","LIFETIME_GIFT_COUNT","RECENCY_STATUS_96NK_S",
                     "MONTHS_SINCE_LAST_GIFT","RECENT_AVG_GIFT_AMT","MONTHS_SINCE_LAST_PROM_RESP","LIFETIME_AVG_GIFT_AMT","FILE_AVG_GIFT"]
dependent_feature=["TARGET_B"]
for cols in independent_features:
    plt.plot(final_train[cols],final_train["TARGET_B"],"o")
    plt.xlabel(cols)
    plt.ylabel("TARGET_B")
    plt.show()
final_train[independent_features].corr()
### train test data
x_train=final_train[independent_features].drop(["RECENT_CARD_RESPONSE_COUNT","RECENT_RESPONSE_PROP","LIFETIME_GIFT_COUNT","FILE_AVG_GIFT"],axis=1)
y_train=final_train["TARGET_B"]
x_test=final_test[independent_features].drop(["RECENT_CARD_RESPONSE_COUNT","RECENT_RESPONSE_PROP","LIFETIME_GIFT_COUNT","FILE_AVG_GIFT"],axis=1)
print("train:",x_train.shape,"test:",x_test.shape)
from sklearn.feature_selection import f_regression
f_regression(x_train,y_train)
### dealing with imbalanced data
x_train.head()
y_train.head()
y_train.value_counts()
### Balancing both the classes using oversampling technique - SMOTE
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=21)
bal_x_train,bal_y_train=sm.fit_sample(x_train,y_train)
bal_y_train.value_counts()
print("balanced:",bal_x_train.shape,"unbalanced:",x_train.shape)
### model selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
log=LogisticRegression()
knn=KNeighborsClassifier()
rfc=RandomForestClassifier()
### hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
### KNeighboursClassifier
knn_params={"n_neighbors":[5,10,15,20,30,40,50,100,200],"leaf_size":[10,20,30,40,50,100,200,500]}
knn_search=RandomizedSearchCV(estimator=knn,
                             param_distributions=knn_params,
                             scoring="precision",
                             cv=10,
                             verbose=2)
knn_search.fit(bal_x_train,bal_y_train)
knn_search.best_score_
knn_best_params=knn_search.best_params_
knn_best_params
knn=KNeighborsClassifier(n_neighbors=knn_best_params["n_neighbors"],leaf_size=knn_best_params["leaf_size"])
### RandomForestClassifier
rfc_params={"n_estimators":[10,20,50,100,500,1000],"min_samples_split":[1,2,3,4,5,10,20],
            "min_samples_leaf":[1,2,3,4,5,10,20]}
rfc_search=RandomizedSearchCV(estimator=rfc,
                             param_distributions=rfc_params,
                             cv=5,
                             scoring="precision",
                             verbose=2)
rfc_search.fit(bal_x_train,bal_y_train)
rfc_search.best_score_
rfc_best_params=rfc_search.best_params_
rfc_best_params
rfc=RandomForestClassifier(n_estimators=rfc_best_params["n_estimators"],
                          min_samples_split=rfc_best_params["min_samples_split"],
                          min_samples_leaf=rfc_best_params["min_samples_leaf"])
### cross validation
from sklearn.model_selection import cross_val_score
log_score=cross_val_score(log,bal_x_train,bal_y_train,cv=10,scoring="precision",verbose=2)
knn_score=cross_val_score(knn,bal_x_train,bal_y_train,cv=10,scoring="precision",verbose=2)
rfc_score=cross_val_score(rfc,bal_x_train,bal_y_train,cv=5,scoring="precision",verbose=2)
print("log:",log_score.mean())
print("knn:",knn_score.mean())
print("rfc:",rfc_score.mean())
### looking at the cross_val_score, RandomForestClassifier seems to be a better model for the problem
### training model
rfc.fit(bal_x_train,bal_y_train)
rfc.score(bal_x_train,bal_y_train)
### Prediction
prediction=rfc.predict(x_test)
prediction_df=pd.DataFrame()
prediction_df["CONTROL_NUMBER"]=cleaned_test["CONTROL_NUMBER"]
prediction_df["PREDICTION"]=prediction
prediction_df
prediction_df["PREDICTION"].value_counts()
