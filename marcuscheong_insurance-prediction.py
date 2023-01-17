import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns



from scipy import stats

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



sns.set_palette("pastel")

pd.options.display.float_format = "{:,.4f}".format
train=pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')

test=pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
train.head()
####################

# Null Data Analysis

####################

nullDF=pd.DataFrame()

nullDF['Train']=train.isnull().sum()

nullDF['Test']=test.isnull().sum()

nullDF
train.groupby(by=['Vehicle_Age']).count()
###########################

# Encoding Categorical Data

###########################



from sklearn.preprocessing import LabelEncoder



le=LabelEncoder()

train['Gender']=le.fit_transform(train['Gender'])

test['Gender']=le.fit_transform(test['Gender'])

print("Gender Encoding Classes:", le.classes_)



train['Vehicle_Damage']=le.fit_transform(train['Vehicle_Damage'])

test['Vehicle_Damage']=le.fit_transform(test['Vehicle_Damage'])

print("Vehicle Damage Encoding Classes:", le.classes_)



def ordered_encoding(lst,x):

    return lst.index(x)

lst = ['< 1 Year','1-2 Year','> 2 Years']

train['Vehicle_Age']=train['Vehicle_Age'].apply(lambda x : ordered_encoding(lst,x))

test['Vehicle_Age']=test['Vehicle_Age'].apply(lambda x : ordered_encoding(lst,x))



train['Region_Code']=train['Region_Code'].apply(lambda x : int(x))

test['Region_Code']=test['Region_Code'].apply(lambda x : int(x))



train['Policy_Sales_Channel']=train['Policy_Sales_Channel'].apply(lambda x : int(x))

test['Policy_Sales_Channel']=test['Policy_Sales_Channel'].apply(lambda x : int(x))



train.drop(columns=['id'])

test.drop(columns=['id'])



train.head()
X=train.drop(columns=['id','Response'])

y=train['Response']
#########################

# Distribution of Target

#########################



sns.countplot(y)

count_0, count_1 = y.value_counts()

total=count_0+count_1

percent_0=float("{:.2f}".format(count_0/total))

percent_1=float("{:.2f}".format(count_1/total))

print("Not Interested: ",count_0,f"{percent_0}%")

print("Interested:     ",count_1,f" {percent_1}%")
from imblearn.combine import SMOTETomek



smt=SMOTETomek(random_state=42)



X,y=smt.fit_sample(X,y)
###################################

# Distribution of resampled Labels

###################################



sns.countplot(y)

count_0, count_1 = y.value_counts()

total=count_0+count_1

percent_0=float("{:.2f}".format(count_0/total))

percent_1=float("{:.2f}".format(count_1/total))

print("Not Interested: ",count_0,f"{percent_0}%")

print("Interested:     ",count_1,f" {percent_1}%")
X.head()
sns.countplot(X['Gender'],hue=y)

plt.legend(labels=["not interested","interested"])



female,male=X['Gender'].value_counts()

print("Number of female:",female)

print("Number of male:",male)



plt.show()



#### Gender - Response #####

gender=X['Gender']



# Form a Contingency Table #

ctb=pd.crosstab(gender, y, normalize=True)



(chi2,p,dof,_)=stats.chi2_contingency([ctb.iloc[0].values,ctb.iloc[1].values])



gender_=['Gender',chi2,p,dof,gender.var()]

(ctb)
age_interested=(X.loc[y[y==1].index.values])['Age']

age_notinterested=(X.loc[y[y==0].index.values])['Age']



f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(X['Age'],ax=ax_box)

sns.distplot(X['Age'],ax=ax_hist)

ax_box.set(xlabel='')

ax_box.set(title="Distribution of customer's age")

plt.show()



sns.distplot(age_notinterested, color='salmon')

sns.distplot(age_interested, color='lightblue')

plt.title("Distribution of customer's age and their interest")

plt.legend(labels=["not interested","interested"])

plt.show()
sns.countplot(X['Driving_License'],hue=y)

plt.legend(labels=["not interested","interested"])



dl=X['Driving_License']



dl1,dl0 = X['Driving_License'].value_counts()

print("Number of customers that have a driving license:", dl1)

print("Number of customers that do not have a driving license:", dl0)

print("Variance:", X['Driving_License'].var())

plt.show()



# Form a Contingency Table #

ctb=pd.crosstab(dl, y)



(chi2,p,dof,_)=stats.chi2_contingency([ctb.iloc[0].values,ctb.iloc[1].values])



driving_license=['Driving_License',chi2,p,dof,dl.var()]

(ctb)
print("Variance:", X['Region_Code'].var())



f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(X['Region_Code'],ax=ax_box)

sns.distplot(X['Region_Code'],ax=ax_hist)

ax_box.set(xlabel='')

ax_box.set(title="Distribution of customer's region code")

plt.show()





rc_interested=(X.loc[y[y==1].index.values])['Region_Code']

rc_notinterested=(X.loc[y[y==0].index.values])['Region_Code']

sns.distplot(rc_notinterested, color='salmon')

sns.distplot(rc_interested, color='lightblue')

plt.title("Distribution of customer's region code and their interest")

plt.legend(labels=["not interested","interested"])

plt.show()
f, ax = plt.subplots(4,4,figsize=(29,29))



count=20



for i in range(4):

    for j in range(4):

        df=(X[['Region_Code','Policy_Sales_Channel']][X['Region_Code']==count])

        if(count==28):

            ax[i,j].hist(df['Policy_Sales_Channel'],color='black')

        else:

            ax[i,j].hist(df['Policy_Sales_Channel'])

        ax[i,j].title.set_text("Region Code: " + str(count))

        count+=1
print("Variance:",X['Previously_Insured'].var())



sns.countplot(X['Previously_Insured'],hue=y)

plt.legend(labels=["not interested","interested"])



plt.show()



#### Previously_Insured - Response #####

pi=X['Previously_Insured']



## Categorical - Categorical ##

# Form a Contingency Table #

ctb=pd.crosstab(pi, y)



(chi2,p,dof,_)=stats.chi2_contingency([ctb.iloc[0].values,ctb.iloc[1].values])



previosly_insured=['Previously_Insured',chi2,p,dof,pi.var()]

(ctb)

sns.countplot(X['Vehicle_Age'],hue=y)

plt.legend(labels=["not interested","interested"])

plt.show()



#### Vehicle_Age - Response #####

va=X['Vehicle_Age']



# Form a Contingency Table #

ctb=pd.crosstab(va, y, normalize=True)



(chi2,p,dof,_)=stats.chi2_contingency([ctb.iloc[0].values,ctb.iloc[1].values])



vehicle_age=['Vehicle_Age',chi2,p,dof,va.var()]

(ctb)
sns.countplot(X['Vehicle_Damage'],hue=y)

plt.legend(labels=["not interested","interested"])

plt.show()



#### Vehicle_Damage - Response #####

vd=X['Vehicle_Damage']



# Form a Contingency Table #

ctb=pd.crosstab(vd, y)



(chi2,p_,dof,_)=stats.chi2_contingency([ctb.iloc[0].values,ctb.iloc[1].values])

vehicle_damage=['Vehicle_Damage',chi2,p_,dof,vd.var()]

(ctb)
print("Variance:", X['Annual_Premium'].var())



f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(X['Annual_Premium'],ax=ax_box)

sns.distplot(X['Annual_Premium'],ax=ax_hist)

ax_box.set(xlabel='')

ax_box.set(title="Distribution of Annual Preium paid")

plt.show()





ap_interested=(X.loc[y[y==1].index.values])['Annual_Premium']

ap_notinterested=(X.loc[y[y==0].index.values])['Annual_Premium']



sns.distplot(ap_notinterested, color='salmon')

plt.title("Distribution of Annual Premium paid for customers that were interested")

plt.show()

sns.distplot(ap_interested, color='lightblue')

plt.title("Distribution of Annual Premium paid for customers that were not interested")

plt.show()
print("Variance:", X['Policy_Sales_Channel'].var())



f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(X['Policy_Sales_Channel'],ax=ax_box)

sns.distplot(X['Policy_Sales_Channel'],ax=ax_hist)

ax_box.set(xlabel='')

ax_box.set(title="Distribution of Policy Sales Channel")

plt.show()





psc_interested=(X.loc[y[y==1].index.values])['Policy_Sales_Channel']

psc_notinterested=(X.loc[y[y==0].index.values])['Policy_Sales_Channel']



sns.distplot(psc_notinterested, color='salmon')

plt.title("Distribution of Policy Sales Channel for customers that were interested")

plt.show()

sns.distplot(psc_interested, color='lightblue')

plt.title("Distribution of Policy Sales Channel for customers that were not interested")

plt.show()



print("Most used channel:")

print((X['Policy_Sales_Channel'].value_counts())[:3])
print("Variance:", X['Vintage'].var())



f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

sns.boxplot(X['Vintage'],ax=ax_box)

sns.distplot(X['Vintage'],ax=ax_hist)

ax_box.set(xlabel='')

ax_box.set(title="Distribution of Policy Sales Channel")

plt.show()





ap_interested=(X.loc[y[y==1].index.values])['Vintage']

ap_notinterested=(X.loc[y[y==0].index.values])['Vintage']



sns.distplot(ap_notinterested, color='salmon')

plt.title("Distribution of Policy Sales Channel for customers that were interested")

plt.show()

sns.distplot(ap_interested, color='lightblue')

plt.title("Distribution of Policy Sales Channel for customers that were not interested")

plt.show()
chi2summary=(gender_,driving_license,previosly_insured,vehicle_age,vehicle_damage)

chi2DF=pd.DataFrame(chi2summary,columns=['FeatureName','Chi2','p-val','DegreeOfFreedom','Variance'])

chi2DF
etc=ExtraTreesClassifier()

etc.fit(X,y)

feat_imp=pd.Series(etc.feature_importances_,

                  index=X.columns)

feat_imp.nlargest(10).plot(kind='barh')

plt.show()
X=X.drop(columns=['Driving_License','Gender'])

test=test.drop(columns=['id','Driving_License','Gender'])
from sklearn.model_selection import train_test_split



xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV, cross_val_score



from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score,roc_curve
#https://stackabuse.com/understanding-roc-curves-with-python/

def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()

    

    

def build_model(clf):

    classifier_name=str(clf).split('(')[0]



    clf.fit(xtrain,ytrain)

    

    ypred = clf.predict(xtest)

    

    accuracy=accuracy_score(ytest,ypred)

    probs = (clf.predict_proba(xtest))[:,1]

    auc=roc_auc_score(ytest, probs)

    kfold=cross_val_score(clf,X,y,cv=10)

    kfold_acc = kfold.mean()



    cr=(classification_report(ypred,ytest,output_dict=True))

    cr0=cr['0']

    cr1=cr['1']

    cm=confusion_matrix(ypred,ytest)

    

    summary=([classifier_name ,cr0['precision'], cr0['recall'], cr0['f1-score'], cr0['support'],cr1['precision'], cr1['recall'], cr1['f1-score'], cr1['support'], cm[0,0], cm[0,1], cm[1,0], cm[1,1], auc, kfold_acc, accuracy])



    fpr, tpr, _ = roc_curve(ytest,probs)

    plot_roc_curve(fpr, tpr)

    

    return clf,summary
# hp={

#     'criterion'         : ['gini','entropy'],

#     'min_samples_split' : [x for x in range(2,25)],

#     'max_depth'         : [x for x in range(90,100)]

# }



# dtc_tune=RandomizedSearchCV(estimator=DecisionTreeClassifier(),

#                            param_distributions = hp, 

#                            scoring='roc_auc',

#                             cv = 5, 

#                             verbose=1,  

#                             n_jobs = -1,

#                            return_train_score=True)



# dtc_tune.fit(X,y)



# dtc_tune.best_params_
# hp={

#     'criterion'         : ['entropy','gini'],

#     'max_features'      : [None, 'sqrt','log2'],

#     'min_samples_split' : [x for x in range(2,11)]

# }



# rfc_tune=RandomizedSearchCV(estimator=RandomForestClassifier(),

#                       param_distributions = hp, 

#                       cv = 5,

#                       scoring='roc_auc',

#                       verbose=1,  

#                       n_jobs = -1,

#                       return_train_score=True)



# rfc_tune.fit(X,y)

# rfc_tune.best_params_
# hp={

#     'n_estimators' : [x for x in range(50,150,10)],

#     'learning_rate': [x for x in range(1,10)]

# }



# abc_tune=RandomizedSearchCV(estimator=XGBClassifier(),

#                            param_distributions = hp, 

#                            scoring='roc_auc',

#                            cv = 5,

#                            verbose=1,

#                            n_jobs = -1,

#                            return_train_score=True)



# abc_tune.fit(X,y)

# abc_tune.best_params_
# hp={

#     'n_estimators' : [x for x in range(50,150,10)],

#     'learning_rate': [x for x in range(1,10)]

# }



# abc_tune=RandomizedSearchCV(estimator=AdaBoostClassifier(),

#                            param_distributions = hp, 

#                            scoring='roc_auc',

#                            cv = 5,

#                            verbose=1,

#                            n_jobs = -1,

#                            return_train_score=True)



# abc_tune.fit(X,y)

# abc_tune.best_params_
dtc=DecisionTreeClassifier(criterion='gini',

                          max_depth=97,

                          min_samples_split=21)



dtc_model, dtc_summary = build_model(dtc)
rfc = RandomForestClassifier(n_estimators=130)



rfc_model, rfc_summary = build_model(rfc)
xgb = XGBClassifier()



xgb_model, xgb_summary = build_model(xgb)
abc = AdaBoostClassifier(n_estimators=100)



abc_model, abc_summary = build_model(abc)
lgbm=LGBMClassifier()



lgbm_model, lgbm_summary = build_model(lgbm)
model_summary=[dtc_summary,rfc_summary,xgb_summary,abc_summary,lgbm_summary]

model_summary=pd.DataFrame(model_summary,columns=['ModelName','precision_0','recall_0','f1_score_0','support_0','precision_1','recall_1','f1_score_1','support_1','TP','FP','FN','TN','AUC','cross_val_score','Accuracy']).set_index('ModelName')

model_summary
prediction=rfc.predict(test)

submission=pd.read_csv('../input/health-insurance-cross-sell-prediction/sample_submission.csv')

submission['Response']=prediction
###################################

# Distribution of predicted Target

###################################



sns.countplot(submission['Response'])

count_0, count_1 = submission['Response'].value_counts()

total=count_0+count_1

percent_0=float("{:.2f}".format(count_0/total))

percent_1=float("{:.2f}".format(count_1/total))

print("Not Interested: ",count_0,f"{percent_0}%")

print("Interested:     ",count_1,f" {percent_1}%")