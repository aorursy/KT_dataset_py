#imports and settings

import pandas as pd 

pd.set_option('display.expand_frame_repr', False)

import numpy as np

from matplotlib import pyplot as plt 

plt.style.use('fivethirtyeight')

from patsy import dmatrices

%pylab inline

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/loan.csv', low_memory = False)
#Returns a random sample of items - 30% so the dataset is easy to work with

df_sample = df.sample(frac = 0.3)

df_sample.head(2)
#Analyzing target variable - loan_status

df_sample['loan_status'].value_counts()
#getting rid of loans with statuses we do not care about

#we do not care about current loans 

#explanation of difference between charged off and default https://help.lendingclub.com/hc/en-us/articles/216127747

#we only care about those loans that are either fully paid or are

#very late

#too little examples with "does not meet the credit policy" to care about these...

mask = df_sample['loan_status'].isin(['Fully Paid','Charged Off','Default'])

df_sample = df_sample[mask]

df_sample['loan_status'].value_counts()

# now we only work with loans that are either fully paid or late > 121 days

# We create target variable with these two possible values. Positive class

# are late loans - we care about these and want to analyze in detail. 



def CreateTarget(status): 

    if status == 'Fully Paid':

        return 0

    else:

        return 1

    

df_sample['Late_Loan'] = df_sample['loan_status'].map(CreateTarget)

df_sample['Late_Loan'].value_counts()
#drop features with more than 25% missing values

features_missing_series = df_sample.isnull().sum() > len(df_sample)/10

features_missing_series = features_missing_series[features_missing_series == True]

features_missing_list =  features_missing_series.index.tolist()

df_sample = df_sample.drop(features_missing_list,axis =1)



# drop features that have no or little predictive power and original target

df_sample = df_sample.drop(['id','member_id','loan_status','url','zip_code','policy_code','application_type','last_pymnt_d','last_credit_pull_d','verification_status','pymnt_plan','funded_amnt','funded_amnt_inv','sub_grade','out_prncp','out_prncp_inv','total_pymnt_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_amnt','initial_list_status','earliest_cr_line'],axis =1)



#replace missing values with Unknown value or mean when feature is numerical

df_sample['emp_title'].fillna('Unknown',inplace = True)

df_sample['title'].fillna('Unknown',inplace = True)

df_sample['revol_util'].fillna(df_sample['revol_util'].mean(),inplace = True)

df_sample['collections_12_mths_ex_med'].fillna(df_sample['collections_12_mths_ex_med'].mean(),inplace = True)



df_sample.isnull().sum() #there are no missing values left 
#old categorical emp_length feature

df_sample['emp_length'].value_counts()
#new numerical emp_length feature



def EmpLength(year):

    if year == '< 1 year':

        return 0.5

    elif year == 'n/a': #assuming that if not filled out employments was < 1 

        return 0.5

    elif year == '10+ years':

        return 10

    else:

        return float(year.rstrip(' years'))

        

df_sample['emp_length_num'] = df_sample['emp_length'].map(EmpLength)

df_sample = df_sample.drop('emp_length',axis =1 )

df_sample['emp_length_num'].value_counts()
#transforming to date datatype

df_sample['issue_d'] = pd.to_datetime(df_sample.issue_d)
#datatypes of features 

# object = string ? 

df_sample.dtypes.value_counts()
#cleaned and transformed data ready for analysis and ML 

#numerical features - means 

print(df_sample.select_dtypes(include=['float64']).apply(np.mean).apply(str))



# categorical variables 

print(df_sample.select_dtypes(include=['object']).columns)



# target variable - boolean



print(df_sample.select_dtypes(include=['bool']).columns)



df_sample['purpose'].value_counts()
#distribution of our class/targer variable Late_Loan , True if loan is late. 

plt.figure(figsize=(5,5))

df_sample['Late_Loan'].value_counts().plot(kind = 'pie',autopct='%.0f%%', startangle=100, fontsize=17)

plt.show()
Amount_By_Year = df_sample.groupby(df_sample['issue_d'].dt.year)['loan_amnt'].mean()

Amount_By_Year = pd.DataFrame(Amount_By_Year)

Amount_By_Year['YoY Change %'] = Amount_By_Year.pct_change()*100

Amount_By_Year.rename(columns = {'loan_amnt':'Average Loan Amount'})
plt.figure(figsize=(8,3))

Amount_By_Year_Status_True = df_sample.groupby([df_sample['issue_d'].dt.year,df_sample['Late_Loan'][df_sample['Late_Loan'] == True]])['loan_amnt'].mean().plot(kind = 'line', label = 'True')

Amount_By_Year_Status_False = df_sample.groupby([df_sample['issue_d'].dt.year,df_sample['Late_Loan'][df_sample['Late_Loan'] == False]])['loan_amnt'].mean().plot(kind = 'line',label = 'False')

plt.xlabel('Year')

plt.ylabel('Average Loan')

plt.legend(loc='best')

plt.show()

#This graph normalizes the purpose variables, using value counts.

#This gives us an idea of what percentage and purpose the clients are taking out loans for.

all_rows = df_sample['purpose']

pur = df_sample['purpose'].value_counts()

purp = pur/len(all_rows)

purp.plot(kind='bar')
flat = pur/len(df_sample['purpose']) * 100

print(flat)
#I'm going to look at the installment payments against the grade and term of the loan.

#This pivot table shows the installment payments by grade and term.

loan_g = pd.pivot_table(df_sample,

                           index= ['grade','term'],

                           columns= ['installment'] ,

                            values= 'loan_amnt',

                           aggfunc = sum)

loan_g.T.idxmax()

loan_g.T.idxmax().plot(kind='bar')
#This graph looks at the people who are late on their loans, renters and people paying their mortages tend to be late on payments

#A possible reason may be that the owners, others, and none may not have a financial burden.

late = df_sample[['home_ownership', 'Late_Loan']]

late_people = late['Late_Loan']== True

people = late[late_people]

sad = people['home_ownership'].value_counts().plot(kind='bar', color= 'orange')

xlabel('Living Status')

ylabel('People With Late Loans')
print(df_sample['int_rate'].mean())

mask_delinq = df_sample['delinq_2yrs'] <= 11

df_sample.groupby(df_sample['delinq_2yrs'][mask_delinq])['int_rate'].mean().plot(kind='line')

plt.show()



df_sample['delinq_2yrs'].value_counts()

mask_pub_rec = df_sample['pub_rec'] <= 6

df_sample.groupby(df_sample['pub_rec'][mask_pub_rec])['int_rate'].mean().plot(kind='line')

plt.show()
mask_pub_rec = df_sample['pub_rec'] <= 6

df_sample.groupby(df_sample['pub_rec'][mask_pub_rec])['annual_inc'].mean().plot(kind='line')

plt.show()
from patsy import dmatrices

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn import metrics

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE



df_sample['installmentAsPercent'] = df_sample['installment']/(df['annual_inc']/12)



def GradeInt(x):

    if x == "A":

        return 1

    elif x == "B":

        return 2

    elif x == "C":

        return 3

    elif x == "D":

        return 4

    elif x == "E":

        return 5

    else:

        return 6



df_sample['GradeInt'] = df_sample['grade'].map(GradeInt)





Y,X = dmatrices('Late_Loan ~ 0 + int_rate + GradeInt + loan_amnt + installment + annual_inc + dti + delinq_2yrs + inq_last_6mths + open_acc + pub_rec + revol_bal + revol_util + total_acc + collections_12_mths_ex_med + acc_now_delinq + emp_length_num + term  + home_ownership + purpose + installmentAsPercent',df_sample, return_type = 'dataframe')

X_columns = X.columns



sm = SMOTE(random_state=42)

X, Y = sm.fit_sample(X, Y)

X = pd.DataFrame(X,columns=X_columns)
#distribution of our class/targer variable Late_Loan , True if loan is late. 



Y_df = pd.DataFrame(Y,columns=['Late_Loan'])

plt.figure(figsize=(5,5))

Y_df['Late_Loan'].value_counts().plot(kind = 'pie',autopct='%.0f%%', startangle=100, fontsize=17)

plt.show()
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

best_model = []



for i in range(1,15):

    model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=i)    

    kfold = StratifiedKFold(n_splits= 10, shuffle = True)

    scores = cross_val_score(model, X_train, Y_train, cv = kfold )

    best_model.append(scores.mean())



plt.plot(range(1,15),best_model)

plt.show()
from sklearn.grid_search import RandomizedSearchCV

from scipy.stats import randint as sp_randint

param_dist = {    "max_depth": sp_randint(6,10),

                  "max_features": sp_randint(3,15),

                  "max_leaf_nodes": [10,20,30,40,50],

                  "min_samples_leaf": [25,50,75,100,150,250,500],

                 }

random_search = RandomizedSearchCV(model,

                        param_distributions=param_dist,

                        n_iter=50)



random_search.fit(X_train, Y_train)

print(random_search.best_score_)

print(random_search.best_estimator_)
best_model = random_search.best_estimator_

best_model.fit(X_train,Y_train)



importance = sorted(zip(map(lambda x: round(x, 4), best_model.feature_importances_), X.columns),reverse=True)

y_val = []

x_val = [x[0] for x in importance]

for x in importance:

    y_val.append(x[1])

    

pd.Series(x_val,index=y_val)[:7].plot(kind='bar')

plt.show()
from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(best_model, random_state=1)

bagging.fit(X,Y)



accuracy = metrics.accuracy_score(bagging.predict(X_test),Y_test)

precision = metrics.precision_score(bagging.predict(X_test),Y_test,pos_label=1)

recall = metrics.recall_score(bagging.predict(X_test),Y_test,pos_label=1)

confusion_matrix = metrics.confusion_matrix(Y_test,bagging.predict(X_test),labels=[1,0])



print(accuracy)

print(precision)

print(recall)

print(confusion_matrix)



labels = ['Late', 'Paid']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(confusion_matrix)

fig.colorbar(cax)

plt.title('Confusion matrix of the classifier')

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()