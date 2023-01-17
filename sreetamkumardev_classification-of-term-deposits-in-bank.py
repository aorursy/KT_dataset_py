# Importing libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from imblearn.combine import SMOTETomek

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression 

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import RFE

from sklearn.metrics import accuracy_score, precision_score, classification_report, recall_score, confusion_matrix, make_scorer, fbeta_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV

%matplotlib inline





#my_details 

__author__ = "sreetam dev"

__email__  = "sreetamkumardev@gmail.com"
#loading the data 

df_bank = pd.read_csv("../input/bank-data/bank-additional-full.csv", sep = ";")

df_bank.head()
df_bank.info()   #fetching data types and length of  data entries
df_bank.isnull().any()   # searching if there are any null values that require imputation. 
# looking for duplicate instances within our dataset

print("The total no of duplicate records within our dataset are: ", df_bank.duplicated().sum())
# Removing the duplicated rows from the dataset



df_bank = df_bank.drop_duplicates()
# Observing distribution of the target variable



df_bank_subcribers_rate     = (len(df_bank[df_bank['y']== 'yes'])/ df_bank.shape[0])*100

df_bank_non_subcribers_rate = (len(df_bank[df_bank['y']== 'no'])/ df_bank.shape[0])*100

print("Percentage of subscribers{}, Percentage of non-subscribers {}".format( df_bank_subcribers_rate, df_bank_non_subcribers_rate))

# fetching summary of numerical and categorical features



df_bank.describe(include = [np.number])

df_bank.describe(include = ['O']) # descriptive statistics for categorical features
df_bank_nr_features     = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']



for feature in df_bank_nr_features:

    y = df_bank[feature]

    plt.figure(figsize = (25,10))

#     plt.subplot(1,2,1)

    sns.boxplot(y)

#     plt.subplot(1,2,2)

#     sns.distplot(y, bins =20)

    plt.show()

   
df_bank_nr_features = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']



for feature in df_bank_nr_features:

    stat_feature = df_bank[feature].describe()

#     print(stat_feature)

    IQR   = stat_feature['75%'] - stat_feature['25%']  # finding the IQR range 

    upper = stat_feature['75%'] + 1.5 * IQR  # 3rd quartile

    lower = stat_feature['25%'] - 1.5 * IQR  # 1st quartile

    print('For the feature {} the upper boundary is {} and lower boundary is {}'.format(feature,upper, lower))
df_bank[(df_bank.age < 9.5)]
df_bank[(df_bank.age > 69.5)]
df_bank[(df_bank.duration < -223.5)]
df_bank[(df_bank.duration > 4000)]
df_bank[(df_bank.campaign < -2.0)]
df_bank[(df_bank.campaign > 40)]
df_bank[(df_bank.pdays < 200)]
df_bank[(df_bank['cons.conf.idx'] > -30)& (df_bank.y == 'yes')]
# As campaign feature adds skewness and doesnot posses any confirmed term deposits with only few instances above the "40".So we can drop those feature considering to be outlier.

df_bank = df_bank[(df_bank.campaign < 40)]

# Since duration affects our target as per documentation obtained from source

df_bank = df_bank.drop('duration',axis =1)
# Now, converting out target variable y in to boolean values using encoding.

df_bank['y'] = df_bank.y.map({'yes':1, 'no':0})

# fetching numerical variables other than "duration"



df_bank_nr_features = ['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
# functions for implementing boxplots and correlation for the numerical explanatory features to understand their distribution with respect to the  term deposit



def boxplots_features_target(size, target, features, data):

    plt.figure(figsize = size)

    for each in range(len(df_bank_nr_features)):

        plt.subplot(5,3, each+1)

        sns.boxplot(x = target, y = features[each], data = data)

        

def crossCorrelation(data):

    corr = data.corr()

    plt.figure(figsize = (10,6))

    sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values)

    print(corr)

    
# creating boxplots with respect to target variable:

boxplots_features_target((30,30), 'y' , df_bank_nr_features, df_bank )
# cross correlation plot

crossCorrelation(df_bank)
# Note: this section executes well within jupyter, while there is an issue within Kaggle terminal.

# #subsetting our response variable 

# term_subscribers = df_bank["y"] == 1

# term_non_subscribers = df_bank["y"] == 0



# #creating plots

# df_bank_nr_features

# labels = ["subscribed", "not subscribed"]



# def creating_dist_target_feature_plots(df_bank_nr_features,labels):

#     plt.figure(figsize = (30,16))

#     for feature in df_bank_nr_features:

#         plt.subplot(3,3, df_bank_nr_features.index(feature)+1)

#         sns.distplot(df_bank[feature][term_subscribers], label = labels[0], color = "b")

#         sns.distplot(df_bank[feature][term_non_subscribers], label = labels[1], color = "y")

#         plt.axvline(df_bank[feature][term_subscribers].mean(), linestyle = '--', color = "b")

#         plt.axvline(df_bank[feature][term_non_subscribers].mean(), linestyle = '--', color = "y")

#         plt.legend()

        

# creating_dist_target_feature_plots(df_bank_nr_features,labels)
# plotting bar plots of our categorical features to see count instances of categorical values



df_bank_cat_features = ["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"]



for cat in df_bank_cat_features:

    cat_table = pd.crosstab(index = df_bank[cat],columns = "count")

    cat_table.plot.bar(figsize=(10,4))

    print("the mean for the category {} is: {}".format( cat,round(np.mean(cat_table["count"]),2)))

    print("the standard deviation for the category {} is: {}".format( cat,round((cat_table["count"].std()),2)))

    

#As, we had seen above in the distribution plots we observed right skewness in "campaign"



plt.figure(figsize = (5,1))

sns.distplot(df_bank["campaign"], bins =20 , color = "b")

#As, we had seen above in the distribution plots we observed right skewness in "age"

plt.figure(figsize = (5,1))

sns.distplot(df_bank["age"], bins =20, color = "y")
df_bank_processed = df_bank.copy()
#applying log transformation to handle the skewness



df_bank_processed["age"] = np.log(df_bank_processed["age"]+1)

df_bank_processed["campaign"] = np.log(df_bank_processed["campaign"]+1)
plt.figure(figsize = (10,6))

sns.distplot(df_bank_processed["age"], bins =20, color = "y")
plt.figure(figsize = (10,6))

sns.distplot(df_bank_processed["campaign"], bins =20, color = "b")
scaler = MinMaxScaler() #initiating a scaler and applying features to it

df_bank_processed[df_bank_nr_features] = scaler.fit_transform(df_bank_processed[df_bank_nr_features]) # applying noramlisation to numerical variables



df_bank_processed.head(10)
#grouping education feature containing basic into one common Basic.

df_bank_processed['education'] = np.where(df_bank_processed['education']=='basic.9y','Basic',df_bank_processed['education'])



df_bank_processed['education'] = np.where(df_bank_processed['education']=='basic.6y','Basic',df_bank_processed['education'])



df_bank_processed['education'] = np.where(df_bank_processed['education']=='basic.4y','Basic',df_bank_processed['education'])
# encoding categorical values with get_dummies method to utilise categorical features for the purpose of modelling

df_bank_processed_cat = ['job', 'marital', 'education', 'default', 'housing', 'loan','contact', 'month', 'day_of_week',  'poutcome']

df_bank_processed = pd.get_dummies(data = df_bank_processed, columns = df_bank_processed_cat )
#splitting oour data sets into train and test features.



X,y = df_bank_processed.drop(["y"],1).values, df_bank_processed["y"].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)

# check the recorded instances of Train and test data sets for X nd y



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
#using logisitic regression unoptimised version to find f score and accuracy



parameters = {"C":[0.001, 0.01, 0.1, 1, 10, 100, 1000]}



log_model_before_smote = LogisticRegression(random_state = 42, penalty = "l2")



fbeta_scorer = make_scorer(fbeta_score, beta = 0.5) #fbeta score



grid_item = GridSearchCV(log_model_before_smote, param_grid = parameters, scoring = fbeta_scorer) #grid search on the classsifier using 'scorer' as the scoring method



grid_fit = grid_item.fit(X_train, y_train) #fitting grid search to training data and find optimal parameters using fit()



best_estimators = grid_fit.best_estimator_ #get the estimator



best_predictions = best_estimators.predict(X_test) #predictions on unoptimised model



print("\nUnoptimised Model\n----")

print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test,best_predictions)))

print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test,best_predictions, beta = 0.5)))

print(best_estimators)

# using parameters obtained above to make an optimised logistitc model



parameters = {"C":[0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6 ]}

log_model_before_smote = LogisticRegression(random_state = 42, penalty = "l2")



fbeta_scorer = make_scorer(fbeta_score, beta = 0.5) #fbeta score



grid_item = GridSearchCV(log_model_before_smote, param_grid = parameters, scoring = fbeta_scorer) #grid search on the classsifier using 'scorer' as the scoring method



grid_fit = grid_item.fit(X_train, y_train) #fitting grid search to training data and find optimal parameters using fit()



best_estimators = grid_fit.best_estimator_ #get the estimator



best_predictions = best_estimators.predict(X_test) #predictions on unoptimised model



print("\nOptimised Model\n----")

print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test,best_predictions)))

print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test,best_predictions, beta = 0.5)))

print(best_estimators)
# Now, using the model with optimised parameters to predict and get the confusion matrix



model_log = LogisticRegression(C=1, random_state=42,penalty = "l2")

model_log.fit(X_train, y_train)

y_pred_log = model_log.predict(X_test)



conf_matrix = confusion_matrix(y_pred_log,y_test)

print(conf_matrix)

print(classification_report(y_test, y_pred_log))

# since earlier we had found that our response variable classes are imbalanced so we decided to use Smote and Tomek method to perform mix of oversmapling and undersampling to balance both outcomes across our response variable.

X_columns = df_bank_processed.drop(["y"],1).columns

smote_tomek  = SMOTETomek(sampling_strategy = 'auto')

X_smt, y_smt = smote_tomek.fit_sample(X_train, y_train)



df_X_smt = pd.DataFrame(data = X_smt, columns = X_columns)

df_y_smt = pd.DataFrame(data = y_smt, columns = ['y'])





#print statements to check 



print("length of oversampled data is ",len(df_y_smt))

print("percentage of subscription ", (len(df_y_smt[df_y_smt['y']== 1])/len(df_y_smt))*100)

print("percentage of no subscription ",(len(df_y_smt[df_y_smt['y']== 0])/len(df_y_smt))*100)

#Now, performing logistic regression with sampled features 



model_lr = LogisticRegression(random_state = 42, penalty = "l2")

model_rfc = RandomForestClassifier(n_estimators= 200, max_features= 'auto', max_depth= 20 , criterion= 'gini')



list_model = [model_lr, model_rfc]

for model in list_model:

    rfe = RFE(model,n_features_to_select = 20)

    X_smt_rfe = rfe.fit_transform(X_smt, y_smt)

    X_test_rfe = rfe.transform(X_test)

    # model  = model.fit(X_smt_rfe,y_smt)

    no_stratified_folds = StratifiedKFold(n_splits = 5, random_state= 1 )

    crossval_score_model = cross_val_score(model,X_smt_rfe ,y_smt, scoring = 'accuracy', cv = no_stratified_folds,n_jobs= 1, error_score='raise'  )

    print("Accuracy for model {} is : {}".format(model,np.mean(crossval_score_model)))

    print("Standard deviation for model {} is : {}".format(model,np.std(crossval_score_model)))
#So we are, performing Random forest classification with sampled features 



model_rfc = RandomForestClassifier(n_estimators= 200, max_features= 'auto', max_depth= 20 , criterion= 'gini')



rfe = RFE(model_rfc,n_features_to_select = 5)

X_smt_rfe = rfe.fit_transform(X_smt, y_smt)

X_test_rfe = rfe.transform(X_test)

model_rfc.fit(X_smt_rfe,y_smt)

y_pred = model_rfc.predict(X_test_rfe)

conf_matrix_rfe = confusion_matrix(y_test,y_pred)

print(conf_matrix_rfe)

print(classification_report(y_test, y_pred))
def metrics_model(y_test, y_pred):

    conf_matrix_rfe = confusion_matrix(y_test,y_pred)

    TP = conf_matrix_rfe[1,1]

    FN = conf_matrix_rfe[1,0]

    FP = conf_matrix_rfe[0,1]

    TN = conf_matrix_rfe[0,0]

    

    #printing confusion matrix

    

    print("confusion matrix:\n",conf_matrix_rfe)

    

    #print the accuracy score

    print("Accuracy:", round(accuracy_score(y_test, y_pred),2))

    

    #print the sensitivity/recall/true positive rate

    print("Sensitivity:", round(recall_score(y_test, y_pred),2))

    

    #precision/positive predictive value

    print("Precision:", round(precision_score(y_test, y_pred),2))

    

    

    

    
print(metrics_model(y_test, y_pred))
# let's see if increaing threshold could reduce false positives



y_pred_prob = model_rfc.predict_proba(X_test_rfe)[:,1]

y_pred_threshold = np.where(y_pred_prob< 0.45, 0 , 1)
metrics_model(y_test, y_pred_threshold)
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title("ROC Curve for Ad Classifier")

plt.xlabel("False Positive Rate (1 - Specificity)")

plt.ylabel("True Positive Rate (Sensitivity)")

plt.grid(True)
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_threshold)

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title("ROC Curve for Ad Classifier")

plt.xlabel("False Positive Rate (1 - Specificity)")

plt.ylabel("True Positive Rate (Sensitivity)")

plt.grid(True)
# We had ealrier kept a dataframe of the sampled outcome, which has been used now to fetch the feature names that have been selected by rfe.

columns = df_X_smt.columns

val = pd.Series(rfe.support_,index = columns)

features_chosen_rfe = val[val==True].index 

print(features_chosen_rfe)
potential_features = ['age', 'campaign', 'cons.conf.idx', 'euribor3m', 'nr.employed']



for feature in potential_features:

    y = df_bank[feature][df_bank['y']== 1]

    print("Statistical description of the feature {}: {}".format(feature,df_bank[feature].describe()))

    plt.figure(figsize = (10,3))

    sns.distplot(y, bins =20)

    plt.show()

    
