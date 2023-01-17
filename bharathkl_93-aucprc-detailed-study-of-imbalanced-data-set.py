# Import libraries necessary for this project

import numpy as np

import pandas as pd

from time import time

from IPython.display import display # Allows the use of display() for DataFrames

import matplotlib.pyplot as plt

from sklearn.utils import resample

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.metrics import fbeta_score, recall_score, precision_score, average_precision_score, precision_recall_curve

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

import seaborn as sns

from collections import Counter

from sklearn.grid_search import GridSearchCV 

from sklearn.metrics import make_scorer



# Pretty display for notebooks

%matplotlib inline





# Load the Census dataset



data = pd.read_csv("../input/creditcard.csv")



# Success - Display the first record

print("Data has {} record with {} features".format(data.shape[0], data.shape[1]))

display(data.head(n=1))

results = {}
# Create figure

fig = plt.figure(figsize = (12,10));

for i, feature in enumerate(['Class', 'Amount']):

    ax = fig.add_subplot(2, 2, i+1)

    ax.hist(data[feature], bins = 25, color = '#00A0A0')

    ax.set_title("'%s'"%(feature), fontsize = 14)

    ax.set_xlabel("Value")

    ax.set_ylabel("Number of Records")

        

   

    fig.tight_layout()

    fig.show()
# Visualizing the Amount.

data['Amount'] = data['Amount'].replace(np.nan,0)

data['Amount'] = data['Amount'].apply(lambda x: np.log(x + 1))

plt.hist(data['Amount'], bins = 25, color = '#00A0A0')

plt.title("'%s'"%(feature), fontsize = 14)

plt.xlabel("Value")

plt.ylabel("Number of Records")
from sklearn import preprocessing

print("                           Data Before normalizing is given below:                       ")

display(data.head(n=1))





scaler = preprocessing.MinMaxScaler()



data['Amount']  = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

#data['Time'] = preprocessing.MinMaxScaler().fit_transform(data['Time'].reshape(-1, 1))



#features[numerical]  = np.log(features[numerical])



print("                           Data After normalizing is given below:                        ")

display(data.head(n=1))
plt.hist(data['Amount'], bins = 25, color = '#00A0A0')

plt.title("'%s'"%(feature), fontsize = 14)

plt.xlabel("Value")

plt.ylabel("Number of Records")
#Function to split the original data and rturn values basd on the flag

def split_data(flag):

       

    y = data.Class

    X = data.drop(['Time', 'Amount', 'Class'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y, 

                                                    test_size = 0.3, 

                                                    random_state = 0)

        

    if flag == 1:

        return (X,y)

    elif flag  == 2:

        return (X_train, X_test, y_train, y_test)

    else:

        return (X, y, X_train, X_test, y_train, y_test)
#Copying the original data

resample_data = data.drop(['Time', 'Amount'], axis=1)

#samples_n = (resample_data['Class' == 1]).sum()



# Initializing the dictionary to store the classifier metrics 

results['oversampled']={}



print("No. of 0's and 1's in the feature Class before oversampling the data")

print(resample_data.Class.value_counts())





# Separate majority and minority classes

data_majority = resample_data[resample_data.Class == 0]

data_minority = resample_data[resample_data.Class == 1]

 

# Upsample minority class

data_minority_oversampled  = resample(data_minority, 

                                 replace=True,     

                                 n_samples=284315, 

                                 random_state=123) 

 

# Combine majority class with upsampled minority class

data_oversampled = pd.concat([data_majority, data_minority_oversampled])



print("No. of 0's and 1's in the feature Class after oversampling the data")

 

print(data_oversampled.Class.value_counts())



y = data_oversampled.Class

X = data_oversampled.drop('Class', axis=1)



# Split the data into training and testing sets

X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X, 

                                                    y, 

                                                    test_size = 0.3, 

                                                    random_state = 0)



    

print("Training set has {} samples.".format(X_train_o.shape[0]))

print("Testing set has {} samples.".format(X_test_o.shape[0]))



start = time()

#Training the Classifier

clf_over_sampled = LogisticRegression().fit(X_train_o, y_train_o)

end = time()

results['oversampled']['train_time'] = end - start



# Predict on training set

start = time()

y_pred_score_o = clf_over_sampled.predict(X_test_o)

end = time()

results['oversampled']['pred_time'] = end - start



results['oversampled']['fbeta'] = fbeta_score(y_test_o,y_pred_score_o,beta=2)

results['oversampled']['recall']= recall_score(y_test_o,y_pred_score_o)

results['oversampled']['precision'] = precision_score(y_test_o,y_pred_score_o)





print ("Train Time:", results['oversampled']['train_time'])

print ("Prediction Time:", results['oversampled']['pred_time'])

print ("fbeta score:", results['oversampled']['fbeta'])

print('recall_score:', results['oversampled']['recall'])

print('precision_score:', results['oversampled']['precision'])

    

# copying the orginal data  

resample_data = data.drop(['Time', 'Amount'], axis=1)



# Initializing the dictionary to store the classifier metrics 

results['undersampled']={}



#Separate majority and minority classes

print("No. of 0's and 1's in the feature Class before undersampling the data")

print(resample_data.Class.value_counts())



data_majority = resample_data[resample_data.Class==0]

data_minority = resample_data[resample_data.Class==1]

 

# Downsample majority class

data_majority_undersampled = resample(data_majority, 

                                 replace=False,   

                                 n_samples=492,   

                                 random_state=5) 

 

# Combine minority class with downsampled majority class

data_undersampled = pd.concat([data_majority_undersampled, data_minority])

 

# Display new class counts

print("No. of 0's and 1's in the feature Class after undersampling the data" )

print(data_undersampled.Class.value_counts())



y = data_undersampled.Class

X = data_undersampled.drop('Class', axis=1)



# Split the data into training and testing sets

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X, 

                                                    y, 

                                                    test_size = 0.3, 

                                                    random_state = 0)



    

print("Training set has {} samples.".format(X_train_u.shape[0]))

print("Testing set has {} samples.".format(X_test_u.shape[0]))



start = time()

# Training the classifier

clf_under_sampled = LogisticRegression().fit(X_train_u, y_train_u)

end = time()

results['undersampled']['train_time'] = end - start



# Predict the testing set

start = time()

y_pred_score_u = clf_under_sampled.predict(X_test_u)

end = time()

results['undersampled']['pred_time'] = end - start



results['undersampled']['fbeta'] = fbeta_score(y_test_u,y_pred_score_u,beta=2)

results['undersampled']['recall'] = recall_score(y_test_u,y_pred_score_u)

results['undersampled']['precision'] = precision_score(y_test_u,y_pred_score_u)





print ("Train Time:", results['undersampled']['train_time'])

print ("Prediction Time:", results['undersampled']['pred_time'])

print ("fbeta score:", results['undersampled']['fbeta'])

print('recall_score:', results['undersampled']['recall'])

print('precision_score:', results['undersampled']['precision'])

# Getting the original features and labels as X,y from split_data function

X_sm,y_sm = split_data(1)

print("No. of 0's and 1's in the feature Class before oversampling the data")

print(Counter(y_sm))



start = time()

# Oversampling the data using SMOTE

X_resampled_sm, y_resampled_sm = SMOTE().fit_sample(X_sm,y_sm)

end = time()

print("No. of 0's and 1's in the feature Class After oversampling the data")

print(Counter(y_resampled_sm))



# Initializng the dictionary to store performance metrics

results['SMOTE'] = {}

results['SMOTE']['resample_time'] = end - start



# Splitting the resampled data 

X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_resampled_sm, 

                                                    y_resampled_sm, 

                                                    test_size = 0.3, 

                                                    random_state = 0)

start = time()

# Training the Classifier

clf_smote = LogisticRegression().fit(X_train_sm, y_train_sm)

end = time()

results['SMOTE']['train_time'] = end - start





# Predict on training set

start = time()

y_pred_score_sm = clf_smote.predict(X_test_sm)

end = time()

results['SMOTE']['pred_time'] = end - start





results['SMOTE']['fbeta'] = fbeta_score(y_test_sm,y_pred_score_sm,beta=2)

results['SMOTE']['recall'] = recall_score(y_test_sm,y_pred_score_sm)

results['SMOTE']['precision'] = precision_score(y_test_sm,y_pred_score_sm)



print ("Train Time:", results['SMOTE']['train_time'])

print ("Prediction Time:", results['SMOTE']['pred_time'])

print ("fbeta score:", results['SMOTE']['fbeta'])

print('recall_score:', results['SMOTE']['recall'])

print('precision_score:', results['SMOTE']['precision'])



# Getting the original features and labels as X,y from split_data function

X_nm,y_nm = split_data(1)

print("No. of 0's and 1's in the feature Class before undersampling the data")

print(Counter(y_nm))



#Initializing the Nearmiss Classifier

nm1 = NearMiss(random_state=0, version=1)

start = time()

# Undersampling the data

X_resampled_nm1, y_resampled_nm1 = nm1.fit_sample(X_nm, y_nm)

end = time()



print("No. of 0's and 1's in the feature Class after undersampling the data")

print(Counter(y_resampled_nm1))



#Initializing the dictionary to store the classifier metrics

results['NearMiss'] = {}

results['NearMiss']['resample_time'] = end - start



#Splitting the undersampled data

X_train_nm, X_test_nm, y_train_nm, y_test_nm = train_test_split(X_resampled_nm1, 

                                                    y_resampled_nm1, 

                                                    test_size = 0.3, 

                                                    random_state = 0)

#Training the classifier 

start = time()

clf_nm = LogisticRegression().fit(X_train_nm, y_train_nm)

end = time()

results['NearMiss']['train_time'] = end - start





# Predict on training set

start = time()

y_pred_score_nm = clf_nm.predict(X_test_nm)

end = time()

results['NearMiss']['pred_time'] = end - start





results['NearMiss']['fbeta'] = fbeta_score(y_test_nm,y_pred_score_nm,beta=2)

results['NearMiss']['recall'] = recall_score(y_test_nm,y_pred_score_nm)

results['NearMiss']['precision'] = precision_score(y_test_nm,y_pred_score_nm)



print ("Train Time:", results['NearMiss']['train_time'])

print ("Prediction Time:", results['NearMiss']['pred_time'])

print ("fbeta score:", results['NearMiss']['fbeta'])

print('recall_score:', results['NearMiss']['recall'])

print('precision_score:', results['NearMiss']['precision'])

# split orginal data into train & test 

X_train_dt, X_test_dt, y_train_dt, y_test_dt = split_data(2)



#Initializing Dictionary to store output of decision tree classifier metrics

results['Decision_Tree'] = {}



#Initialize classfier

clf_DT = DecisionTreeClassifier(random_state=5)



start = time()

# Train the Classifier

clf_DT.fit(X_train_dt, y_train_dt)

end = time()

results['Decision_Tree']['train_time'] = end - start



start = time()

# Test the classifier

y_pred_score_DT = clf_DT.predict(X_test_dt)

end = time()

results['Decision_Tree']['pred_time'] = end - start

    

results['Decision_Tree']['fbeta'] = fbeta_score(y_test_dt,y_pred_score_DT,beta=2)

results['Decision_Tree']['recall'] = recall_score(y_test_dt,y_pred_score_DT)

results['Decision_Tree']['precision'] = precision_score(y_test_dt,y_pred_score_DT)



print ("Train Time:", results['Decision_Tree']['train_time'])

print ("Prediction Time:", results['Decision_Tree']['pred_time'])

print ("fbeta score:", results['Decision_Tree']['fbeta'])

print('recall_score:', results['Decision_Tree']['recall'])

print('precision_score:', results['Decision_Tree']['precision'])

# split orginal data into train & test 

#X_train_rt, X_test_rt, y_train_rt, y_test_rt = split_data(2)





# split orginal data into train & test 

y = data.Class

X = data.drop(['Time','Class'], axis=1)



X_train_rt, X_test_rt, y_train_rt, y_test_rt =  train_test_split(X, 

                                                    y, 

                                                    test_size = 0.3, 

                                                    random_state = 0)



#Initializing Dictionary to store output of classifier metrics

results['Random_forest'] = {}



clf_RF = RandomForestClassifier(random_state=5, class_weight ='balanced')

start = time()

#Training the Classifier

clf_RF.fit(X_train_rt, y_train_rt)

end = time()



results['Random_forest']['train_time'] = end - start

start = time()

# PRedicting the test set

y_pred_score_RF = clf_RF.predict(X_test_rt)

end = time()

results['Random_forest']['pred_time'] = end - start



results['Random_forest']['fbeta'] = fbeta_score(y_test_rt,y_pred_score_RF,beta=2)

results['Random_forest']['recall'] = recall_score(y_test_rt,y_pred_score_RF)

results['Random_forest']['precision'] = precision_score(y_test_rt,y_pred_score_RF)





print ("Train Time:", results['Random_forest']['train_time'])

print ("Prediction Time:", results['Random_forest']['pred_time'])

print ("fbeta score:", results['Random_forest']['fbeta'])

print('recall_score:', results['Random_forest']['recall'])

print('precision_score:', results['Random_forest']['precision'])
# split orginal data into train & test 

from sklearn.naive_bayes import GaussianNB



# split orginal data into train & test 

X_train_gnb, X_test_gnb, y_train_gnb, y_test_gnb = split_data(2)



#Initializing Dictionary to store output of classifier metrics

results['GNB'] = {}



start = time()



# Training the classifier

clf_gnb = GaussianNB().fit(X_train_gnb, y_train_gnb)

end = time()

results['GNB']['train_time'] = end - start





# Predict on training set

start = time()

y_pred_score_gnb = clf_gnb.predict(X_test_gnb)

end = time()



results['GNB']['pred_time'] = end - start





results['GNB']['fbeta'] = fbeta_score(y_test_gnb,y_pred_score_gnb,beta=0.5)

results['GNB']['recall'] = recall_score(y_test_gnb,y_pred_score_gnb)

results['GNB']['precision'] = precision_score(y_test_gnb,y_pred_score_gnb)



print ("Train Time:", results['GNB']['train_time'])

print ("Prediction Time:", results['GNB']['pred_time'])

print ("fbeta score:", results['GNB']['fbeta'])

print('recall_score:', results['GNB']['recall'])

print('precision_score:', results['GNB']['precision'])
results_df = pd.DataFrame(results)  

display(results_df)

#print "Columns are ", results_df.columns
train_time = [results['undersampled']['train_time'],results['oversampled']['train_time'], 

              results['SMOTE']['train_time'],results['NearMiss']['train_time'], 

              results['Decision_Tree']['train_time'], results['Random_forest']['train_time'], 

              results['GNB']['train_time']]

pred_time = [results['undersampled']['pred_time'],results['oversampled']['pred_time'], 

             results['SMOTE']['pred_time'],results['NearMiss']['pred_time'], 

             results['Decision_Tree']['pred_time'],results['Random_forest']['pred_time'], 

             results['GNB']['pred_time']]



N = 7

width = 0.25       # the width of the bars



fig, ax = plt.subplots(figsize=(15,12))



ind = np.arange(N)

    

plt.bar(ind, train_time, width, label='train_time', color='#33FF7D')

plt.bar(ind + width, pred_time, width, label='pred_time', color='#FFAC33')





plt.ylim(0, 25)

plt.ylabel('Time in (Seconds)', fontsize = 20)

plt.xlabel('Classifiers', fontsize = 20)

plt.title('Time Comparison of the Models', fontsize = 25)



plt.xticks(ind + width, ('Undersampled', 'Oversampled', 'SMOTE', 'NearMiss', 'Decision Tree', 

                         'Random Forest', 'Gaussian Naive Bayes'))

plt.legend(loc='best', fontsize = 16 )

plt.show()
fbeta = [results['undersampled']['fbeta'],results['oversampled']['fbeta'], results['SMOTE']['fbeta'],

         results['NearMiss']['fbeta'], results['Decision_Tree']['fbeta'], results['Random_forest']['fbeta'],

         results['GNB']['fbeta']]

recall = [results['undersampled']['recall'],results['oversampled']['recall'], 

          results['SMOTE']['recall'],results['NearMiss']['recall'], 

          results['Decision_Tree']['recall'], results['Random_forest']['recall'], 

          results['GNB']['recall']]

    

    

precision= [results['undersampled']['precision'],results['oversampled']['precision'], 

            results['SMOTE']['precision'],results['NearMiss']['precision'],

            results['Decision_Tree']['precision'], results['Random_forest']['precision'], 

            results['GNB']['precision']]

    

N = 7

width = 0.25       # the width of the bars



fig, ax = plt.subplots(figsize=(15,12))

ind = np.arange(N)



plt.bar(ind , fbeta, width, label='fbeta', color='#33C4FF')

plt.bar(ind + width, recall, width, label='recall', color='#FF5733')

plt.bar(ind + width*2, precision, width, label='precision', color='#FFE333')

plt.ylim(0, 1 )

plt.ylabel('Scores', fontsize = 20)

plt.xlabel('Classifiers', fontsize = 20)

plt.title('Metrics Comparison of the Models', fontsize = 25)



plt.xticks(ind + width, ('Undersampled', 'Oversampled', 'SMOTE', 'NearMiss', 'Decision Tree', 

                         'Random Forest','Gaussian Naive Bayes'))

plt.legend(loc='best', fontsize = 16 )

plt.show()
resample_time = [results['SMOTE']['resample_time'],results['NearMiss']['resample_time']]

N = 2

fig, ax = plt.subplots(figsize=(5,5))

ind = np.arange(N)

 

plt.bar(ind, resample_time, width, label='resample_time', color='#33FF7D')

plt.ylim(0, 25)

plt.ylabel('Time in (Seconds)', fontsize = 20)

plt.xlabel('Classifiers', fontsize = 20)

plt.xticks(ind + width, ('SMOTE', 'NearMiss'))

plt.show()
#X_train_opt, X_test_opt, y_train_opt, y_test_dt = split_data(2)



#parameters list you wish to tune

parameters = { 'C' : [0.001, 0.1, 0.25, 0.55, 0.72, 0.85, 2, 5, 7, 9, 20, 30, 40, 60, 70, 90, 100] }



# Make an fbeta_score scoring object

scorer = make_scorer(fbeta_score,beta=2)



# Perform grid search on the classifier using 'scorer' as the scoring method

grid_obj = GridSearchCV(clf_smote,parameters,scoring=scorer)



# TODO: Fit the grid search object to the training data and find the optimal parameters

grid_fit = grid_obj.fit(X_train_sm,y_train_sm)



# Get the estimator

best_clf = grid_fit.best_estimator_

print("Optimzed Classfieris ", best_clf)

# Make predictions 

opt_classifier_score = best_clf.predict(X_test_sm)



print("F-score after optimizing:", fbeta_score(y_test_sm, opt_classifier_score, beta =2))

print("Recall-score after optimizing:", recall_score(y_test_sm, opt_classifier_score))

print("Precision-score after optimizing:", precision_score(y_test_sm, opt_classifier_score))



y_pred_prb_final = best_clf.predict_proba(X_test_sm)[:,1]
average_precision = average_precision_score(y_test_sm, opt_classifier_score)



print('Average precision-recall score: {0:0.2f}'.format(average_precision))



precision, recall, thresholds = precision_recall_curve(y_test_sm, opt_classifier_score)



plt.step(recall, precision, color='b', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2,

                 color='b')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))

thresholds = np.append(thresholds, 1)



plt.plot(thresholds, precision, color=sns.color_palette()[0])  

plt.plot(thresholds, recall, color=sns.color_palette()[1])  



leg = plt.legend(('precision', 'recall'), frameon=True)  

leg.get_frame().set_edgecolor('k')  

plt.xlabel('threshold')  

plt.ylabel('%')