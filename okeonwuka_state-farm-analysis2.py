import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input"))



#read in data

raw_data = pd.read_csv("../input/exercise05train/exercise_05_train.csv")

raw_data_test_set = pd.read_csv("../input/exercise05test/exercise_05_test.csv")



print("no. rows of raw training data is {}".format(len(raw_data)))

print("no. rows of raw test data is {}".format(len(raw_data_test_set)))
# Remove null data

raw_data = raw_data.dropna()



# As per instruction, we need to maintain the 10000 rows of test data.

# Therefore do not remove NULL data, even though it exists. Replace it with 0, to avoid errors

data_test_set = raw_data_test_set.fillna(0)



# Convert column 'x41' to a string to enable us later use 'replace' method in this column

data_test_set['x41'] = data_test_set.x41.fillna(0).astype(str)



# raw_data_test_set.fillna(0)

# data_test_set = raw_data_test_set





print("the no of rows of raw data is {}".format(len(raw_data)))

print("the no of rows of test data is {}".format(len(data_test_set)))



#View features. Set pandas to show all columns

pd.set_option('display.max_columns',500)



print("First 5 rows of raw data:")

display(raw_data.head(5))



print("First 5 rows of test data:")

display(data_test_set.head(5))



# Select random subset of raw_data (10000 rows) for faster analysis. 

# Use random_state for reproducibility

#data = raw_data.sample(n=10000, random_state=2)



#UPDATE: Work with all the available training data

data = raw_data







print("sample of random subset of data:")

display(data.head(5))



#reset index of random_subset_data_train_set dataframe. 

#This prevents errors from popping up later

data = data.reset_index(drop = True)



print("sample of reindexed data:")

display(data.head(5))





# examine stats of data

print("descriptive data stats")

display(data.describe())
# Remove '$' in col x41 from train set

remove_dollarsign = data.x41.apply(lambda x: x.replace('$','')).astype('float'); # had to add "astype('float')" to convert values in col from string to float. if i dont, i get errors in histogram  plotam

data['x41'] = remove_dollarsign



# Remove '$' in col x41 from test set

remove_dollarsign_test = data_test_set.x41.apply(lambda x: x.replace('$','')).astype('float');

data_test_set['x41'] = remove_dollarsign_test

# pd.set_option('display.max_columns',500)

# display(random_subset_data_test_set.head(5))



# Remove '%' in col x45 from train set

remove_percentsign = data.x45.apply(lambda x: x.replace('%','')).astype('float'); # had to add "astype('float')" to convert values in col from string to float. if i dont, i get errors in histogram  plot;

data['x45'] = remove_percentsign



# Remove '%' in col x45 from test set

remove_percentsign_test = data_test_set.x45.apply(lambda x: x.replace('%','')).astype('float');

data_test_set['x45'] = remove_percentsign_test



#view data

display(data.head(5))

display(data_test_set.head(5))

# Split the data into features and target label



#Select y labels from training data

label = data['y']



#Get features from training data

features = data.drop('y',axis = 1)
# Feature Exploration

# Check for skewed data by vizualizing histogram of features

for i in range(100):

    #skip categorical data

    if i == 34 or i == 35 or i == 68 or i == 93:

        pass

    else:

        print(features.hist(column = 'x' + str(i)))
# Since features are observed to be normaly distributed, there is no need for adjusting for skewed data



# normalise numerical features

from sklearn.preprocessing import MinMaxScaler



# Initialise scaler

scaler = MinMaxScaler()



# Apply scaler to numeric features

numerical = []

for i in range(100):

    #skip categorical data

    if i == 34 or i == 35 or i == 68 or i == 93:

        pass

    else:

        numerical.append('x' + str(i))        

print(numerical)



# Minmax transform x_train 

features_minmax_transform = pd.DataFrame(data = features)

features_minmax_transform[numerical] = scaler.fit_transform(features[numerical])



# Minmax transform test data

x_test_minmax_transform = pd.DataFrame(data = data_test_set)

x_test_minmax_transform[numerical] = scaler.fit_transform(data_test_set[numerical])



display(features_minmax_transform.head(5))

display(x_test_minmax_transform.head(5))
# use one-hot encoding to transform the non-numeric features

features_final = pd.get_dummies(features_minmax_transform)

x_test_final = pd.get_dummies(x_test_minmax_transform)



display(features_final.head(5))

display(len(features_final.columns))

display(x_test_final.head(5))

display(len(x_test_final.columns))



# Remove column mismatches between features_final and x_test_final

x_test_final = x_test_final.drop(['x34_0','x35_0','x68_0','x93_0'],axis =1)



# Now both dataframes have equal columns

print("The number of columns in features_final is {}".format(len(features_final.columns)))

print("The number of columns in x_test_final is {}".format(len(x_test_final.columns)))



# Lets visualise the final cleansed training data ie features_final

print(" Correlation of feature values:")

#print(features_final.corr())



# create a heatmap to see the correlation of final features. A heatmap shows what areas are 'hot' eg in a web page, the most visited area when rep in heat map will be extremely red

# you will notice a red diagonal on the correlation because as usual correlating same features give the highest correlation value : 1. So red hot!

print("heatmap of the correlation of final features. However to many features to see clearly heatmap pattern")

plt.figure(figsize=(20, 15))

sns.heatmap(features_final.corr(), cmap= 'coolwarm', annot = False, linewidth = 0.5) # annot writes values on map(default is False), coolwarm is the color spectrum, linewidth puts lines on heatmap



print("Heatmap of Final Features:")

plt.figure(figsize=(20, 15))

ax2 = sns.heatmap(features_final) # or just simply "sns.heatmap(features_final)"... no need for ax1, ax2 etc



# # or 

# ax = sns.heatmap(features_final)
# shuffle and split data with cross validation

from sklearn.model_selection import train_test_split



# Split the 'features_final' and 'label' data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(features_final,label,test_size = 0.2,random_state = 0)

# Show the results of the split

print("Training set has {} samples.".format(X_train.shape[0]))

print("Testing set has {} samples.".format(X_test.shape[0]))
# Model trained with logistic regression



#Create regression function

def stateFarmRegressor(X_train,y_train):

    from sklearn.linear_model import LogisticRegression

    

    #create regressor

    reg = LogisticRegression()

    

    #fit regressor to data

    reg.fit(X_train,y_train)

    

    return reg



stateFarmRegressor(X_train,y_train)

# Model trained with SVM



#Create SVM function

def stateFarmSVM(X_train,y_train):

    from sklearn import svm

    

    # create model

    clf = svm.SVC(gamma='scale',probability=True) # Set probability to 'True' to enable us use 'predict_proba' later

    

    #fit model to data

    clf.fit(X_train,y_train)

    

    return clf



stateFarmSVM(X_train,y_train)
# Predict test set results

logreg_model = stateFarmRegressor(X_train,y_train)

svm_model = stateFarmSVM(X_train,y_train)



# check the accuracy score with test data from the split:

# Method 1:

accu_score_logreg_model1 = logreg_model.score(X_test,y_test)

accu_score_svm_model1 = svm_model.score(X_test,y_test)



print("The accuracy of the logistic regression model,logreg_model is {}".format(accu_score_logreg_model1))

print("The accuracy of the svm model,svm_model is {}".format(accu_score_svm_model1))



# Method 2:

# Generate y predictions from X_test and compare with true values from y_test

y_pred_logreg = logreg_model.predict(X_test)

y_pred_svm = svm_model.predict(X_test)



from sklearn.metrics import accuracy_score

accu_score_logreg_model2 = accuracy_score(y_test,y_pred_logreg)

accu_score_svm_model2 = accuracy_score(y_test,y_pred_svm)

print("The accuracy of the logistic regression model,logreg_model is {}".format(accu_score_logreg_model2))

print("The accuracy of the svm model,svm_model is {}".format(accu_score_svm_model2))



# when training was done with 10000 data points (for faster analysis) accuracy score was 0.8805 and 0.908

# foe logistic regressor and svm model respt

from sklearn.metrics import precision_score

preci_score_logreg_model = precision_score(y_test,y_pred_logreg)

preci_score_svm_model = precision_score(y_test,y_pred_svm)



# Output:

print("The precision of the logistic regression model,logreg_model is {}".format(preci_score_logreg_model))

print("The precision of the svm model,svm_model is {}".format(preci_score_svm_model))



# when training was done with 10000 data points (for faster analysis) precision score was 0.7814569536423841 

# and 0.96296296296296298 for logistic regressor and svm model respt
from sklearn.metrics import recall_score

recall_score_logreg_model = recall_score(y_test,y_pred_logreg)

recall_score_svm_model = recall_score(y_test,y_pred_svm)



# Output:

print("The recall of the logistic regression model,logreg_model is {}".format(recall_score_logreg_model))

print("The recall of the svm model,svm_model is {}".format(recall_score_svm_model))



# when training was done with 10000 data points (for faster analysis) recall score was 0.5770171149144254 

# and 0.5721271393643031 for logistic regressor and svm model respt
from sklearn.metrics import f1_score

f1_score_logreg_model = f1_score(y_test,y_pred_logreg)

f1_score_svm_model = f1_score(y_test,y_pred_svm)



# Output:

print("The f1 score of the logistic regression model,logreg_model is {}".format(f1_score_logreg_model))

print("The f1 score of the svm model,svm_model is {}".format(f1_score_svm_model))



# when training was done with 10000 data points (for faster analysis) recall score was 0.6638537271448663 

# and 0.7177914110429447 for logistic regressor and svm model respt



#Recap:

print("Check f1 score for logreg_model with raw formula: {} ".format(2*((preci_score_logreg_model*recall_score_logreg_model)/(preci_score_logreg_model+recall_score_logreg_model))))

print("Check f1 score for logreg_model with raw formula: {} ".format(2*((preci_score_svm_model*recall_score_svm_model)/(preci_score_svm_model+recall_score_svm_model))))



#Note

# when training was done with 10000 data points (for faster analysis) f1 score was 0.6638537271448663

# and 0.7177914110429447 for logistic regressor and svm model respt

# A confusion matrix is a table that is used to evaluate the performance of a classification model

# Diagonal values represent accurate predictions, while non-diagonal elements are inaccurate predictions. 

# In the output, 1525 and 236 are actual predictions, and 173 and 66 are incorrect predictions

# import the metrics class

from sklearn import metrics

cnf_matrix_logreg = metrics.confusion_matrix(y_test, y_pred_logreg)

cnf_matrix_logreg



#Note

# when training was done with 10000 data points (for faster analysis) confusion matrix was 1525 and 236 are actual predictions, 

# and 173 and 66 are incorrect predictions

# Visualizing Confusion Matrix using Heatmap

class_names=[0,1] # name  of classes

fig3, ax3 = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix_logreg), annot=True, cmap="YlGnBu" ,fmt='g')

ax3.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn import metrics

cnf_matrix_svm = metrics.confusion_matrix(y_test, y_pred_svm)

cnf_matrix_svm
class_names=[0,1] # name  of classes

fig4, ax4 = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix_svm), annot=True, cmap="YlGnBu" ,fmt='g')

ax3.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')



#Note

# when training was done with 10000 data points (for faster analysis) confusion matrix was 1582 and 234 are actual predictions, 

# and 175 and 9 are incorrect predictions
#Already imported metrics from sklearn so no need to repeat

# prediction probability performance of logistic regressor model, logreg_model, on given training test data set from split ie X_test

y_label_prediction_probability_logreg = logreg_model.predict_proba(X_test)[::,1]



#Get the false positive rate, fpr.... skip threshold

fpr_logreg, tpr_logreg, _ = metrics.roc_curve(y_test, y_label_prediction_probability_logreg)

auc_logreg = metrics.roc_auc_score(y_test, y_label_prediction_probability_logreg)

plt.plot(fpr_logreg,tpr_logreg,label="data 1,auc="+str(auc_logreg))

plt.legend(loc=4)

plt.show()

#Already imported metrics from sklearn so no need to repeat

# prediction probability performance of logistic regressor model, logreg_model, on given training test data set from split ie X_test

y_label_prediction_probability_svm = svm_model.predict_proba(X_test)[::,1]



#Get the false positive rate, fpr.... skip threshold

fpr_svm, tpr_svm, _ = metrics.roc_curve(y_test, y_label_prediction_probability_svm)

auc_svm = metrics.roc_auc_score(y_test, y_label_prediction_probability_svm)

plt.plot(fpr_svm,tpr_svm,label="data 1,auc="+str(auc_svm))

plt.legend(loc=4)

plt.show()

# prediction probability performance of logistic regressor model, logreg_model, on given test data set, x_test_final

y_label_prediction_probability_logreg = logreg_model.predict_proba(x_test_final)

print("The probability, given the features x_test_final, of the predicted y_label belonging to each class 0 or 1 using log_reg model is {}".format(y_label_prediction_probability_logreg))



# prediction probability performance of svm model, svm_model, on given test data set, x_test_final

y_label_prediction_probability_svm = svm_model.predict_proba(x_test_final)

print("The probability, given the features x_test_final, of the predicted y_label belonging to each class 0 or 1 using svm model is {}".format(y_label_prediction_probability_svm))



# As per case study requirements/instructions, prediction probability test data x_test_final of belonging to 'y = 1' class ie (y_positive)

y_positive_logreg = logreg_model.predict_proba(x_test_final)[:,1]

y_positive_svm = svm_model.predict_proba(x_test_final)[:,1]

print("The probability, given the features x_test_final, of the predicted y_label belonging to each class 1 is {} for log_reg model and {} for svm_model".format(y_positive_logreg,y_positive_svm))



# Check length or positive (y = 1) predictions to see if it coincides with x_test_final, as precation

print(len(y_positive_logreg))

print(len(y_positive_svm))
# Save results

# convert y_positive of both models to a dataframe format

y_positive_logreg = pd.DataFrame(y_positive_logreg)

y_positive_svm = pd.DataFrame(y_positive_svm)



# Generate results1 output

y_positive_logreg.to_csv('result1.csv',header=['y_positive'],index=False)



# Generate results2 output

y_positive_svm.to_csv('result2.csv',header=['y_positive'],index=False)


