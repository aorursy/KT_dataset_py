# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Import the test and train datasets



test_data = pd.read_csv("../input/test.csv", index_col=0, low_memory = False)

train_data = pd.read_csv("../input/train.csv", index_col=0, low_memory = False)



import os

print(os.listdir("../input"))



from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics



from string import ascii_letters

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix



# Any results you write to the current directory are saved as output.
#Drop the 'Unnamed:0.1' column because I think it is an error



test_data=test_data.drop(['Unnamed: 0.1'],1)



print('done')

#The mean of the default (0 or 1) gives the overall default rate

print(f'Default Rate in train_data: {train_data.default.mean()*100:.2f}%')

print(f'Default Rate in test_data:  {test_data.default.mean()*100:.2f}%')
#Group on Zip and displace the mean of default

grouped_train = train_data.groupby(['ZIP'], sort=False).mean().default #Do the function

grouped_train = grouped_train.sort_values(ascending=False) #Sort it by default rate



grouped_test = test_data.groupby(['ZIP'], sort=False).mean().default #Do the function

grouped_test = grouped_test.sort_values(ascending=False) #Sort it by default rate



#Print the output

print('Train Data')

print(grouped_train)

print()

print('Zip Code with highest default rate:', grouped_train.idxmax())

print()

print('Test Data')

print(grouped_test)

print('Zip Code with highest default rate:', grouped_test.idxmax())
#Calculate the mean of default in the data where year is 0

print(f'Train Data: {train_data.default[train_data.year==0].mean()*100:.2f}%')

print(f'Test Data: {train_data.default[test_data.year==0].mean()*100:.2f}%')
#Get the number of loans in each year of the test data

test_data.groupby('year').size()
#Use the Corr function

print(f'Correlation of Income and Age in Train: {train_data["income"].corr(train_data["age"]):.4f}')

print(f'Correlation of Income and Age in Test:  {test_data["income"].corr(test_data["age"]):.4f}')
#set X_train to include features included in assignment instructions

X_train = train_data[['ZIP','rent','education','income','loan_size','payment_timing','job_stability','occupation']]



#Turn categorical features to dummies

X_train = pd.get_dummies(X_train, columns=["ZIP", "occupation"])



#set y_train to be 'default' train_data and get dummies

y_train = train_data.default



# Create the model with 100 trees

model = RandomForestClassifier(n_estimators=100,

                              random_state=42,

                               n_jobs=-1,

                              oob_score = True)

# Fit on training data

model.fit(X_train, y_train)



#Run the test

y_pred = model.predict(X_train)



print(f'In-Sample Accuracy: {metrics.accuracy_score(y_train, y_pred)*100:.4}%')
print(f'Out of Bag Score: {model.oob_score_*100:.4f}%')
#set X_test to be all data in test_data, but drop default

X_test = test_data[['ZIP','rent','education','income','loan_size','payment_timing','job_stability','occupation']]



#Turn categorical features to dummies

X_test = pd.get_dummies(X_test, columns=["ZIP", "occupation"])



#set y_train to be 'default' train_data and get dummies

y_test = test_data.default



#Run the test

y_pred = model.predict(X_test)



print(f' Out-of-Sample Accuracy: {metrics.accuracy_score(y_test, y_pred)*100:.4f}%')
#confusion matrix

def print_confusion_matrix(test_default,predictions_test):

    cm = confusion_matrix(test_default,predictions_test)

    print('Paying Applicant Approved     :', cm[0][0], '---', cm[0][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100,"%")

    print('Paying Applicant Rejected     :', cm[0][1], '---', cm[0][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100,"%")

    print('Defaulting Applicant Approved :', cm[1][0], '---', cm[1][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100,"%")

    print('Defaulting Applicant Rejected :', cm[1][1], '---', cm[1][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100,"%")

print_confusion_matrix(y_test,y_pred)
pred_proba = model.predict_proba(X_test) #Get the actual probabilities from the model, not just binary

pred_proba = pd.DataFrame(pred_proba, columns=["proba_pay","proba_default"]) #rename those columns to make it easier to understand

test_data['proba_pay']= pred_proba.proba_pay #initiate a column on the test_data for the model's predicted probability of pay

test_data['proba_default']=pred_proba.proba_default #initiate a column on the test_data for the model's predicted probability of default

test_data['y_pred'] = y_pred #initiate a column on the test_data for the model prediction (binary)



#Print predicted default rate by minority status

print("Predicted Default Probability by Minority Status (%):")

print(test_data.groupby(["minority"]).mean().proba_default*100)



#Look at some other demographics

print()

print("Predicted Default Probability by Gender (%):")

print(test_data.groupby(["sex"]).mean().proba_default*100)

print()

print("Predicted Default Probability by Status and Gender (%):")

print(test_data.groupby(["minority","sex"]).mean().proba_default*100)
#Print rate of acceptance rate by minority status

print("Acceptance rate by Minority Status (%):")

print(100-test_data.groupby(["minority"]).mean().y_pred*100)



#Look at some other demographics

print()

print("Acceptance Rate of Defaults by Gender (%):")

print(100-test_data.groupby(["sex"]).mean().y_pred*100)

print()

print("Acceptance Rates of Default by Minority Status and Gender (%):")

print(100-test_data.groupby(["minority","sex"]).mean().y_pred*100)
print('Confusion Matrix of Minorities')

print_confusion_matrix(test_data.default[test_data.minority==1],test_data.y_pred[test_data.minority==1])



print()

print('Confusion Matrix of Non-Minorities')

print_confusion_matrix(test_data.default[test_data.minority==0],test_data.y_pred[test_data.minority==0])

print('Confusion Matrix of Female')

print_confusion_matrix(test_data.default[test_data.sex==1],test_data.y_pred[test_data.sex==1])



print()

print('Confusion Matrix of Male')

print_confusion_matrix(test_data.default[test_data.sex==0],test_data.y_pred[test_data.sex==0])
features = ['education', 'age', 'income', 'loan_size', 'payment_timing', 'job_stability']



for x in features:

    plt.hist(train_data[x], bins=100, alpha=0.5, label='train')

    plt.hist(test_data[x], bins=100, alpha=0.5, label='test')

    plt.legend(loc='upper right')

    print(x)

    print(plt.show())

print("++++++++++++++++++")

print("Job Stability")

print("++++++++++++++++++")



bins = pd.qcut(train_data['job_stability'], 10) #Cut the data into deciles

print("Train Data")

print(train_data.groupby(bins)['default'].mean())

print("")

bins = pd.qcut(test_data['job_stability'], 10) #Cut the data into deciles

print("Test Data")

print(test_data.groupby(bins)['default'].mean())



print("")

print("++++++++++++++++++")

print("Age")

print("++++++++++++++++++")



bins = pd.qcut(train_data['age'], 10) #Cut the data into deciles

print("Train Data")

print(train_data.groupby(bins)['default'].mean())

print("")

bins = pd.qcut(test_data['age'], 10) #Cut the data into deciles

print("Test Data")

print(test_data.groupby(bins)['default'].mean())



print("")

print("++++++++++++++++++")

print("Income")

print("++++++++++++++++++")



bins = pd.qcut(train_data['income'], 10) #Cut the data into deciles

print("Train Data")

print(train_data.groupby(bins)['default'].mean())

print("")

bins = pd.qcut(test_data['income'], 10) #Cut the data into deciles

print("Test Data")

print(test_data.groupby(bins)['default'].mean())
#Pivot tables for train and test of default rate by group. 

#'len default' is a bit of a misnomer, it counts how many people exist in the group.



print('+++++++++++++++++++++++++++')

print('Train')

print('+++++++++++++++++++++++++++')

print(pd.pivot_table(train_data,index=["ZIP"],values=["default"],aggfunc=[np.mean,len]))

print('')



print('+++++++++++++++++++++++++++')

print('Test')

print('+++++++++++++++++++++++++++')

print(pd.pivot_table(test_data,index=["ZIP"],values=["default"],aggfunc=[np.mean,len]))
#Pivot tables for train and test of default rate by group. 

#'len default' is a bit of a misnomer, it counts how many people exist in the group.



print('+++++++++++++++++++++++++++')

print('Train')

print('+++++++++++++++++++++++++++')

print(pd.pivot_table(train_data,index=["minority"],values=["default"],aggfunc=[np.mean,len]))

print('')



print('+++++++++++++++++++++++++++')

print('Test')

print('+++++++++++++++++++++++++++')

print(pd.pivot_table(test_data,index=["minority"],values=["default"],aggfunc=[np.mean,len]))
#Pivot tables for train and test of default rate by group. 

#'len default' is a bit of a misnomer, it counts how many people exist in the group.



print('+++++++++++++++++++++++++++')

print('Train')

print('+++++++++++++++++++++++++++')

print(pd.pivot_table(train_data,index=["sex"],values=["default"],aggfunc=[np.mean,len]))

print('')



print('+++++++++++++++++++++++++++')

print('Test')

print('+++++++++++++++++++++++++++')

print(pd.pivot_table(test_data,index=["sex"],values=["default"],aggfunc=[np.mean,len]))
#Pivot tables for train and test of default rate by group. 

#'len default' is a bit of a misnomer, it counts how many people exist in the group.



print('+++++++++++++++++++++++++++')

print('Train')

print('+++++++++++++++++++++++++++')

print(pd.pivot_table(train_data,index=["ZIP","minority"],values=["default"],aggfunc=[np.mean,len]))

print('')



print('+++++++++++++++++++++++++++')

print('Test')

print('+++++++++++++++++++++++++++')

print(pd.pivot_table(test_data,index=["ZIP","minority"],values=["default"],aggfunc=[np.mean,len]))
#Pivot tables for train and test of default rate by group. 

#'len default' is a bit of a misnomer, it counts how many people exist in the group.



print('+++++++++++++++++++++++++++')

print('Train')

print('+++++++++++++++++++++++++++')

print(pd.pivot_table(train_data,index=["sex","minority"],values=["default"],aggfunc=[np.mean,len]))

print('')



print('+++++++++++++++++++++++++++')

print('Test')

print('+++++++++++++++++++++++++++')

print(pd.pivot_table(test_data,index=["sex","minority"],values=["default"],aggfunc=[np.mean,len]))
train_data=pd.get_dummies(train_data, columns=["ZIP", "occupation"])
#What is the correlation of features in training data?



#produce a correlation matrix



sns.set(style="white")



# Compute the correlation matrix

corr = train_data.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#What is feature importance in the model?



feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)

feat_importances.nlargest(20).plot(kind='barh')
#Let's produce a correlation matrix



sns.set(style="white")



# Compute the correlation matrix

corr = X_test.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#How often did the model predict a default for zip code?

test_data_dummies = pd.get_dummies(test_data, columns=["ZIP", "occupation"])



#0,1 predictions

print('Calculated based on default/no default')

print('ZIP_MT01RA:', test_data_dummies.y_pred[test_data_dummies.ZIP_MT01RA==1].mean())

print('ZIP_MT04PA:', test_data_dummies.y_pred[test_data_dummies.ZIP_MT04PA==1].mean())

print('ZIP_MT12RA:', test_data_dummies.y_pred[test_data_dummies.ZIP_MT12RA==1].mean())

print('ZIP_MT15PA:', test_data_dummies.y_pred[test_data_dummies.ZIP_MT15PA==1].mean())



#prob predictions

print()

print('calculated based on model probability')

print('ZIP_MT01RA:', test_data_dummies.proba_default[test_data_dummies.ZIP_MT01RA==1].mean())

print('ZIP_MT04PA:', test_data_dummies.proba_default[test_data_dummies.ZIP_MT04PA==1].mean())

print('ZIP_MT12RA:', test_data_dummies.proba_default[test_data_dummies.ZIP_MT12RA==1].mean())

print('ZIP_MT15PA:', test_data_dummies.proba_default[test_data_dummies.ZIP_MT15PA==1].mean())

print('train')

print(pd.pivot_table(train_data,index=["rent"],values=["minority"],aggfunc=[np.mean,len]))

print()

print('test')

print(pd.pivot_table(test_data,index=["rent"],values=["minority"],aggfunc=[np.mean,len]))
bins = pd.qcut(train_data['job_stability'], 10) #Cut the data into deciles

print("Train Data")

print(train_data.groupby(bins)['minority'].mean())

print("")

bins = pd.qcut(test_data['job_stability'], 10) #Cut the data into deciles

print("Test Data")

print(test_data.groupby(bins)['minority'].mean())
print('Job Stability and Occupation_MZ01CD:', test_data_dummies['job_stability'].corr(test_data_dummies['occupation_MZ01CD']))

print('Job Stability and Occupation_MZ10CD:', test_data_dummies['job_stability'].corr(test_data_dummies['occupation_MZ10CD']))

print('Job Stability and Occupation_MZ11CD:', test_data_dummies['job_stability'].corr(test_data_dummies['occupation_MZ11CD']))

print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

print('Default and Occupation_MZ01CD:', test_data_dummies['default'].corr(test_data_dummies['occupation_MZ01CD']))

print('Default and Occupation_MZ10CD:', test_data_dummies['default'].corr(test_data_dummies['occupation_MZ10CD']))

print('Default and Occupation_MZ11CD:', test_data_dummies['default'].corr(test_data_dummies['occupation_MZ11CD']))
