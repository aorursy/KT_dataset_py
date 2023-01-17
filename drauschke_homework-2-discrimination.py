#Python 3 environment



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #graphical functions

import seaborn as sns #graphical functions

from sklearn.ensemble import RandomForestClassifier #random forest classifier tool

from sklearn.model_selection import train_test_split #split tool

from sklearn import metrics #analysis tools

from sklearn.metrics import accuracy_score #accuracy score

from sklearn.metrics import confusion_matrix #confusion matrix



import os

print(os.listdir("../input"))
df_train = pd.read_csv("../input/train.csv", low_memory = False) 

       #Load the train data using pandas read csv function, low_memory is turned on by default and 

       #allows the file to be processed in chunks default



df_test = pd.read_csv("../input/test.csv", low_memory = False) 

       #Load test data
print(df_train.dtypes)

print()

print(df_test.dtypes)

print()

print("Shape Training Date:", df_train.shape)

print()

print("Shape Test Data:", df_test.shape)

print()

print(df_train.head())

print()

print(df_test.head())

print()

print("Gender Train:", np.unique(df_train.sex))

print("Gender Test:", np.unique(df_test.sex))

print("Minority Train:", np.unique(df_train.minority))

print("Minority Test:", np.unique(df_test.minority))

print("Maturity Train:", np.unique(df_train.year))

print("Maturity Test:", np.unique(df_test.year))

print("Rent Train:", np.unique(df_train.rent))

print("Rent Test:", np.unique(df_test.rent))

print("ZIP Train:", np.unique(df_train.ZIP))

print("ZIP Test:", np.unique(df_test.ZIP))

print("Occupation Train:", np.unique(df_train.occupation))

print("Occupation Test:", np.unique(df_test.occupation))

       #show the features, dataset shapes,top five rows of data, and unique entries for binary data

       #duplicating to ensure that Train and Test datasets are the same

        

       #notice that the train and test data dimensions do not match (14 vs 15 columns)

       #notice that our train data accounts for 480k rows or 75% of total data (480 / (480 + 160))

       #we can see that the first column in the train data is just a row count -> remove this column

       #we can see that the first two columns in the test data are row counts -> remove first two columns

       #notice that default, gender, minority, rent are binary

    

       #QUESTIONS: (1) education is numerical; (2) job_stability is numerical; (3) how is payment_timing determined; 

       #(4) year is maturity? why 30 years seems like a mortgage but then what does the rent parameter mean? 

       #also IMPORTANT notice that the train data has year 0-29 and the test data 30 - 49

       #(5) income self-reported?; (6) loan_size is funded amount or outstanding? accrued interest?

       
#train features

base_train = df_train[["loan_size","payment_timing","education","occupation","income",

                "job_stability","ZIP","rent"]].copy() 

                   #pull relevant data from dataframe

train = pd.get_dummies(data=base_train,columns=["occupation","ZIP"])

print(train.head())

print(train.shape)

print()







#test features

base_test = df_test[["loan_size","payment_timing","education","occupation","income",

                "job_stability","ZIP","rent"]].copy()

test = pd.get_dummies(data=base_test,columns=["occupation","ZIP"])







print(test.head())

print(test.shape)

print()

print()

      

      #notice the new datasets now have 13 features (for train = 8 selected features

      #+ 4 ZIP dummies + 3 Occupation dummies - 2) and the same number of rows as before



        

        

#train labels

train_default = df_train.default

print(train_default.shape)

print()



#test labels

test_default = df_test.default

print(test_default.shape)

      

      #train and test labels created for later use

      
#HISTOGRAMS OF KEY CONTINIOUS FEATURES



#

plt.hist(df_train["education"], bins = 100, alpha=0.5, label='Train') #train distr for education, 100 bins, alpha for transparency setting (0-1)

plt.hist(df_test["education"], bins = 100, alpha=0.5, label='Test') #test distr for education

plt.title("Distribution of Education Across Datasets") #title

plt.ylabel("Observations #") #y axis label

plt.legend(loc='upper right') #legend

plt.show() #show

      #The distribution across sets significantly different + strange dual peak distributions -> CONCERN

plt.hist(df_train["income"], bins = 100, alpha=0.5, label="Train")

plt.hist(df_test["income"], bins = 100, alpha=0.5, label="Test")

plt.title("Distribution of Income Across Datasets")

plt.ylabel("Observations #")

plt.legend(loc='upper right')

plt.show()

      #The distribution across sets significantly different + strange dual peak distributions -> CONCERN + Correlation with education?

    
plt.hist(df_train["job_stability"], bins = 100, alpha=0.5, label="Train")

plt.hist(df_test["job_stability"], bins = 100, alpha=0.5, label="Test")

plt.title("Distribution of Job Stability Across Datasets")

plt.ylabel("Observations #")

plt.legend(loc='upper right')

plt.show()

      #The distribution across sets significantly different + strange dual peak distributions -> CONCERN
plt.hist(df_train["age"], bins = 100, alpha=0.5, label="Train")

plt.hist(df_test["age"], bins = 100, alpha=0.5, label="Test")

plt.title("Distribution of Age Across Datasets")

plt.ylabel("Observations #")

plt.legend(loc='upper right')

plt.show()

      #Equal distribution across sets -> FINE
plt.hist(df_train["loan_size"], bins = 100, alpha=0.5, label="Train")

plt.hist(df_test["loan_size"], bins = 100, alpha=0.5, label="Test")

plt.title("Distribution of Loan Size Across Datasets")

plt.ylabel("Observations #")

plt.legend(loc='upper right')

plt.show()

      #Equal distribution across sets -> FINE
plt.hist(df_train["payment_timing"], bins = 100, alpha=0.5, label="Train")

plt.hist(df_test["payment_timing"], bins = 100, alpha=0.5, label="Test")

plt.title("Distribution of Payment Timing Across Datasets")

plt.ylabel("Observations #")

plt.legend(loc='upper right')

plt.show()

      #Equal distribution across sets but strange distribution -> FINE
#Correlation matrix for train data

print("Correlation Matrix in Training Data")

correlation_train = df_train[["minority","sex","ZIP","rent","education","age",

                "income","payment_timing","job_stability","occupation","year"]].copy() 

                #create new dataframe for correlation matrix

corr = correlation_train.corr() #correlation matrix

corr.style.background_gradient(cmap='coolwarm')
#Correlation matrix for train data

correlation_test = df_test[["minority","sex","ZIP","rent","education","age",

                "income","payment_timing","job_stability","occupation","year"]].copy() 

                #create new dataframe for correlation matrix

corr = correlation_test.corr() #correlation matrix

corr.style.background_gradient(cmap='coolwarm')
print("Training Data Defaults (%):", round(train_default.mean()*100,2), "%")

print("Training Data Defaults (#):", len(train_default[train_default == True]))

print()

print("Test Data Defaults (%):", round(test_default.mean()*100,2), "%")

print("Test Data Defaults (#):", len(test[test_default == True]))



      #notice that the training data has a significantly higher default rate -> CONCERN
#ANSWER

print("ZIP Code with Highest Default Rate in Train Data:   ", df_train.groupby(by="ZIP").default.mean().idxmax())

print()

print("ZIP Code with Highest Default Rate in Test Data:   ", df_test.groupby(by="ZIP").default.mean().idxmax())

print()

      #print the ZIP codes with the highest percentage of defaults (mean = % since sum of 1 divided by number of observations)
###############

#ANALYSIS BACKUP

print("ANALYSIS:")

print()

print("ZIP Codes in Train Data:", np.unique(df_train.ZIP))

print()

print("ZIP Codes in Test Data:", np.unique(df_test.ZIP))

print()

      #show unique ZIP codes (there are four and they are the same across the datasets)



print("Train Data", df_train.groupby(by="ZIP").default.count())

print()

print("Test Data",df_test.groupby(by="ZIP").default.count())

print()

print()

      #count the number of observations across ZIP code

      #notice that exactly 25% of observations come from each of the four ZIP codes 



        

print("Default by ZIP Code in Train (#)", df_train.groupby(by="ZIP").default.sum())

print()

print("Default by ZIP Code in Train (%)", round(df_train.groupby(by="ZIP").default.mean()*100,2))

print()

      #group by ZIP code and count defaults (True = 1 therfore sum works)

      #for percentages divide number of observations per ZIP code / 100 (possible since equal number of observations)

      

      #interestingly, in both dataset defaults are almost equally split between ZIP MT04PA and MT15PA

      #since MT04PA has three more defaults in the train data it has the marginally higher default rate

    

    

print("Default by ZIP Code in Test (#)", df_test.groupby(by="ZIP").default.sum())

print()

print("Default by ZIP Code in Test (%)", round(df_test.groupby(by="ZIP").default.mean()*100,2))

print()

      #defaults are still only found in MT04PA and MT15PA but the default rate is significantly 

      #lower in the test set (and the marginally higher default rate is found in the latter in contrast to the train set)

      #-> CONCERN
#ANSWER

print("Year 0 Train Data Default Rate:  ", round(df_train.default[df_train.year==0].mean()*100,2),"%")

print("Year 1 Train Data Default Rate:  ", round(df_train.default[df_train.year==1].mean()*100,2),"%")

     #49.97% in year zero and 49.98% in year 1

     #since year data starts at 30 in the training set cannot calculate
#ANSWER

print("Correlation Age-Income Train Data:  ", round(df_train['age'].corr(df_train['income']),4))

      #in training set correlation between age and income = 0.1027

print("Correlation Age-Income Test Data:   ", round(df_test['age'].corr(df_test['income']),4))

      #in test set correlation between age and income = 0.5298

      

      #this difference could have been predicted based on our histograms which showed that income was strangly distributed

      #-> CONCERN but we are not using age in our data so should not affect our model per se
###############

#ANALYSIS BACKUP



plt.scatter(df_train["age"],df_train["income"])

plt.title("TRAIN DATA: Scatter Plot Age vs Income")

plt.xlabel("Age")

plt.ylabel("Income ($)")

plt.show()

     #scatter plots highlight the effect of the strange income distribution on the correlation

    

plt.scatter(df_test["age"],df_test["income"])

plt.title("TEST DATA: Scatter Plot Age vs Income")

plt.xlabel("Age")

plt.ylabel("Income ($)")

plt.show()

     #the income distrbution in the test set is not affected and therefore the correlation is higher
#Fit model

rf = RandomForestClassifier(n_jobs=-1, n_estimators = 100, oob_score = True, random_state = 42)

       #we use the classifier becasue we have a binary output; n_estimator is number of trees in forest; 

       #max_depth is maximum depth of the tree; random_state is seed for RNG

    

rf.fit(train,train_default)

       #fit the model

    

predictions_train = rf.predict(train)

       #save in-sample predictions

predictions_test = rf.predict(test)

       #save model predictions for the test set
#ANSWER

#print score

print("In-Sample Accuracy:  ", rf.score(train, train_default)*100, "%")

      #accuracy of 100%!
###############

#ANALYSIS BACKUP



#Show feature importance

feat_importances = pd.Series(rf.feature_importances_, index=train.columns)

feat_importances.nlargest(20).plot(kind="barh")

    #notice the importance of occupation and job_stability in the model     



     

#show confusion matrix

def print_confusion_matrix(train_default,predictions_train):

    cm = confusion_matrix(train_default,predictions_train)

    print('Paying Applicant Approved =     ', cm[0][0])

    print('Paying Applicant Rejected =     ', cm[0][1])

    print('Defaulting Applicant Accepted = ', cm[1][0])

    print('Defaulting Applicant Rejected = ', cm[1][1])

    #define the condusion matrix so that it prints output in easily interpretable way

    #note that the print settings call predefined parts of the output 2x2 matrix (i.e. true positive is 0;0 i.e. top-left)

    #to execute command print_confusion_matrix(train_default,predictions_train)

    

print_confusion_matrix(train_default,predictions_train)

    #all default (negatives) and repay (positives) predictions are correct!
#ANSWER

#print OOB

print("Out of Bag Score:  ", round(rf.oob_score_*100, 4), "%")

      #very high OOB score
#ANSWER

print("Test Accuracy:  ", round(rf.score(test, test_default)*100, 4), "%")



      #Lower but still very high. However, notice the significant difference between the OOB 

      #and the test accuracy - a testimony to the significance of the structural breaks between the test and the train data
###############

#ANALYSIS BACKUP



#confusion matrix

def print_confusion_matrix(test_default,predictions_test):

    cm = confusion_matrix(test_default,predictions_test)

    print('Paying Applicant Approved     =', cm[0][0])

    print('Paying Applicant Rejected     =', cm[0][1])

    print('Defaulting Applicant Approved =', cm[1][0])

    print('Defaulting Applicant Rejected =', cm[1][1])

print_confusion_matrix(test_default,predictions_test)

print()

    #shows that the model has very bad performance in dealing with 

    

print("Number of Defaults Predicted:", predictions_test.sum())

print("Actual Number of Defaults:", test_default.sum())
#ANSWER



#since predictions from classifier are binary we need to extract probabilities from the model

prob_prediction = rf.predict_proba(test)

p_prediction = pd.DataFrame(prob_prediction, columns=["p_repay","p_default"])

p_prediction["reject_application"] = pd.Series(list(predictions_test))

     #turn probability arrays into dataframe and label columns also 

     #add model prediction (lendion decision where true = rejected)



df_predict = pd.concat([df_test, p_prediction,test_default], axis=1)

     #add probabilities, default prediction, and actual default to orriginal test dataset



#Print predicted default rate by minority status

print("Predicted Default Rate by Minority Status (%):")

print(df_predict.groupby(["minority"]).mean().reject_application*100)

     #for non-minorities the predicted default rate (rejection rate) was 0.11%

###############

#ANALYSIS BACKUP



#check that new concatenate worked

print("Shape of Dataset with Probability:", df_predict.shape)

print()

     #shows that the concatenate worked and the probabilities are part of the dataset (15 + 4 columns)



#inspect prediction probability and rejection data

print(p_prediction.head())

print()

     #notice when p_repay > 0.5 then application is accepted (reject application = False). All as it should be



#percentage of minorities in test data

print("Minorities in Test Data (%):  ", df_test.minority.sum()/df_test.minority.count()*100,"%")

print()



#show number for rejected applications (predicted defaults)

print(df_predict.groupby("reject_application").size())

print()
#ANSWER

    

#Print predicted default rate by minority status

print("Predicted Default Rate by Minority Status (%):")

print(df_predict.groupby(["minority"]).mean().reject_application*100)

     #for minorities the predicted default rate (rejection rate) was 4.18%
###############

#ANALYSIS BACKUP



#lets look if this discrepancy holds for average predicted probability (not rejection rate)

print("Average Predicted Default Probability by Minority Status (%):")

print(df_predict.groupby(["minority"]).mean().p_default*100)

print()

     #for minorities the actual predicted average default probability is 21.18% 

     #difference to non-minorities (19.89%) not significantly different

     #as can be seen in the below histogram the difference of the rejection rate is 

     #driven by a minority within the minority 

        

#Visualize the distribution of default probability by minority

plt.hist(df_predict[df_predict.minority==0].p_default, bins = 100, alpha=0.5, label='Non-Minority')

plt.hist(df_predict[df_predict.minority==1].p_default, bins = 100, alpha=0.5, label='Minority')

plt.plot([0.5, 0.5], [0, 40000], color="r", linestyle="--", lw=2) #plot cut-off threshold (red, dashed, width 2)

plt.title("Distribution of Default Probability by Minority")

plt.ylabel("Observations #")

plt.legend(loc='upper right')

plt.show()

     #all applications to the left of dashed line are accepted
#ANSWER

#Yes, it is unaware because it applies the same probability threshold 

#to all features (if predicted probability of default is above 50% then reject). 

#Indeed, our model doesnt even use the protected features and therefore it could 

#not apply a different threshold.
#ANSWER

#Demographic parity requires that a decision (issuing a loan) be 

#independent of the protected attribute (gender, race, etc). If 

#demographic parity exists then the rejection rate should be the 

#same across the protected attribute



rejected_applicants = df_predict.loc[df_predict["reject_application"] == True]

      #create dataset with only rejected applications



#gender

print("Rejected Applications (Proportion with Female):   ", round(rejected_applicants.sex.mean()*100,2),"%")

print()

print("Number of Rejections (Female):")

print(df_predict.groupby(["reject_application","sex"]).sex.count())

print()

print()

      #out of rejected applications what proportion is female? = 37.11% 

      #suggest significant imbalance. 1280 rejections vs 2169! 1.69x more male rejections.



#minorities

print("Rejected Applications (Proportion with Minority Status):   ", round(rejected_applicants.minority.mean()*100,2),"%")

print()

print("Number of Rejections by Minority Status:")

print(df_predict.groupby(["reject_application","minority"]).minority.count())

print()

      #out of rejected applications what proportion is from the minority? = 97.36% 

      #suggest even more significant imbalance. 3,358 rejections vs 91! 36.90x more
###############

#ANALYSIS BACKUP



#rejection rates grouped by gender

print("Application Rejection Rate by Gender (%):")

print(df_predict.groupby(["sex"]).mean().reject_application*100)

print()

print("Number of Rejections by Gender")

print(df_predict.groupby(["reject_application","sex"]).sex.count())

print()

      #The rejection rate for male (0) is 2.71% (2,169 rejections) vs 1.60% for females (1,280 rejection)

      #The 1.69x more male rejections than females. Therefore, there is no demographic parity by gender.

    

#rejection rates grouped by minority status

print("Application Rejection Rate by Minority Status (%):")

print(df_predict.groupby(["minority"]).mean().reject_application*100)

print()



       #The rejection rate for non-minorities (0) is 0.11% (91 rejections) vs 4.18% for minorities (3,358 rejection)

       #The 36.90x more minority rejections than non-minority. Therefore, no demographic parity by minority status.

        

#in accepted applications what proportion is from minorities

accepted_applicants = df_predict.loc[df_predict["reject_application"] == False] #new dataset with only accepted applications

print("Accepted Applications (Proportion with Minority Status):   ", round(accepted_applicants.minority.mean()*100,2),"%")

print("Accepted Applications (# with Minority Status):            ", accepted_applicants.minority.count())

print()

      #out of accepted applications what proportion is from the minority? 49.18% suggest relativly even

    

#in accepted applications what proportion is female

print("Accepted Applications (Proportion with Minority Status):   ", round(accepted_applicants.sex.mean()*100,2),"%")

print("Accepted Applications (# with Minority Status):            ", accepted_applicants.sex.count())

print()

      #out of accepted applications what proportion is from the minority? 49.18% suggest relativly even
#ANSWER

#Equal opportunity requires (source: https://research.google.com/bigpicture/attacking-discrimination-in-ml/)

#that of the people who can pay back a loan, the same fraction in each group should actually be granted a loan.

#That is, the true positive rate must be equal across groups



print("Actual Default Rate by Minority Status (%):")

print(df_predict.groupby(by="minority").mean().default*100)

print()

print("Rejection Rate by Minority Status (%):")

print(df_predict.groupby(by="minority").mean().reject_application*100)

print()

     #85.10% of non-minorities can repay whilst 99.89% are granted a loan

     #84.97% of minorities can repay whilst 95.82% are granted loans

     #Given the three percentage point gap it seems that there is a gap in equal opportunity.

    



print("Actual Default Rate by Gender (%):")

print(df_predict.groupby(by="sex").mean().default*100)

print()

print("Rejection Rate by Gender (%):")

print(df_predict.groupby(by="sex").mean().reject_application*100)

     #85.00% of 0 gender can repay whilst 97.29% are granted a loan

     #85.07% of 1 gender can repay whilst 98.40% are granted loans

     #It does not seem like there is a gender equal opportunity gap. 