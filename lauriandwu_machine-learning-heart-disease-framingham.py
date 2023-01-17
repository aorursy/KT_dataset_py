# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# importing libraries and magic functions



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format ='retina'

%matplotlib inline
# read data

df = pd.read_csv('/kaggle/input/framingham-heart-study-dataset/framingham.csv')



# first glimpse at data

df.head(20)



# data shape

df.shape



# data types

df.dtypes
# check for dupicates

duplicate_df = df[df.duplicated()]

duplicate_df
# checking for missing values

df.isna().sum()

null = df[df.isna().any(axis=1)]

null
# checking distributions using histograms

fig = plt.figure(figsize = (15,20))

ax = fig.gca()

df.hist(ax = ax)
# checking which features are correlated with each other and are correlated with the outcome variable

df_corr = df.corr()

sns.heatmap(df_corr)
# Dropping columns education and glucose

df = df.drop(['education'], axis=1)
# Checking for more missing data 

df.isna().sum()
# Dropping all rows with missing data

df = df.dropna()

df.isna().sum()

df.columns
# Identify the features with the most importance for the outcome variable Heart Disease



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



# separate independent & dependent variables

X = df.iloc[:,0:14]  #independent columns

y = df.iloc[:,-1]    #target column i.e price range



# apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)



#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(11,'Score'))  #print 10 best features
featureScores = featureScores.sort_values(by='Score', ascending=False)

featureScores
# visualizing feature selection

plt.figure(figsize=(20,5))

sns.barplot(x='Specs', y='Score', data=featureScores, palette = "GnBu_d")

plt.box(False)

plt.title('Feature importance', fontsize=16)

plt.xlabel('\n Features', fontsize=14)

plt.ylabel('Importance \n', fontsize=14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
# selecting the 10 most impactful features for the target variable

features_list = featureScores["Specs"].tolist()[:10]

features_list
# Create new dataframe with selected features



df = df[['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male','TenYearCHD']]

df.head()
# Checking correlation again

df_corr = df.corr()

sns.heatmap(df_corr)
# Checking for outliers

df.describe()

sns.pairplot(df)
# Zooming into cholesterin outliers



sns.boxplot(df.totChol)

outliers = df[(df['totChol'] > 500)] 

outliers
# Dropping 2 outliers in cholesterin

df = df.drop(df[df.totChol > 599].index)

sns.boxplot(df.totChol)
df_clean = df
scaler = MinMaxScaler(feature_range=(0,1)) 



#assign scaler to column:

df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)

df_scaled.describe()

df.describe()
# clarify what is y and what is x label

y = df_scaled['TenYearCHD']

X = df_scaled.drop(['TenYearCHD'], axis = 1)



# divide train test: 80 % - 20 %

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=29)
len(X_train)

len(X_test)
# Checking balance of outcome variable

target_count = df_scaled.TenYearCHD.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



sns.countplot(df_scaled.TenYearCHD, palette="OrRd")

plt.box(False)

plt.xlabel('Heart Disease No/Yes',fontsize=11)

plt.ylabel('Patient Count',fontsize=11)

plt.title('Count Outcome Heart Disease\n')

plt.savefig('Balance Heart Disease.png')

plt.show()
# Shuffle df

shuffled_df = df_scaled.sample(frac=1,random_state=4)



# Put all the fraud class in a separate dataset.

CHD_df = shuffled_df.loc[shuffled_df['TenYearCHD'] == 1]



#Randomly select 492 observations from the non-fraud (majority class)

non_CHD_df = shuffled_df.loc[shuffled_df['TenYearCHD'] == 0].sample(n=611,random_state=42)



# Concatenate both dataframes again

normalized_df = pd.concat([CHD_df, non_CHD_df])



# check new class counts

normalized_df.TenYearCHD.value_counts()



# plot new count

sns.countplot(normalized_df.TenYearCHD, palette="OrRd")

plt.box(False)

plt.xlabel('Heart Disease No/Yes',fontsize=11)

plt.ylabel('Patient Count',fontsize=11)

plt.title('Count Outcome Heart Disease after Resampling\n')

#plt.savefig('Balance Heart Disease.png')

plt.show()
y_train = normalized_df['TenYearCHD']

X_train = normalized_df.drop('TenYearCHD', axis=1)



from sklearn.pipeline import Pipeline



classifiers = [LogisticRegression(),SVC(),DecisionTreeClassifier(),KNeighborsClassifier(2)]



for classifier in classifiers:

    pipe = Pipeline(steps=[('classifier', classifier)])

    pipe.fit(X_train, y_train)   

    print("The accuracy score of {0} is: {1:.2f}%".format(classifier,(pipe.score(X_test, y_test)*100)))

# logistic regression again with the balanced dataset



normalized_df_reg = LogisticRegression().fit(X_train, y_train)



normalized_df_reg_pred = normalized_df_reg.predict(X_test)



# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, normalized_df_reg_pred)

print(f"The accuracy score for LogReg is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, normalized_df_reg_pred)

print(f"The f1 score for LogReg is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, normalized_df_reg_pred)

print(f"The precision score for LogReg is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, normalized_df_reg_pred)

print(f"The recall score for LogReg is: {round(recall,3)*100}%")
# plotting confusion matrix LogReg



cnf_matrix_log = confusion_matrix(y_test, normalized_df_reg_pred)



sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True,cmap="Reds" , fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix Logistic Regression\n', y=1.1)
# Support Vector Machine



#initialize model

svm = SVC()



#fit model

svm.fit(X_train, y_train)



normalized_df_svm_pred = svm.predict(X_test)



# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, normalized_df_svm_pred)

print(f"The accuracy score for SVM is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, normalized_df_svm_pred)

print(f"The f1 score for SVM is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, normalized_df_svm_pred)

print(f"The precision score for SVM is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, normalized_df_svm_pred)

print(f"The recall score for SVM is: {round(recall,3)*100}%")

# plotting confusion matrix SVM



cnf_matrix_svm = confusion_matrix(y_test, normalized_df_svm_pred)



sns.heatmap(pd.DataFrame(cnf_matrix_svm), annot=True,cmap="Reds" , fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix SVM\n', y=1.1)
# Decision Tree



#initialize model

dtc_up = DecisionTreeClassifier()



# fit model

dtc_up.fit(X_train, y_train)



normalized_df_dtc_pred = dtc_up.predict(X_test)



# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, normalized_df_dtc_pred)

print(f"The accuracy score for DTC is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, normalized_df_dtc_pred)

print(f"The f1 score for DTC is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, normalized_df_dtc_pred)

print(f"The precision score for DTC is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, normalized_df_dtc_pred)

print(f"The recall score for DTC is: {round(recall,3)*100}%")
# plotting confusion matrix Decision Tree



cnf_matrix_dtc = confusion_matrix(y_test, normalized_df_dtc_pred)



sns.heatmap(pd.DataFrame(cnf_matrix_dtc), annot=True,cmap="Reds" , fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix Decision Tree\n', y=1.1)

# KNN Model



#initialize model

knn = KNeighborsClassifier(n_neighbors = 2)



#fit model

knn.fit(X_train, y_train)



# prediction = knn.predict(x_test)

normalized_df_knn_pred = knn.predict(X_test)





# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, normalized_df_knn_pred)

print(f"The accuracy score for KNN is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, normalized_df_knn_pred)

print(f"The f1 score for KNN is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, normalized_df_knn_pred)

print(f"The precision score for KNN is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, normalized_df_knn_pred)

print(f"The recall score for KNN is: {round(recall,3)*100}%")
# Check overfit of the KNN model

# accuracy test and train

acc_test = knn.score(X_test, y_test)

print("The accuracy score of the test data is: ",acc_test*100,"%")

acc_train = knn.score(X_train, y_train)

print("The accuracy score of the training data is: ",round(acc_train*100,2),"%")



# Perform cross validation

'''Cross Validation is used to assess the predictive performance of the models and and to judge 

how they perform outside the sample to a new data set'''



cv_results = cross_val_score(knn, X, y, cv=5) 



print ("Cross-validated scores:", cv_results)

print("The Accuracy of Model with Cross Validation is: {0:.2f}%".format(cv_results.mean() * 100))
# plotting confusion matrix KNN



cnf_matrix_knn = confusion_matrix(y_test, normalized_df_knn_pred)



ax= plt.subplot()

sns.heatmap(pd.DataFrame(cnf_matrix_knn), annot=True,cmap="Reds" , fmt='g')



ax.set_xlabel('Predicted ');ax.set_ylabel('True'); 

# AU ROC CURVE KNN

'''the AUC ROC Curve is a measure of performance based on plotting the true positive and false positive rate 

and calculating the area under that curve.The closer the score to 1 the better the algorithm's ability to 

distinguish between the two outcome classes.'''



fpr, tpr, _ = roc_curve(y_test, normalized_df_knn_pred)

auc = roc_auc_score(y_test, normalized_df_knn_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.box(False)

plt.title ('ROC CURVE KNN')

plt.show()



print(f"The score for the AUC ROC Curve is: {round(auc,3)*100}%")
def start_questionnaire():

    my_predictors = []

    parameters=['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male']

    

    print('Input Patient Information:')

    

    age = input("Patient's age: >>> ") 

    my_predictors.append(age)

    male = input("Patient's gender. male=1, female=0: >>> ") 

    my_predictors.append(male)

    cigsPerDay = input("Patient's smoked cigarettes per day: >>> ") 

    my_predictors.append(cigsPerDay)

    sysBP = input("Patient's systolic blood pressure: >>> ") 

    my_predictors.append(sysBP)

    diaBP = input("Patient's diastolic blood pressure: >>> ")

    my_predictors.append(diaBP)

    totChol = input("Patient's cholesterin level: >>> ") 

    my_predictors.append(totChol)

    prevalentHyp = input("Was Patient hypertensive? Yes=1, No=0 >>> ") 

    my_predictors.append(prevalentHyp)

    diabetes = input("Did Patient have diabetes? Yes=1, No=0 >>> ") 

    my_predictors.append(diabetes)

    glucose = input("What is the Patient's glucose level? >>> ") 

    my_predictors.append(diabetes)

    BPMeds = input("Has Patient been on Blood Pressure Medication? Yes=1, No=0 >>> ")

    my_predictors.append(BPMeds)

    

    my_data = dict(zip(parameters, my_predictors))

    my_df = pd.DataFrame(my_data, index=[0])

    scaler = MinMaxScaler(feature_range=(0,1)) 

   

    # assign scaler to column:

    my_df_scaled = pd.DataFrame(scaler.fit_transform(my_df), columns=my_df.columns)

    my_y_pred = knn.predict(my_df)

    print('\n')

    print('Result:')

    if my_y_pred == 1:

        print("The patient will develop a Heart Disease.")

    if my_y_pred == 0:

        print("The patient will not develop a Heart Disease.")

        

start_questionnaire()