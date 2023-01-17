#importing libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.patches import ConnectionPatch

import matplotlib.mlab as mlab 

%matplotlib inline





import scipy.optimize as opt

import warnings

warnings.simplefilter("ignore")

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB





df=pd.read_csv(r"../input/heart-disease-prediction-using-logistic-regression/framingham.csv")

df
df.head(10)
df.tail()
df.columns
df.shape
df.columns.nunique()
df['male'].value_counts()
df['education'].value_counts()
df['currentSmoker'].value_counts()
df['BPMeds'].value_counts()
df['prevalentStroke'].value_counts()
df['diabetes'].value_counts()
df['TenYearCHD'].value_counts()
df.info()
df.isnull().sum()
# percentage of missing data per category

total = df.isnull().sum().sort_values(ascending=False)

percent_total = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100

missing = pd.concat([total, percent_total], axis=1, keys=["Total", "Percentage"])

missing_data = missing[missing['Total']>0]

missing_data
plt.figure(figsize=(9,6))

sns.barplot(x=missing_data.index, y=missing_data['Percentage'], data = missing_data)

plt.title('Percentage of missing data by feature')

plt.xlabel('Features', fontsize=14)

plt.ylabel('Percentage', fontsize=14)

plt.show()
# we can drop education as it doesnt effect heart disease

df = df.drop(['education'], axis=1)
print(df.isnull().sum().sum())

df=df.dropna()

print(df.isnull().sum().sum())

df.shape
df.isna().sum()
#Outliers

cols =['age','BMI','heartRate','sysBP','totChol','diaBP']

plt.title("OUTLIERS VISUALIZATION")

for i in cols:

    df[i]

    sns.distplot(df[i],color='grey')

    plt.show()
df.describe().T
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),linewidths=0.1,annot=True)

# linewidths is white space between boxes and annot gives value

plt.show()
sns.boxplot(y='age',x='TenYearCHD',data=df)
sns.violinplot(y='age',x='TenYearCHD',data=df)
sns.violinplot(y='cigsPerDay',x='TenYearCHD',data=df)
sns.violinplot(y='sysBP',x='TenYearCHD',data=df)
sns.boxplot(y='diaBP',x='TenYearCHD',data=df)
sns.violinplot(y='BMI',x='TenYearCHD',data=df)
sns.boxplot(y='heartRate',x='TenYearCHD',data=df)
sns.countplot(x=df['male'], hue=df['TenYearCHD'])
sns.countplot(x='currentSmoker',data=df,hue='TenYearCHD')
sns.countplot(x='prevalentHyp',data=df,hue='TenYearCHD')
sns.countplot(x='BPMeds',data=df,hue='TenYearCHD')
sns.countplot(x='diabetes',data=df,hue='TenYearCHD')
sns.countplot(x='prevalentStroke',data=df,hue='TenYearCHD')
plt.figure(figsize=(10,10))

sns.boxplot(x='TenYearCHD', y='age', data=df, hue='currentSmoker')
plt.figure(figsize=(10,10))

sns.violinplot(x='TenYearCHD', y='age', data=df, hue='currentSmoker', split=True)
sns.boxplot(y='sysBP',x='prevalentHyp',data=df)
plt.figure(figsize=(20,10))

sns.boxplot(y='diaBP',hue='prevalentHyp',data=df,x='TenYearCHD')

#split=True combines two plots
plt.figure(figsize=(20,10))

sns.violinplot(y='glucose',hue='diabetes',data=df,x='TenYearCHD',split=True)
# plot histogram to see the distribution of the data

fig = plt.figure(figsize = (15,20))

ax = fig.gca()

df.hist(ax = ax)

plt.show()
# Identify the features with the most importance for the outcome variable Heart Disease



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2





X = df.iloc[:,0:14]  

y = df.iloc[:,-1]    



bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)



featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  

print(featureScores.nlargest(11,'Score'))  
featureScores = featureScores.sort_values(by='Score', ascending=False)

featureScores
plt.figure(figsize=(20,5))

sns.barplot(x='Specs', y='Score', data=featureScores, palette = "plasma")

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
df = df[['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male','TenYearCHD']]

df
# Checking for outliers again

df.describe()

sns.pairplot(df)
sns.boxplot(df.totChol)

outliers = df[(df['totChol'] > 500)] 

outliers
#Dropping 2 outliers in cholesterin

df = df.drop(df[df.totChol > 599].index)

sns.boxplot(df.totChol)
df_clean = df
scaler = MinMaxScaler(feature_range=(0,1)) 

scaled_df= pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)
scaled_df.describe()

df.describe()
y = scaled_df['TenYearCHD']

X = scaled_df.drop(['TenYearCHD'], axis = 1)



# divide train test: 60 % - 40 %

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=29)

print(len(X_train))

print(len(X_test))
target_count = scaled_df.TenYearCHD.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')



sns.countplot(scaled_df.TenYearCHD, palette="OrRd")

plt.box(False)

plt.xlabel('Heart Disease No/Yes',fontsize=11)

plt.ylabel('Patient Count',fontsize=11)

plt.title('Count Outcome Heart Disease\n')

plt.savefig('Balance Heart Disease.png')

plt.show()
# Shuffle df

shuffled_df = scaled_df.sample(frac=1,random_state=4)



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
#initialize model

logreg = LogisticRegression()



# fit model

logreg.fit(X_train, y_train)



# prediction = knn.predict(x_test)

normalized_df_logreg_pred = logreg.predict(X_test)



# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, normalized_df_logreg_pred)

print(f"The accuracy score for LogisticRegression is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, normalized_df_logreg_pred)

print(f"The f1 score for LogisticRegression is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, normalized_df_logreg_pred)

print(f"The precision score for LogisticRegression is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, normalized_df_logreg_pred)

print(f"The recall score for LogisticRegression is: {round(recall,3)*100}%")

knn = KNeighborsClassifier(n_neighbors = 2)



#fit model

knn.fit(X_train, y_train)



# prediction = knn.predict(x_test)

normalized_df_knn_pred = knn.predict(X_test)





# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, normalized_df_knn_pred)

print(f"The accuracy score for KNN is: {round(acc,3)*100}%")



f1 = f1_score(y_test, normalized_df_knn_pred)

print(f"The f1 score for KNN is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, normalized_df_knn_pred)

print(f"The precision score for KNN is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, normalized_df_knn_pred)

print(f"The recall score for KNN is: {round(recall,3)*100}%")
#initialize model

dtc_up = DecisionTreeClassifier()



# fit model

dtc_up.fit(X_train, y_train)



normalized_df_dtc_up_pred = dtc_up.predict(X_test)



# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, normalized_df_dtc_up_pred)

print(f"The accuracy score for DTC is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, normalized_df_dtc_up_pred)

print(f"The f1 score for DTC is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, normalized_df_dtc_up_pred)

print(f"The precision score for DTC is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, normalized_df_dtc_up_pred)

print(f"The recall score for DTC is: {round(recall,3)*100}%")
#initialize model

svm = SVC()



#fit model

svm.fit(X_train, y_train)



normalized_df_svm_pred = svm.predict(X_test)



print('Observations:')

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, normalized_df_svm_pred)

print(f"The accuracy score for SVM is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, normalized_df_svm_pred)

print(f"The f1 score for SVM is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, normalized_df_svm_pred)

print(f"The precision score for SVM is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, normalized_df_svm_pred)

print(f"The recall score for SVM is: {round(recall,3)*100}%")
rfc =  RandomForestClassifier()



#fit model

rfc.fit(X_train, y_train)



normalized_df_rfc_pred = rfc.predict(X_test)



print('Observations:')

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, normalized_df_rfc_pred)

print(f"The accuracy score for RFC is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, normalized_df_rfc_pred)

print(f"The f1 score for RFC is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, normalized_df_rfc_pred)

print(f"The precision score for RFC is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, normalized_df_rfc_pred)

print(f"The recall score for RFC is: {round(recall,3)*100}%")
nb =  GaussianNB()



#fit model

nb.fit(X_train, y_train)



normalized_df_nb_pred = nb.predict(X_test)



print('Observations:')

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, normalized_df_nb_pred)

print(f"The accuracy score for Naive Bayes is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, normalized_df_nb_pred)

print(f"The f1 score for Naive Bayes is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, normalized_df_nb_pred)

print(f"The precision score for Naive Bayes is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, normalized_df_nb_pred)

print(f"The recall score for Naive Bayes is: {round(recall,3)*100}%")





data = {'Model':['Logistic Regression','KNN','Decision Tree','SVM','Random Forest','Naive Bayes'],

        'F1 Score':[6.60,12.5,22.7,1.70,13.0,26.0],'Accuracies':[84.89,84.1,74.1,84.7,83.89,81.8],'Recall':[3.40,7.30,24.6,0.89,7.80,20.70],'Precision':[72.70,41.50,21.00,100.00,40.00,35.00]}



# Create DataFrame

df = pd.DataFrame(data)

 

# Print the output.

print(df)
Accuracies=[84.89,84.1,73.2,84.7,83.89,81.8]

Accuracies
plt.figure(figsize=(9,6))

sns.barplot(x='Model', y='Accuracies', data = df)

plt.title('Comparison of accuracy of models')

plt.xlabel('model algorithms', fontsize=14)

plt.ylabel('Accuracy', fontsize=14)

plt.show()
acc_test = logreg.score(X_test, y_test)

print("The accuracy score of the test data is: ",acc_test*100,"%")

acc_train = logreg.score(X_train, y_train)

print("The accuracy score of the training data is: ",round(acc_train*100,2),"%")

cnf_matrix_logreg = confusion_matrix(y_test, normalized_df_logreg_pred)



sns.heatmap(pd.DataFrame(cnf_matrix_logreg), annot=True,cmap="Reds" , fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix Logistic Regression\n', y=1.1)
fpr, tpr, _ = roc_curve(y_test, normalized_df_logreg_pred)

auc = roc_auc_score(y_test, normalized_df_logreg_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.box(False)

plt.title ('ROC CURVE LOGREG')

plt.show()



print(f"The score for the AUC ROC Curve is: {round(auc,3)*100}%")