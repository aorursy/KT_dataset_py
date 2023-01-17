# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt 

plt.rc("font", size=14)



import seaborn as sns

sns.set(style="white")

sns.set(style="darkgrid", color_codes=True)



from scipy.stats import zscore

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score

from sklearn.metrics import roc_auc_score, roc_curve, classification_report
df = pd.read_csv("../input/indian-liver-patient-records/indian_liver_patient.csv")
df.shape
df.dtypes
df.head(5)
df.tail(5)
df.info()
print(df.drop('Dataset', axis=1).dtypes)
print("Dataset : values are {}, dtype is {}".format(df['Dataset'].unique(),

                                                          df['Dataset'].dtype))
df.isna().sum()
print("\nThere are 4 Null/Missing values in the dataset\n")
df[df['Albumin_and_Globulin_Ratio'].isna()]    
# Drop Nan values as there are only 4 NaN's

df.dropna(inplace=True)
df.isna().sum().value_counts()
(df.drop('Gender', axis=1) < 0).sum()
print("\nThere are no Negative values in the dataset\n")
df.duplicated().sum()
df[df.duplicated()]
print("\nThere are 13 duplicate records in the dataset\n")
#Removing Duplicate Rows



df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
#check changed shape

df.shape
#check columns

df.columns
num_columns = ['Age','Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 

               'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 

               'Albumin_and_Globulin_Ratio']
cat_columns = ['Gender','Dataset']
df.describe().T
df.describe().drop('count',axis=0).plot(figsize=(20,8))

plt.show()
#Identifying Outliers in Numeric columns using IQR (Inter Quartile Range) and Q1 (25% Quantile), Q3(75% Quantile).



def identify_outliers(col):    

    q1 = df[col].quantile(0.25)

    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5*iqr

    upper_limit = q3 + 1.5*iqr

    return(col, q1, q3, iqr, lower_limit, upper_limit)
#Checking for Outliers and identifying them by calling identify_outliers() function.

#observations below Q1- 1.5*IQR, or those above Q3 + 1.5*IQR  are defined as outliers.



for col in num_columns :

    col, q1, q3, iqr, lower_limit, upper_limit = identify_outliers(col)

    print("\nColumn name : {}\n Q1 = {} \n Q3 = {}\n IQR = {}".format(col, q1, q3, iqr))

    print(" Lower limit = {}\n Upper limit = {}\n".format(lower_limit, upper_limit))

    outlier_count = len(df.loc[(df[col] < lower_limit) | (df[col] > upper_limit)])

    if outlier_count != 0 :

        print(outlier_count, "OUTLIERS ARE PRESENT in {} column.".format(col))

        print("Outlier datapoints in {} column are:".format(col))

        print(np.array(df.loc[(df[col] < lower_limit) | (df[col] > upper_limit)][col]))

    else:

        print("OUTLIERS ARE NOT PRESENT in {} column\n".format(col))
#Visualizing Outliers in dataset using boxplot



print('\n\t\tBoxplot to check the presence of outliers in numeric columns')

print('\t\t==============================================================\n')

#num_columns = ['Age','Income', 'CCAvg', 'Mortgage']

fig, ax = plt.subplots(3,3,figsize=(15, 10))

for col,subplot in zip(num_columns,ax.flatten()) :

    sns.boxplot(x=df[[col]], width=0.5, color='orange', ax=subplot)

    #subplot.set_title('Boxplot for {}'.format(col))

    subplot.set_xlabel(col)    

plt.show()
df[num_columns].var()
plt.xticks(rotation = 90, fontsize=10)

plt.yticks(fontsize=10)

plt.plot(df[num_columns].var(), color='green', marker='s',linewidth=2, markersize=5)

plt.yscale('log')

plt.show()
fig, ax = plt.subplots(3,3,figsize=(15, 10))

for col,subplot in zip(num_columns,ax.flatten()) :

    ax =sns.distplot(df[col], ax=subplot, hist_kws={'color':'g','alpha':1}, kde_kws={'color':'black', 'lw':2})
# Apart from Dataset which is the Target column there is only one other categorical column, Gender

# Value counts and distribution of Gender column



df.Gender.value_counts()
ax = sns.countplot(df.Gender)
# The Target column is 'Dataset'.

# Value counts and distribution of Target column

df.groupby(by='Dataset').count()
sns.countplot(df['Dataset'], palette = 'plasma')

plt.show()
for col in df.drop('Dataset', axis=1).columns :

    pd.crosstab(df[col], df['Dataset']).plot(kind='bar',color=('b', 'r'), figsize=(20,5))
sns.pairplot(vars=df.drop(['Gender', 'Dataset'], axis=1).columns,hue='Dataset',data=df)

plt.show()
#Dropping categorical column and target for finding correlation

corr = df[num_columns].corr()

corr.style.background_gradient(cmap='YlGnBu')
plt.figure(figsize=(10,10))

sns.heatmap(corr, annot=True, square=True)

plt.show()
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
le.classes_
df['Gender'].value_counts()
X = df.drop('Dataset',axis=1)

y = df['Dataset']
print('Shape of Feture-set : ', X.shape)

print('Shape of Target-set : ', y.shape)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.30, random_state=7)
print("Training Set Shape:\nFeatures : {0}  Target : {1}\n".format(X_train.shape, y_train.shape))

print("Test Set Shape:\nFeatures : {0}  Target : {1}".format(X_test.shape, y_test.shape))
#Standardization using Standard Scaler class of sklearn.preprocessing module



scaler = StandardScaler().fit(X_train)
#Training set transformed to fit Standard Scaler



X_trainS = scaler.transform(X_train)
#Test set transformed to fit Standard Scaler



X_testS = scaler.transform(X_test)
print(X_trainS.mean(), X_trainS.std())

print(X_testS.mean(), X_testS.std())
#DataFrame to store model Performance metrics of all the classification methods

compare_metrics_df = pd.DataFrame(index=('K-NearestNeighbors', 'Logistic Regression', 'Gaussian Naive Bayes'), 

                                  columns=('Trainingset Accuracy', 'Testset Accuracy', 'Precision Score', 

                                           'Recall Score', 'F1 Score', 'ROC_AUC Score'))
compare_metrics_df.index.name = 'Classifier Name'
#Implementing KNN Classifier for default k value 5



knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
#Fit the model to the training set



knn_clf.fit(X_trainS, y_train)
# Predict classes using the built model



yhat_knn = knn_clf.predict(X_testS)
# Model accuracy score using score() function on Training data set



compare_metrics_df.loc['K-NearestNeighbors','Trainingset Accuracy'] = round(knn_clf.score(X_trainS, y_train), 2)

knn_clf.score(X_trainS, y_train)
# Model accuracy score using score() function on Test data set



compare_metrics_df.loc['K-NearestNeighbors','Testset Accuracy'] = round(knn_clf.score(X_testS, y_test), 2)

knn_clf.score(X_testS, y_test)
k_range = 100

mean_train_acc_knn = np.zeros(k_range)

mean_test_acc_knn = np.zeros(k_range)



for n in range(1,k_range+1) :

    KNN = KNeighborsClassifier(n_neighbors=n, weights='distance')

    KNN.fit(X_trainS, y_train)

    mean_train_acc_knn[n-1] = KNN.score(X_trainS, y_train)

    mean_test_acc_knn[n-1] = KNN.score(X_testS, y_test)
print('\nBest test accuracy is {0} for a K value of {1}'.format(mean_test_acc_knn.max(), mean_test_acc_knn.argmax()+1))

print('\nThe train accuracy for best test accuracy is {}'.format(mean_train_acc_knn[mean_test_acc_knn.argmax()+1]))

print('\nThe Best K-value for the classification is K = {}'.format(mean_test_acc_knn.argmax()+1))
confusion_matrix_knn = confusion_matrix(y_test, yhat_knn)

confusion_matrix(y_test, yhat_knn)
print("Accuracy Score: ",accuracy_score(y_test, yhat_knn))

compare_metrics_df.loc['K-NearestNeighbors','Precision Score'] = round(precision_score(y_test, yhat_knn), 2)

print("Precision Score: ",precision_score(y_test, yhat_knn))

compare_metrics_df.loc['K-NearestNeighbors','Recall Score'] = round(recall_score(y_test, yhat_knn), 2)

print("Recall Score: ",recall_score(y_test, yhat_knn))

compare_metrics_df.loc['K-NearestNeighbors','F1 Score'] = round(f1_score(y_test, yhat_knn), 2)

print("F1 Score: ",f1_score(y_test, yhat_knn))

compare_metrics_df.loc['K-NearestNeighbors','ROC_AUC Score'] = round(roc_auc_score(y_test, yhat_knn), 2)

print("ROC_AUC Score: ",roc_auc_score(y_test, yhat_knn))

print("Classification Report\n",classification_report(y_test, yhat_knn))
#Implementing Logistic Regression Classifier



lgr_clf = LogisticRegression(solver='lbfgs', random_state=7)
#Fit the model to the training set



lgr_clf.fit(X_trainS, y_train)
# Predict classes using the built model



yhat_lgr = lgr_clf.predict(X_testS)
# Model accuracy score using score() function on Training data set



compare_metrics_df.loc['Logistic Regression','Trainingset Accuracy'] = round(lgr_clf.score(X_trainS, y_train), 2)

lgr_clf.score(X_trainS, y_train)
# Model accuracy score using score() function on Test data set



compare_metrics_df.loc['Logistic Regression','Testset Accuracy'] = round(lgr_clf.score(X_testS, y_test), 2)

lgr_clf.score(X_testS, y_test)
confusion_matrix_lgr = confusion_matrix(y_test, yhat_lgr)

confusion_matrix(y_test, yhat_lgr)
print("Accuracy Score: ",accuracy_score(y_test, yhat_lgr))

compare_metrics_df.loc['Logistic Regression','Precision Score'] = round(precision_score(y_test, yhat_lgr), 2)

print("Precision Score: ",precision_score(y_test, yhat_lgr))

compare_metrics_df.loc['Logistic Regression','Recall Score'] = round(recall_score(y_test, yhat_lgr), 2)

print("Recall Score: ",recall_score(y_test, yhat_lgr))

compare_metrics_df.loc['Logistic Regression','F1 Score'] = round(f1_score(y_test, yhat_lgr), 2)

print("F1 Score: ",f1_score(y_test, yhat_lgr))

compare_metrics_df.loc['Logistic Regression','ROC_AUC Score'] = round(roc_auc_score(y_test, yhat_lgr), 2)

print("ROC_AUC Score: ",roc_auc_score(y_test, yhat_lgr))

print("Classification Report\n",classification_report(y_test, yhat_lgr))
#Implementing Logistic Regression Classifier



gnb_clf = GaussianNB()
#Fit the model to the training set



gnb_clf.fit(X_trainS, y_train)
# Predict classes using the built model



yhat_gnb = gnb_clf.predict(X_testS)
# Model accuracy score using score() function on Training data set



compare_metrics_df.loc['Gaussian Naive Bayes','Trainingset Accuracy'] = round(gnb_clf.score(X_trainS, y_train), 2)

gnb_clf.score(X_trainS, y_train)
# Model accuracy score using score() function on Test data set



compare_metrics_df.loc['Gaussian Naive Bayes','Testset Accuracy'] = round(gnb_clf.score(X_testS, y_test), 2)

gnb_clf.score(X_testS, y_test)
confusion_matrix_gnb = confusion_matrix(y_test, yhat_gnb)

confusion_matrix(y_test, yhat_gnb)
print("Accuracy Score: ",accuracy_score(y_test, yhat_gnb))

compare_metrics_df.loc['Gaussian Naive Bayes','Precision Score'] = round(precision_score(y_test, yhat_gnb), 2)

print("Precision Score: ",precision_score(y_test, yhat_gnb))

compare_metrics_df.loc['Gaussian Naive Bayes','Recall Score'] = round(recall_score(y_test, yhat_gnb), 2)

print("Recall Score: ",recall_score(y_test, yhat_gnb))

compare_metrics_df.loc['Gaussian Naive Bayes','F1 Score'] = round(f1_score(y_test, yhat_gnb), 2)

print("F1 Score: ",f1_score(y_test, yhat_gnb))

compare_metrics_df.loc['Gaussian Naive Bayes','ROC_AUC Score'] = round(roc_auc_score(y_test, yhat_gnb), 2)

print("ROC_AUC Score: ",roc_auc_score(y_test, yhat_gnb))

print("Classification Report\n",classification_report(y_test, yhat_gnb))
compare_metrics_df
print("Confusion Matrix of all the 3 models")

print("====================================")

print("\nK-Nearest Neighbors:\n")

print(confusion_matrix_knn)

print("\nLogistic Regression:\n")

print(confusion_matrix_lgr)

print("\nGaussian Naive Bayes:\n")

print(confusion_matrix_gnb)