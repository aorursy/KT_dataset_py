import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(color_codes=True)



from scipy import stats

from scipy.stats import zscore



from sklearn import metrics

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression



from sklearn.naive_bayes import GaussianNB  # using Gaussian algorithm for Naive Bayes



from sklearn.neighbors import KNeighborsClassifier



from sklearn.svm import SVC  # using Gaussian Kernel or Radial Basis Function
data = pd.read_csv('../input/bank-personal-loan-modelling/Bank_Personal_Loan_Modelling.csv')
data.head()
personal_loan = data['Personal Loan']

data.drop(labels=['Personal Loan'], axis=1, inplace = True)

data.insert(13, 'Personal Loan', personal_loan)

data.head()
rows_count, columns_count = data.shape

print("Total number of rows :", rows_count)

print("Total number of columns :", columns_count)
data.columns
data.info()
def missing_check(df):

    total = df.isnull().sum().sort_values(ascending=False) # Total number of null values

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) # Percentage of values that are null

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) # Putting the above two together

    return missing_data
missing_check(data)
data.isnull().sum().sum()
data.describe().transpose()
# Total number of observations with negative values in their Experience column

negExp = data.Experience < 0

negExp.value_counts()
# Checking all the negative values present in the Experience column

data[data['Experience'] < 0]['Experience'].value_counts()
quantitativeAttr = ['Age', 'Income', 'CCAvg', 'Mortgage']

# Create an instance of the PairGrid class.

grid = sns.PairGrid(data=data, y_vars='Experience', x_vars=quantitativeAttr, height = 4)

grid.map(sns.regplot, line_kws={"color": "cyan"});
# Create two different dataframes with records where experience value is greater than 0 and lesser than 0 respectively

# Get the list of Customer IDs from the dataframe containing records with negative experince values

# Iterating over the Customer ID list

    # Get the Age and Education level for the corresponding ID from the negative experience dataframe

    # Filter the records form the positive experience dataframe based on the obtained age and education value

        # calculate the median experience value from the filtered dataframe and store in "experience" variable

        # if the filtered dataframe is empty, filter the negative experience dataframe and obtain the median experience value

    # Replace the negative experience with the absolute value of the median "experience" value





df_positive_experience = data[data['Experience'] > 0]

df_negative_experience = data[data['Experience'] < 0]

negative_experience_id_list = df_negative_experience['ID'].tolist()



for id in negative_experience_id_list:

    age = data.loc[np.where(data['ID']==id)]['Age'].tolist()[0]

    education = data.loc[np.where(data['ID']==id)]['Education'].tolist()[0]

    positive_experience_filtered = df_positive_experience[(df_positive_experience['Age'] == age) &

                                                         (df_positive_experience['Education'] == education)]

    if positive_experience_filtered.empty:

        negative_experience_filtered = df_negative_experience[(df_negative_experience['Age'] == age) &

                                                         (df_negative_experience['Education'] == education)]

        experience = round(negative_experience_filtered['Experience'].median())

    else:

        experience = round(positive_experience_filtered['Experience'].median())

    data.loc[data.ID == id, 'Experience'] = abs(experience)
data[data['Experience'] < 0]['Experience'].count()
data.Experience.describe()
sns.distplot(data['ID'])

plt.title('ID Distribution with KDE');
sns.distplot(data['Age'])

plt.title('Age Distribution with KDE');
sns.distplot(data['Experience'])

plt.title('Experience Distribution with KDE');
sns.distplot(data['Income'])

plt.title('Income Distribution with KDE');
data['Income'].skew()
sns.distplot(data['ZIP Code'])

plt.title('ZIP Code Distribution with KDE');
sns.countplot(data['Family'])

plt.title('Family Distribution with count for each family size');
sns.distplot(data['CCAvg'])

plt.title('CCAvg Distribution with KDE');
data['CCAvg'].skew()
sns.countplot(data['Education'])

plt.title('Education Distribution with count for each education level');
sns.distplot(data['Mortgage'])

plt.title('Mortgage Distribution with KDE');
data['Mortgage'].skew()
sns.countplot(data['Securities Account'])

plt.title('Securities Account Distribution with a Yes/No count');
sns.countplot(data['CD Account'])

plt.title('CD Account Distribution with a Yes/No count');
sns.countplot(data['Online'])

plt.title('Online Distribution with a Yes/No count');
sns.countplot(data['CreditCard'])

plt.title('CreditCard Distribution with a Yes/No count');
df_cc = data['CreditCard']

df_cc = df_cc.astype({'CreditCard': 'float64'})

sns.distplot(df_cc);
n_true = len(data.loc[data['Personal Loan'] == True])

n_false = len(data.loc[data['Personal Loan'] == False])

print('Number of customers who accepted the PL offer: {0} ({1:2.2f}%)'

      .format(n_true, (n_true / (n_true + n_false)) * 100))

print('Number of customers who did not accept the PL offer: {0} ({1:2.2f}%)'

      .format(n_false, (n_false / (n_true + n_false)) * 100))
loan_acceptance_count = pd.DataFrame(data['Personal Loan'].value_counts()).reset_index()

loan_acceptance_count.columns = ['Labels', 'Personal Loan']

loan_acceptance_count
# Creating dataset 

pie_labels = loan_acceptance_count['Labels']

pie_labels = ['Not Accepted' if x == 0 else 'Accepted' for x in pie_labels]

  

pie_data = loan_acceptance_count['Personal Loan'] 



# Creating explode data 

explode = (0, 0.15) 



# Wedge properties 

wp = { 'linewidth' : 1, 'edgecolor' : '#666666' }



# Creating autocpt arguments 

def func(pct, allvalues): 

    absolute = int(np.round(pct / 100.*np.sum(allvalues)))

    return "{:.1f}%\n({:d})".format(pct, absolute)



# Creating plot 

fig, ax = plt.subplots(figsize =(10, 5))



ax.pie(pie_data,  

       autopct = lambda pct: func(pct, pie_data), 

       explode = explode,  

       labels = pie_labels, 

       shadow = True, 

       startangle = 70, 

       wedgeprops = wp)



ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title('Personal Loan Acceptance Percentage', size=19)

plt.show();
sns.pairplot(data.iloc[:,1:]);
data.corr()
plt.figure(figsize=(15,7))

plt.title('Correlation of Attributes', size=15)

sns.heatmap(data.corr(), annot=True, linewidths=3, fmt='.3f', center=1);
plt.figure(figsize=(10,5))

sns.boxplot(x='Education', y='Income', hue='Personal Loan', data=data);
plt.figure(figsize=(10,5))

sns.boxplot(x='Education', y='Mortgage', hue='Personal Loan', data=data);
plt.figure(figsize=(10,5))

sns.barplot(x='Personal Loan', y='CCAvg', data=data);
plt.figure(figsize=(10,5))

sns.countplot(x='CD Account', data=data, hue='Personal Loan');
# Make a copy of dataset before doing any changes to the original data

modelData = data.copy()
modelData.drop(['ID', 'ZIP Code'], axis=1, inplace=True)
X = modelData.drop(['Personal Loan'], axis=1)    # Predictor(Independent) Feature columns

y = modelData['Personal Loan']                   # Target Feature column
# Split X and y into training and test set in 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)    # 1 is just any random seed number
X_train.head()
LRM = LogisticRegression(solver='liblinear')
LRM.fit(X_train, y_train)
coeff_df = pd.DataFrame(LRM.coef_)

coeff_df['Intercept'] = LRM.intercept_

print(coeff_df)
logistic_training_predict = LRM.predict(X_train)



print('Logistic Regression Model In-Sample (Training Set) Accuracy: {0:.4f}'.format(metrics.accuracy_score(y_train, 

                                                                                            logistic_training_predict)))

print('')
logistic_test_predict = LRM.predict(X_test)



LRM_accuracy = metrics.accuracy_score(y_test, logistic_test_predict)



print('Logistic Regression Model Out-Sample (Test Set) Accuracy: {0:.4f}'.format(LRM_accuracy))

print('')
logistic_cm = metrics.confusion_matrix(y_test, logistic_test_predict, labels=[1,0])

print(logistic_cm)
logistic_cm_df = pd.DataFrame(logistic_cm, index = [i for i in ["1","0"]], columns = [i for i in ["Predict 1", "Predict 0"]])

plt.figure(figsize=(7,5))

plt.title('Confusion Matrix for Logistic Regression Model', size=15)

sns.heatmap(logistic_cm_df, annot=True, fmt='g'); # fmt='g' (format) is used to get rid of scientific formats
print('Logistic Regression Model - Classification Report')

print('')

print(metrics.classification_report(y_test, logistic_test_predict, labels=[1,0]))
GNBM = GaussianNB()
GNBM.fit(X_train, y_train)
nB_training_predict = GNBM.predict(X_train)



print('Naive Bayes Model In-Sample (Training Set) Accuracy: {0:.4f}'.format(metrics.accuracy_score(y_train, 

                                                                                            nB_training_predict)))

print('')
nB_test_predict = GNBM.predict(X_test)



GNBM_accuracy = metrics.accuracy_score(y_test, nB_test_predict)



print('Naive Bayes Model Out-Sample (Test Set) Accuracy: {0:.4f}'.format(GNBM_accuracy))

print('')
nB_cm = metrics.confusion_matrix(y_test, nB_test_predict, labels=[1,0])

print(nB_cm)
nB_cm_df = pd.DataFrame(nB_cm, index = [i for i in ["1","0"]], columns = [i for i in ["Predict 1", "Predict 0"]])

plt.figure(figsize=(7,5))

plt.title('Confusion Matrix for Naive Bayes Model', size=15)

sns.heatmap(nB_cm_df, annot=True, fmt='g');
print('Naive Bayes Model - Classification Report')

print('')

print(metrics.classification_report(y_test, nB_test_predict, labels=[1,0]))
X_train_scaled = X_train.apply(zscore)

X_test_scaled = X_test.apply(zscore)
X_train_scaled.describe()
kNNM = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')
kNNM.fit(X_train_scaled, y_train)
kNN_test_predict = kNNM.predict(X_test_scaled)



print('kNN Model Accuracy with k = 5: {0:.4f}'.format(metrics.accuracy_score(y_test, kNN_test_predict)))

print('')
error_rate = []

for k in range(1,50):

 kNN_k_test = KNeighborsClassifier(n_neighbors = k, weights = 'distance')

 kNN_k_test.fit(X_train_scaled,y_train)

 pred_k = kNN_k_test.predict(X_test_scaled)

 error_rate.append(np.mean(pred_k != y_test))



plt.figure(figsize=(10,6))

plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', 

         marker='o',markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. k Value')

plt.xlabel('k')

plt.ylabel('Error Rate')

print("Minimum error:-",min(error_rate),"at k =",error_rate.index(min(error_rate)))
acc = []

for k in range(1,50):

    kNN_acc_test = KNeighborsClassifier(n_neighbors = k, weights = 'distance')

    kNN_acc_test.fit(X_train_scaled,y_train)

    pred_acc = kNN_acc_test.predict(X_test_scaled)

    acc.append(metrics.accuracy_score(y_test, pred_acc))

    

plt.figure(figsize=(10,6))

plt.plot(range(1,50),acc,color = 'blue',linestyle='dashed', 

         marker='o',markerfacecolor='red', markersize=10)

plt.title('Accuracy vs. k Value')

plt.xlabel('k')

plt.ylabel('Accuracy')

print("Maximum accuracy:-",max(acc),"at k =",acc.index(max(acc)))
optimal_k = 3

kNNM_optimal = KNeighborsClassifier(n_neighbors = optimal_k, weights = 'distance')

kNNM_optimal.fit(X_train_scaled, y_train)

kNN_test_predict_optimal = kNNM_optimal.predict(X_test_scaled)

kNNM_accuracy = metrics.accuracy_score(y_test, kNN_test_predict_optimal)



print('kNN Model Accuracy with k ={0:2d} :- {1:.4f}'.format(optimal_k, kNNM_accuracy))

print('')
kNN_cm = metrics.confusion_matrix(y_test, kNN_test_predict_optimal, labels=[1,0])

print(kNN_cm)
kNN_cm_df = pd.DataFrame(kNN_cm, index = [i for i in ["1","0"]], columns = [i for i in ["Predict 1", "Predict 0"]])

plt.figure(figsize=(7,5))

plt.title('Confusion Matrix for kNN Model', size=15)

sns.heatmap(kNN_cm_df, annot=True, fmt='g');
print('kNN Model - Classification Report')

print('')

print(metrics.classification_report(y_test, kNN_test_predict_optimal, labels=[1,0]))
SVCM = SVC(gamma=0.025, C=3)
SVCM.fit(X_train, y_train)
svc_test_predict = SVCM.predict(X_test)



SVCM_accuracy = metrics.accuracy_score(y_test, svc_test_predict)



print('SVC Model Accuracy: {0:.4f}'.format(SVCM_accuracy))

print('')
svc_cm = metrics.confusion_matrix(y_test, svc_test_predict, labels=[1,0])

print(svc_cm)
svc_cm_df = pd.DataFrame(svc_cm, index = [i for i in ["1","0"]], columns = [i for i in ["Predict 1", "Predict 0"]])

plt.figure(figsize=(7,5))

plt.title('Confusion Matrix for SVC Model', size=15)

sns.heatmap(svc_cm_df, annot=True, fmt='g');
print('SVC Model - Classification Report')

print('')

print(metrics.classification_report(y_test, svc_test_predict, labels=[1,0]))
models = ['Logistic Regression', 'Naive Bayes', 'kNN', 'SVM']

model_accuracy_scores = [LRM_accuracy, GNBM_accuracy, kNNM_accuracy, SVCM_accuracy]

comp_df = pd.DataFrame([model_accuracy_scores], index=['Accuracy'], columns=models)

comp_df
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])



# Logisitic Regression

LRM_fpr, LRM_tpr, _ = metrics.roc_curve(y_test, logistic_test_predict)

LRM_auc = metrics.roc_auc_score(y_test, logistic_test_predict)

result_table = result_table.append({'classifiers':'Logistic Regression',

                                        'fpr':LRM_fpr, 

                                        'tpr':LRM_tpr, 

                                        'auc':LRM_auc}, ignore_index=True)



# Naive Bayes

GNBM_fpr, GNBM_tpr, _ = metrics.roc_curve(y_test, nB_test_predict)

GNBM_auc = metrics.roc_auc_score(y_test, nB_test_predict)

result_table = result_table.append({'classifiers':'Naive Bayes',

                                        'fpr':GNBM_fpr, 

                                        'tpr':GNBM_tpr, 

                                        'auc':GNBM_auc}, ignore_index=True)





# kNN

kNNM_fpr, kNNM_tpr, _ = metrics.roc_curve(y_test, kNN_test_predict_optimal)

kNNM_auc = metrics.roc_auc_score(y_test, kNN_test_predict_optimal)

result_table = result_table.append({'classifiers':'k-Nearest Neighbors',

                                        'fpr':kNNM_fpr, 

                                        'tpr':kNNM_tpr, 

                                        'auc':kNNM_auc}, ignore_index=True)





# SVM

SVCM_fpr, SVCM_tpr, _ = metrics.roc_curve(y_test, svc_test_predict)

SVCM_auc = metrics.roc_auc_score(y_test, svc_test_predict)

result_table = result_table.append({'classifiers':'Support Vector Machines',

                                        'fpr':SVCM_fpr, 

                                        'tpr':SVCM_tpr, 

                                        'auc':SVCM_auc}, ignore_index=True)



# Set name of the classifiers as index labels

result_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(8,6))



for i in result_table.index:

    plt.plot(result_table.loc[i]['fpr'], 

             result_table.loc[i]['tpr'], 

             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    

plt.plot([0,1], [0,1], color='orange', linestyle='--')



plt.xticks(np.arange(0.0, 1.1, step=0.1))

plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=15)



plt.yticks(np.arange(0.0, 1.1, step=0.1))

plt.ylabel("True Positive Rate (Recall)", fontsize=15)



plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)

plt.legend(prop={'size':13}, loc='lower right')



plt.show();