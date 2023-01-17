# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split



import statsmodels.api as sm



from scipy import stats



from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

# To model the Gaussian Navie Bayes classifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import accuracy_score # Performance measure – Accuracy



from sklearn import preprocessing
df  = pd.read_csv('/kaggle/input/bank-personal-loan-modellingthera-bank/Bank_Personal_Loan_Modelling.csv')

df.head()
personal_loan = df['Personal Loan']

df.drop(['Personal Loan'], axis=1, inplace = True)

df['Personal Loan'] = personal_loan

df.head(5)
rows_count, columns_count = df.shape

print('Total Number of rows :', rows_count)

print('Total Number of columns :', columns_count)
df.dtypes
df.info()
df.isnull().sum() 
df.isnull().values.any()
sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')
df.nunique()
df.describe().T
### Five point summary of  attributes and label :-
df_transpose = df.describe().T

df_transpose[['min', '25%', '50%', '75%', 'max']]
sns.pairplot(df.iloc[:,1:]) 
# Checking the negative values

df[df['Experience'] < 0]['Experience'].value_counts()
# Total records of negative experience

df[df['Experience'] < 0]['Experience'].count()
quantitiveVar = ['Age', 'Income', 'Income', 'CCAvg', 'Mortgage']

expGrid = sns.PairGrid(df, y_vars = 'Experience', x_vars = quantitiveVar)

expGrid.map(sns.regplot)
df_Possitive_Experience = df[df['Experience'] > 0]

df_Negative_Experience =  df[df['Experience'] < 0]

df_Negative_Experience_List = df_Negative_Experience['ID'].tolist()



for id in df_Negative_Experience_List:

    age_values = df.loc[np.where(df['ID']==id)]["Age"].tolist()[0]

    education_values = df.loc[np.where(df['ID']==id)]["Education"].tolist()[0]

    possitive_Experience_Filtered = df_Possitive_Experience[(df_Possitive_Experience['Age'] == age_values) & (df_Possitive_Experience['Education'] == education_values)]

    if possitive_Experience_Filtered.empty :

        negative_Experience_Filtered = df_Negative_Experience[(df_Negative_Experience['Age'] == age_values) & (df_Negative_Experience['Education'] == education_values)]

        exp = round(negative_Experience_Filtered['Experience'].median())

    else:

        exp = round(possitive_Experience_Filtered['Experience'].median())

    df.loc[df.loc[np.where(df['ID']==id)].index, 'Experience'] = abs(exp)
# Total records of negative experience

df[df['Experience'] < 0]['Experience'].count()
df.Experience.describe()
sns.distplot(df['ID'])
sns.distplot(df['Age'])
sns.distplot(df['Experience'])
sns.distplot(df['Income'])
sns.distplot(df['ZIP Code'])
sns.distplot(df['CCAvg'])
sns.distplot(df['Education'])
sns.distplot(df['Mortgage'])
sns.distplot(df['Online'])
sns.distplot(df['CreditCard'])
loan_counts = pd.DataFrame(df["Personal Loan"].value_counts()).reset_index()

loan_counts.columns =["Labels","Personal Loan"]

loan_counts
fig1, ax1 = plt.subplots()

explode = (0, 0.15)

ax1.pie(loan_counts["Personal Loan"], explode=explode, labels=loan_counts["Labels"], autopct='%1.1f%%',

        shadow=True, startangle=70)

ax1.axis('equal')  

plt.title("Personal Loan Percentage")

plt.show()
sns.catplot(x='Family', y='Income', hue='Personal Loan', data = df, kind='swarm')
sns.boxplot(x='Education', y='Income', hue='Personal Loan', data = df)
sns.boxplot(x="Education", y='Mortgage', hue="Personal Loan", data=df)
sns.countplot(x="Securities Account", data=df,hue="Personal Loan")
sns.countplot(x='Family',data=df,hue='Personal Loan')
sns.countplot(x='CD Account',data=df,hue='Personal Loan')
sns.boxplot(x="CreditCard", y='CCAvg', hue="Personal Loan", data=df)

sns.catplot(x='Age', y='Experience', hue='Personal Loan', data = df, height=8.27, aspect=11/5)
plt.figure(figsize=(10,4))

sns.distplot(df[df["Personal Loan"] == 0]['CCAvg'], color = 'r',label='Personal Loan=0')

sns.distplot(df[df["Personal Loan"] == 1]['CCAvg'], color = 'b',label='Personal Loan=1')

plt.legend()

plt.title("CCAvg Distribution")
print('Credit card spending of Non-Loan customers: ',df[df['Personal Loan'] == 0]['CCAvg'].median()*1000)

print('Credit card spending of Loan customers    : ', df[df['Personal Loan'] == 1]['CCAvg'].median()*1000)
plt.figure(figsize=(10,4))

sns.distplot(df[df["Personal Loan"] == 0]['Income'], color = 'r',label='Personal Loan=0')

sns.distplot(df[df["Personal Loan"] == 1]['Income'], color = 'b',label='Personal Loan=1')

plt.legend()

plt.title("Income Distribution")
df.boxplot(return_type='axes', figsize=(20,5))
plt.figure(figsize = (15,7))

plt.title('Correlation of Attributes', y=1.05, size=19)

sns.heatmap(df.corr(), cmap='plasma',annot=True, fmt='.2f')
df.head(1)
df = df.drop(['ID','ZIP Code'], axis=1)
df.head(1)
loan_with_experience = df

loan_without_experience = df.drop(['Experience'], axis=1)
print('Columns With Experience : ', loan_with_experience.columns)

print('Columns Without Experience : ', loan_without_experience.columns)
# From Exprenece Dataframe:

X_Expr = loan_with_experience.drop('Personal Loan', axis=1)

Y_Expr = loan_with_experience[['Personal Loan']]
# From Exprenece Dataframe:

X_Without_Expr = loan_without_experience.drop('Personal Loan', axis=1)

Y_Without_Expr = loan_without_experience[['Personal Loan']]
# From Experience Dataframe:

X_Expr_train, X_Expr_test, y_Expr_train, y_Expr_test = train_test_split(X_Expr, Y_Expr, test_size=0.30, random_state=1)

print('x train data {}'.format(X_Expr_train.shape))

print('y train data {}'.format(y_Expr_train.shape))

print('x test data  {}'.format(X_Expr_test.shape))

print('y test data  {}'.format(y_Expr_test.shape))
# From Without Experience Dataframe:

X_train, X_test, y_train, y_test = train_test_split(X_Without_Expr, Y_Without_Expr, test_size=0.30, random_state=1)

print('x train data {}'.format(X_train.shape))

print('y train data {}'.format(y_train.shape))

print('x test data  {}'.format(X_test.shape))

print('y test data  {}'.format(y_test.shape))
#X_Exp_train, X_Exp_test, y_Exp_train, y_Exp_test

logreg_expr_model = LogisticRegression()

logreg_expr_model.fit(X_Expr_train, y_Expr_train)

print(logreg_expr_model , '\n')



# Predicting for test set

logreg_expr_y_predicted = logreg_expr_model.predict(X_Expr_test)

logreg_expr_score = logreg_expr_model.score(X_Expr_test, y_Expr_test)

logreg_expr_accuracy = accuracy_score(y_Expr_test, logreg_expr_y_predicted)



logestic_confusion_matrix_expr = metrics.confusion_matrix(y_Expr_test, logreg_expr_y_predicted)
#X_train, X_test, y_train, y_test

logreg_model = LogisticRegression()

logreg_model.fit(X_train, y_train)



# Predicting for test set

logreg_y_predicted = logreg_model.predict(X_test)

logreg_score = logreg_model.score(X_test, y_test)

logreg_accuracy = accuracy_score(y_test, logreg_y_predicted)

logestic_confusion_matrix = metrics.confusion_matrix(y_test, logreg_y_predicted)
# Accuracy

print('Logistic Regression Model Accuracy Score W/O Experience  : %f'  % logreg_accuracy)

print('Logistic Regression Model Accuracy Score With Experience : %f'  % logreg_expr_accuracy)



# Confusion Matrix

print('\nLogistic Regression Confusion Matrix W/O Experience: \n', logestic_confusion_matrix)

print('\nTrue Possitive    = ', logestic_confusion_matrix[1][1])

print('True Negative     = ',   logestic_confusion_matrix[0][0])

print('False Possive     = ',   logestic_confusion_matrix[0][1])

print('False Negative    = ',   logestic_confusion_matrix[1][0])

print('\nLogistic Regression Confusion Matrix With Experience: \n', logestic_confusion_matrix_expr)

print('\nTrue Possitive    = ', logestic_confusion_matrix_expr[1][1])

print('True Negative     = ',   logestic_confusion_matrix_expr[0][0])

print('False Possive     = ',   logestic_confusion_matrix_expr[0][1])

print('False Negative    = ',   logestic_confusion_matrix_expr[1][0])

#X_Expr_train, X_Expr_test, y_Expr_train, y_Expr_test

X_train_scaled = preprocessing.scale(X_Expr_train)

X_test_scaled = preprocessing.scale(X_Expr_test)
scaled_logreg_model = LogisticRegression()

scaled_logreg_model.fit(X_train_scaled, y_Expr_train)



# Predicting for test set

scaled_logreg_y_predicted = scaled_logreg_model.predict(X_test_scaled)

scaled_logreg_model_score = scaled_logreg_model.score(X_test_scaled, y_Expr_test)

scaled_logreg_accuracy = accuracy_score(y_Expr_test, scaled_logreg_y_predicted)



scaled_logreg_confusion_matrix = metrics.confusion_matrix(y_Expr_test, scaled_logreg_y_predicted)



print('----------------------Final Analysis of Logistic Regression----------------------------\n')

print('After Scalling Logistic Regression Model Accuracy Score with Experience: %f'  % scaled_logreg_accuracy)

print('\nAfter Scalling Logistic Regression Confusion Matrix With Experience: \n', scaled_logreg_confusion_matrix)

print('\nTrue Possitive    = ', scaled_logreg_confusion_matrix[1][1])

print('True Negative     = ',   scaled_logreg_confusion_matrix[0][0])

print('False Possive     = ',   scaled_logreg_confusion_matrix[0][1])

print('False Negative    = ',   scaled_logreg_confusion_matrix[1][0])

print('\nK-NN classification Report : \n',metrics.classification_report(y_Expr_test, scaled_logreg_y_predicted))

conf_table = scaled_logreg_confusion_matrix

a = (conf_table[0,0] + conf_table[1,1]) / (conf_table[0,0] + conf_table[0,1] + conf_table[1,0] + conf_table[1,1])

p = conf_table[1,1] / (conf_table[1,1] + conf_table[0,1])

r = conf_table[1,1] / (conf_table[1,1] + conf_table[1,0])

f = (2 * p * r) / (p + r)

print("Accuracy of accepting Loan  : ",round(a,2))

print("precision of accepting Loan : ",round(p,2))

print("recall of accepting Loan    : ",round(r,2))

print("F1 score of accepting Loan  : ",round(f,2))
#Creating number list from range 1 to 20 of K for KNN



numberList = list(range(1,20))

neighbors = list(filter(lambda x: x % 2 != 0 , numberList)) #subsetting just the odd ones



#Declearing a empty list that will hold the accuracy scores

ac_scores = []

#performing accuracy metrics for value from 1,3,5....19

for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    #predict the response

    knn.fit(X_train, y_train.values.ravel())               

    y_pred = knn.predict(X_test)

    #evaluate accuracy

    scores = accuracy_score(y_test, y_pred)

    #insert scores to the list

    ac_scores.append(scores)                



MSE = [1 - x for x in ac_scores] # changing to misclassification error





# determining best k

optimal_k = neighbors[MSE.index(min(MSE))]



print('Odd Neighbors : \n', neighbors)

print('\nAccuracy Score : \n', ac_scores)

print('\nMisclassification error :\n', MSE)

print("\nThe optimal number of neighbor is k=",optimal_k)



# plot misclassification error vs k

plt.plot(neighbors, MSE)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()
# instantiating learning model (optimal_k = 3)

knn_model = KNeighborsClassifier(n_neighbors=optimal_k , weights = 'uniform', metric='euclidean')

knn_model.fit(X_train, y_train)

knn_y_predicted = knn_model.predict(X_test)

knn_score = knn_model.score(X_test, y_test)

knn_accuracy = accuracy_score(y_test, knn_y_predicted)

knn_confusion_matrix = metrics.confusion_matrix(y_test, knn_y_predicted)
# instantiating learning model (optimal_k = 3)

knn_model_expr = KNeighborsClassifier(n_neighbors=optimal_k , weights = 'uniform', metric='euclidean')

knn_model_expr.fit(X_Expr_train, y_Expr_train)

knn_expr_y_predicted = knn_model_expr.predict(X_Expr_test)

knn_expr_score = knn_model_expr.score(X_Expr_test, y_Expr_test)

knn_expr_accuracy = accuracy_score(y_Expr_test, knn_expr_y_predicted)

knn_confusion_matrix_expr = metrics.confusion_matrix(y_Expr_test, knn_expr_y_predicted)
# Comparison 

print('K-NN Model Accuracy Score W/O Experience  : %f'  % knn_accuracy)

print('K-NN Model Accuracy Score With Experience : %f'  % knn_expr_accuracy)



# Confusion Matrix

print('\nK-NN Confusion Matrix W/O Experience: \n', knn_confusion_matrix)

print('\nTrue Possitive    = ', knn_confusion_matrix[1][1])

print('True Negative     = ',   knn_confusion_matrix[0][0])

print('False Possive     = ',   knn_confusion_matrix[0][1])

print('False Negative    = ',   knn_confusion_matrix[1][0])

print('\nK-NN Confusion Matrix With Experience: \n', knn_confusion_matrix_expr)

print('\nTrue Possitive    = ', knn_confusion_matrix_expr[1][1])

print('True Negative     = ',   knn_confusion_matrix_expr[0][0])

print('False Possive     = ',   knn_confusion_matrix_expr[0][1])

print('False Negative    = ',   knn_confusion_matrix_expr[1][0])

#X_train, X_test, y_train, y_test

X_train_scaled = preprocessing.scale(X_train)

X_test_scaled = preprocessing.scale(X_test)
scaled_knn_model = KNeighborsClassifier(n_neighbors=optimal_k , weights = 'uniform', metric='euclidean')

scaled_knn_model.fit(X_train_scaled, y_train)

scaled_knn_y_predict = scaled_knn_model.predict(X_test_scaled)

scaled_knn_score = scaled_knn_model.score(X_test_scaled, y_test)

scaled_knn_accuracy = accuracy_score(y_test, scaled_knn_y_predict)

scaled_knn_confusion_matrix = metrics.confusion_matrix(y_test, scaled_knn_y_predict)


print('----------------------Final Analysis of K-NN----------------------------\n')

print('After Scalling K-NN Model Accuracy Score without Experience: %f'  % scaled_knn_accuracy)

print('\nAfter Scalling K-NN Confusion Matrix Without Experience: \n', scaled_knn_confusion_matrix)

print('\nTrue Possitive    = ', scaled_knn_confusion_matrix[1][1])

print('True Negative     = ',   scaled_knn_confusion_matrix[0][0])

print('False Possive     = ',   scaled_knn_confusion_matrix[0][1])

print('False Negative    = ',   scaled_knn_confusion_matrix[1][0])

print('\nK-NN classification Report : \n',metrics.classification_report(y_test, scaled_knn_y_predict))

knn_conf_table = scaled_knn_confusion_matrix

a = (knn_conf_table[0,0] + knn_conf_table[1,1]) / (knn_conf_table[0,0] + knn_conf_table[0,1] + knn_conf_table[1,0] + knn_conf_table[1,1])

p = knn_conf_table[1,1] / (knn_conf_table[1,1] + knn_conf_table[0,1])

r = knn_conf_table[1,1] / (knn_conf_table[1,1] + knn_conf_table[1,0])

f = (2 * p * r) / (p + r)

print("\nAccuracy of accepting Loan  : ",round(a,2))

print("precision of accepting Loan : ",round(p,2))

print("recall of accepting Loan    : ",round(r,2))

print("F1 score of accepting Loan  : ",round(f,2))
gnb_model = GaussianNB()

gnb_model.fit(X_train, y_train)

gnb_y_predicted = gnb_model.predict(X_test)

gnb_score = gnb_model.score(X_test, y_test)

gnb_accuracy = accuracy_score(y_test, gnb_y_predicted)

gnb_confusion_matrix = metrics.confusion_matrix(y_test, gnb_y_predicted)
gnb_expr_model = GaussianNB()

gnb_expr_model.fit(X_Expr_train, y_Expr_train)

gnb_expr_y_predicted = gnb_expr_model.predict(X_Expr_test)

gnb_expr_score = gnb_expr_model.score(X_Expr_test, y_Expr_test)

gnb_expr_accuracy = accuracy_score(y_Expr_test, gnb_expr_y_predicted)

gnb_expr_confusion_matrix = metrics.confusion_matrix(y_Expr_test, gnb_expr_y_predicted)
# Comparison 

print('Naïve Bayes Model Accuracy Score W/O Experience  : %f'  % gnb_accuracy)

print('Naïve Bayes Model Accuracy Score With Experience : %f'  % gnb_expr_accuracy)



# Confusion Matrix

print('\nNaïve Bayes Confusion Matrix W/O Experience: \n', gnb_confusion_matrix)

print('\nTrue Possitive    = ', gnb_confusion_matrix[1][1])

print('True Negative     = ',   gnb_confusion_matrix[0][0])

print('False Possive     = ',   gnb_confusion_matrix[0][1])

print('False Negative    = ',   gnb_confusion_matrix[1][0])

print('\nNaïve Bayes Confusion Matrix With Experience: \n', gnb_expr_confusion_matrix)

print('\nTrue Possitive    = ', gnb_expr_confusion_matrix[1][1])

print('True Negative     = ',   gnb_expr_confusion_matrix[0][0])

print('False Possive     = ',   gnb_expr_confusion_matrix[0][1])

print('False Negative    = ',   gnb_expr_confusion_matrix[1][0])



scaled_gnb_model = GaussianNB()

scaled_gnb_model.fit(X_train_scaled, y_train)

scaled_gnb_y_predict = scaled_gnb_model.predict(X_test_scaled)

scaled_gnb_score = scaled_gnb_model.score(X_test_scaled, y_test)

scaled_gnb_accuracy = accuracy_score(y_test, scaled_gnb_y_predict)

scaled_gnb_connfusion_matrix = metrics.confusion_matrix(y_test, scaled_gnb_y_predict)


print('----------------------Final Analysis of Naïve Bayes----------------------------\n')

print('After Scalling Naïve Bayes Model Accuracy Score: %f'  % scaled_gnb_accuracy)

print('\nAfter Scalling Naïve Bayes Confusion Matrix: \n', scaled_gnb_connfusion_matrix)

print('\nTrue Possitive    = ', scaled_gnb_connfusion_matrix[1][1])

print('True Negative     = ',   scaled_gnb_connfusion_matrix[0][0])

print('False Possive     = ',   scaled_gnb_connfusion_matrix[0][1])

print('False Negative    = ',   scaled_gnb_connfusion_matrix[1][0])

print('\n Gaussian Naive Bayes classification Report : \n',metrics.classification_report(y_test, gnb_y_predicted))

gnb_conf_table = scaled_gnb_connfusion_matrix

a = (gnb_conf_table[0,0] + gnb_conf_table[1,1]) / (gnb_conf_table[0,0] + gnb_conf_table[0,1] + gnb_conf_table[1,0] + knn_conf_table[1,1])

p = gnb_conf_table[1,1] / (gnb_conf_table[1,1] + gnb_conf_table[0,1])

r = gnb_conf_table[1,1] / (gnb_conf_table[1,1] + gnb_conf_table[1,0])

f = (2 * p * r) / (p + r)

print("\nAccuracy of accepting Loan   : ",round(a,2))

print("precision of accepting Loan  : ",round(p,2))

print("recall of accepting Loan     : ",round(r,2))

print("F1 score of accepting Loan   : ",round(f,2))
print('Overall Model Accuracy After scaling:\n')

print ('Logistic Regression : {0:.0f}%'. format(scaled_logreg_accuracy * 100))

print ('K-Nearest Neighbors : {0:.0f}%'. format(scaled_knn_accuracy * 100))

print ('Naive Bayes         : {0:.0f}%'. format(scaled_gnb_accuracy * 100))



print('\nOverall Model Confusion matrix After scaling:\n')

print('\nLogistic Regression: \n', scaled_logreg_confusion_matrix)

print('\n     True Possitive    = ', scaled_logreg_confusion_matrix[1][1])

print('     True Negative     = ',   scaled_logreg_confusion_matrix[0][0])

print('     False Possive     = ',   scaled_logreg_confusion_matrix[0][1])

print('     False Negative    = ',   scaled_logreg_confusion_matrix[1][0])



print('\nK-Nearest Neighbors: \n', scaled_knn_confusion_matrix)

print('\n    True Possitive    = ', scaled_knn_confusion_matrix[1][1])

print('    True Negative     = ',   scaled_knn_confusion_matrix[0][0])

print('    False Possive     = ',   scaled_knn_confusion_matrix[0][1])

print('    False Negative    = ',   scaled_knn_confusion_matrix[1][0])



print('\nNaive Bayes: \n', scaled_gnb_connfusion_matrix)

print('\n    True Possitive    = ', scaled_gnb_connfusion_matrix[1][1])

print('    True Negative     = ',   scaled_gnb_connfusion_matrix[0][0])

print('    False Possive     = ',   scaled_gnb_connfusion_matrix[0][1])

print('    False Negative    = ',   scaled_gnb_connfusion_matrix[1][0])





print('\n\nReceiver Operating Characteristic (ROC) curve to evalute the classifier output quality.  If area of curve is closer to 1 which means better the model and if area of curve is closer to 0 which means poor the model.')



knn_fpr, knn_tpr, knn_threshold = metrics.roc_curve(y_test, scaled_knn_y_predict)

knn_roc_auc = metrics.roc_auc_score(y_test, scaled_knn_y_predict)

fig1_graph = plt.figure(figsize=(15,4))

fig1_graph.add_subplot(1,3,1)

plt.plot(knn_fpr, knn_tpr, label='KNN Model (area = %0.2f)' % knn_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic (ROC)')

plt.legend(loc="lower right")





logistic_fpr, logistic_tpr, logistic_threshold = metrics.roc_curve(y_Expr_test, scaled_logreg_y_predicted)

logistic_roc_auc = metrics.roc_auc_score(y_Expr_test, scaled_logreg_y_predicted)

fig1_graph.add_subplot(1,3,2)

plt.plot(logistic_fpr, logistic_tpr, label='Logistic Model (area = %0.2f)' % logistic_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic (ROC)')

plt.legend(loc="lower right")



nb_fpr, nb_tpr, nb_threshold = metrics.roc_curve(y_test, scaled_gnb_y_predict)

nb_roc_auc = metrics.roc_auc_score(y_test, scaled_gnb_y_predict)

fig1_graph.add_subplot(1,3,3)

plt.plot(nb_fpr, nb_tpr, label='Naive-Bayes Model (area = %0.2f)' % nb_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic (ROC)')

plt.legend(loc="lower right")

plt.show()
