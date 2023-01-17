import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
#load the csv file and make the data frame

bank_df = pd.read_csv('/kaggle/input/bank-loan/Bank_Personal_Loan_Modelling.csv')
#display the first 5 rows of data frame

bank_df.head()
print("The dataframe has {} rows and {} columns".format(bank_df.shape[0],bank_df.shape[1]))
#display information of data frame

bank_df.info()
#another way to check if null values are there or not

bank_df.apply(lambda x:sum(x.isnull()))
#5 point summary of dataframe

bank_df.describe().transpose()
#display histogram plot of each attribute/column

for i in bank_df.columns:

    plt.hist(bank_df[i])

    plt.xlabel(i)

    plt.ylabel('frequency')

    plt.show()
bank_df['Personal Loan'].value_counts()
print("Percentage of customer accept personal loan is {}%".format((480/5000)*100))

print("Percentage of customer not accept personal loan is {}%".format((4520/5000)*100))
sns.distplot(bank_df['Personal Loan'],kde=False)

plt.show()
new_bank_df = bank_df.copy()
print("the total customers whose experience is in negative is {}".format((new_bank_df[new_bank_df['Experience']<0]).shape[0]))
#converting negative experience values into positive

new_bank_df['Experience'] = new_bank_df['Experience'].apply(lambda x : abs(x) if(x<0) else x)
print("now after manipulation total customers whose experience is in negative is {}".format((new_bank_df[new_bank_df['Experience']<0]).shape[0]))
#dropping ID and ZIP Code columns from new_bank_df dataframe

new_bank_df.drop(['ID','ZIP Code'],axis=1,inplace=True)
#display first 5 rows of dataframe.

new_bank_df.head()
#display pair plot

sns.pairplot(data=new_bank_df,hue='Personal Loan')

plt.show()
X = new_bank_df.drop('Personal Loan',axis=1)

y = new_bank_df['Personal Loan']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=1)
print("The training feature are {} % of dataset and training labels are {} % of dataset".format(((X_train.shape[0]/5000)*100),((y_train.shape[0]/5000)*100)))

print("The test feature are {} % of dataset and test labels are {} % of dataset".format(((X_test.shape[0]/5000)*100),((y_test.shape[0]/5000)*100)))
#importing the library

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score,f1_score 
lr = LogisticRegression() #Instantiate the LogisticRegression object

lr.fit(X_train,y_train) #call the fit method of logistic regression to train the model or to learn the parameters of model
y_predict = lr.predict(X_test) #predicting the result of test dataset and storing in a variable called y_predict
print(accuracy_score(y_test,y_predict))#printing overall accuracy score
print("Confusion matrix")

print(confusion_matrix(y_test,y_predict))#creating confusion matrix
#displaying precision,recall and f1 score.

df_table = confusion_matrix(y_test,y_predict)

a = (df_table[0,0] + df_table[1,1]) / (df_table[0,0] + df_table[0,1] + df_table[1,0] + df_table[1,1])

p = df_table[1,1] / (df_table[1,1] + df_table[0,1])

r = df_table[1,1] / (df_table[1,1] + df_table[1,0])

f = (2 * p * r) / (p + r)



print("accuracy : ",round(a,2))

print("precision: ",round(p,2))

print("recall   : ",round(r,2))

print("F1 score : ",round(f,2))
#another way of displaying precision,recall and f1 score

print("precision:",precision_score(y_test,y_predict))

print("recall   :",recall_score(y_test,y_predict))

print("f1 score :",f1_score(y_test,y_predict))
for idx, col_name in enumerate(X_train.columns):

    print("The coefficient for {} is {}".format(col_name, lr.coef_[0][idx]))
print("The intercept is {}".format(lr.intercept_))
#importing the library

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5) #Initialize the object

knn.fit(X_train,y_train)  #call the fit method of knn classifier to train the model
knn_y_predict = knn.predict(X_test) #predicting the result of test dataset and storing in a variable called knn_y_predict
print(accuracy_score(y_test,knn_y_predict)) #printing overall accuracy score
print("Confusion matrix")

print(confusion_matrix(y_test,knn_y_predict)) #creating confusion matrix
#displaying precision,recall and f1 score

print("precision:",precision_score(y_test,knn_y_predict))

print("recall   :",recall_score(y_test,knn_y_predict))

print("f1 score :",f1_score(y_test,knn_y_predict))
#importing the library

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB() #Initialize the object

nb.fit(X_train,y_train)  #call the fit method of gaussian naive bayes to train the model or to learn the parameters of model
nb_y_predict = nb.predict(X_test)  #predicting the result of test dataset and storing in a variable called nb_y_predict
print(accuracy_score(y_test,nb_y_predict))  #printing overall accuracy score
print("Confusion matrix")

print(confusion_matrix(y_test,nb_y_predict))  #printing confusion matrix
#displaying precision,recall and f1 score

print("precision:",precision_score(y_test,nb_y_predict))

print("recall   :",recall_score(y_test,nb_y_predict))

print("f1 score :",f1_score(y_test,nb_y_predict))
#importing the library

from sklearn.svm import SVC
svc = SVC()  #Initialize the object

svc.fit(X_train,y_train)  #call the fit method of support vector machine to train the model or to learn the parameters of model
svc_y_predict = svc.predict(X_test)  #predicting the result of test dataset and storing in a variable called svc_y_predict
print(accuracy_score(y_test,svc_y_predict))  #printing overall accuracy score
print("Confusion matrix")

print(confusion_matrix(y_test,svc_y_predict))#printing confusion matrix
#displaying precision,recall and f1 score

print("precision:",precision_score(y_test,svc_y_predict))

print("recall   :",recall_score(y_test,svc_y_predict))

print("f1 score :",f1_score(y_test,svc_y_predict))
#Earlier we select k randomly as 5 now we will see which k value will give least misclassification error

# creating odd list of K for KNN

myList = list(range(1,20))



# subsetting just the odd ones

neighbors = list(filter(lambda x: x % 2 != 0, myList))
# empty list that will hold accuracy scores

ac_scores = []



# perform accuracy metrics for values from 1,3,5....19

for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    # predict the response

    y_pred_var = knn.predict(X_test)

    # evaluate accuracy

    scores = accuracy_score(y_test, y_pred_var)

    ac_scores.append(scores)



# changing to misclassification error

MSE = [1 - x for x in ac_scores]



# determining best k

optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)
# plot misclassification error vs k

plt.plot(neighbors, MSE)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()
knn_opt = KNeighborsClassifier(n_neighbors=9) #Initialize the object

knn_opt.fit(X_train,y_train)#call the fit method of knn classifier to train the model
knn_opt_y_predict = knn_opt.predict(X_test)#predicting the result of test dataset and storing in a variable called knn_opt_y_predict
print(accuracy_score(y_test,knn_opt_y_predict))#printing overall accuracy score
print("Confusion matrix")

print(confusion_matrix(y_test,knn_opt_y_predict))#creating confusion matrix
#displaying precision,recall and f1 score

print("precision:",precision_score(y_test,knn_opt_y_predict))

print("recall   :",recall_score(y_test,knn_opt_y_predict))

print("f1 score :",f1_score(y_test,knn_opt_y_predict))
lr_scores = []

thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for i in range(0,len(thresh)):

    preds = np.where(lr.predict_proba(X_test)[:,1] >=thresh[i], 1, 0)

    accurcy_scores = accuracy_score(y_test, preds)

    lr_scores.append(accurcy_scores)



df = pd.DataFrame(data={'thresh':thresh,'accuracy_scores':lr_scores})

print(df)
plt.plot(thresh,lr_scores)

plt.xlabel('Threshold')

plt.ylabel('Accuracy_scores')

plt.show()