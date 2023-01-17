#importing the libraries that we use

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

sns.set(color_codes=True) # adds a nice background to the graphs

%matplotlib inline
cust_df = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx','Data')
cust_df.head(10).style.background_gradient(cmap="RdYlBu")
cust_df.columns
cust_df.info()  #Shape of the DataSet
cust_df[["Age","Experience","Income","CCAvg","Mortgage"]] = cust_df[["Age","Experience","Income","CCAvg","Mortgage"]].astype(float)
cust_df.info()
cust_df.describe().T
cust_cat = cust_df.loc[:,["Family","Education","Personal Loan","Securities Account","CD Account","Online","CreditCard"]]

cust_num = cust_df.loc[:,["ID","Age","Experience","Income","ZIP Code","CCAvg","Mortgage"]]
cust_cat.head()
cust_num.head()
cust_num["Mortgage"].value_counts()
cust_df.skew() #skewness of the data
Target = cust_df["Personal Loan"]
#Plots to see the distribution of the Categorical features individually



plt.figure(figsize= (20,15))

plt.subplot(3,3,1)

sns.countplot(cust_df["Family"], color='lightblue')

plt.xlabel('Family')



plt.subplot(3,3,2)

sns.countplot(cust_df["Education"], color='lightgreen')

plt.xlabel('Education')



plt.subplot(3,3,3)

sns.countplot(cust_df["Securities Account"], color='pink')

plt.xlabel('Securities Account')



plt.subplot(3,3,4)

sns.countplot(cust_df["CD Account"], color='gray')

plt.xlabel('CD Account')



plt.subplot(3,3,5)

sns.countplot(cust_df["Online"], color='cyan')

plt.xlabel('Online')



plt.subplot(3,3,6)

sns.countplot(cust_df["CreditCard"], color='Aquamarine')

plt.xlabel('CreditCard')



plt.show()
#Plots to see the distribution of the Continuos features individually

plt.figure(figsize= (20,15))

plt.subplot(2,2,1)

plt.hist(cust_df["Age"], color='lightblue', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Age')



plt.subplot(2,2,2)

plt.hist(cust_df["Experience"], color='lightgreen', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Experience')



plt.subplot(2,2,3)

plt.hist(cust_df["Income"], color='orange', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Income')



plt.subplot(2,2,4)

plt.hist(cust_df["Mortgage"], color='Pink', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Mortgage')



plt.figure(figsize= (20,15))

plt.subplot(4,1,1)

sns.boxplot(x= cust_df["Age"], color='lightblue')



plt.subplot(4,1,2)

sns.boxplot(x= cust_df["Experience"], color='lightblue')



plt.subplot(4,1,3)

sns.boxplot(x= cust_df["Income"], color='lightblue')



plt.subplot(4,1,4)

sns.boxplot(x= cust_df["Mortgage"], color='lightblue')



plt.show()
#Checking the pair plot between each feature

sns.pairplot(cust_df)  #pairplot

plt.show()
#Analysis between Target Variable and Other categorical variables.

plt.figure(figsize= (20,15))

plt.subplot(3,3,1)

sns.countplot(cust_df["Education"],palette="Greens",hue=cust_df["Personal Loan"])



plt.subplot(3,3,2)

sns.countplot(cust_df["Online"],palette="Greens",hue=cust_df["Personal Loan"])



plt.subplot(3,3,3)

sns.countplot(cust_df["Family"],palette="Greens",hue=cust_df["Personal Loan"])



plt.subplot(3,3,4)

sns.countplot(cust_df["CreditCard"],palette="Greens",hue=cust_df["Personal Loan"])



plt.subplot(3,3,5)

sns.countplot(cust_df["Securities Account"],palette="Greens",hue=cust_df["Personal Loan"])



plt.subplot(3,3,6)

sns.distplot(cust_df[cust_df["Personal Loan"]==1]["Mortgage"],kde=True,hist=False,color='red',label="Mortgage customer with PL")

sns.distplot(cust_df[cust_df["Personal Loan"]==0]["Mortgage"],hist=False,kde=True,color='green',label="Mortgage customer without PL")

plt.legend()

plt.show();
cust_df.corr()
#Correlation

plt.figure(figsize= (15,10))

sns.heatmap(cust_df.corr())

plt.show()
#count plot of 

sns.countplot(Target, palette='hls')

plt.show()
n_true = len(cust_df.loc[cust_df["Personal Loan"] == 1])

n_false = len(cust_df.loc[cust_df["Personal Loan"] == 0])

print("Number of customers who bought personal loan: {0} ({1:2.2f}%)".format(n_true, (n_true / (n_true + n_false)) * 100 ))

print("Number of customers who didn't bought personal loan: {0} ({1:2.2f}%)".format(n_false, (n_false / (n_true + n_false)) * 100))
cust_df.info()
avg_Exp = cust_df["Experience"].mean()

print(f"Average Experience {avg_Exp}")

cust_df["Experience"] = cust_df["Experience"].apply(lambda x : avg_Exp if x<0 else x)
Q1 = cust_df["Income"].quantile(0.25)

Q3 = cust_df["Income"].quantile(0.75)

IQR = Q3 - Q1

whisker = Q1 + 1.5 * IQR

cust_inc = cust_df["Income"].apply(lambda x : whisker if x>whisker else x)
Q1 = cust_df["Mortgage"].quantile(0.25)

Q3 = cust_df["Mortgage"].quantile(0.75)

IQR = Q3 - Q1

whisker = Q1 + 1.5 * IQR

cust_mor = cust_df["Mortgage"].apply(lambda x : whisker if x>whisker else x)
cust_df["Income"]=cust_inc

cust_df["Mortgage"]=cust_mor
cust_df.head().style.background_gradient(cmap="RdYlBu")
cust_df["Mortgage"]= np.log1p(cust_df["Mortgage"])

sns.distplot(cust_df["Mortgage"])

plt.show()
#Standardise the numerical columns

scalar = StandardScaler()

cust_df["Experience"]=scalar.fit_transform(cust_df[["Experience"]])

cust_df["Income"]=scalar.fit_transform(cust_df[["Income"]])

cust_df["CCAvg"]=scalar.fit_transform(cust_df[["CCAvg"]])

cust_df["Mortgage"]=scalar.fit_transform(cust_df[["Mortgage"]])
X = cust_df.drop(['Personal Loan'],axis=1)     # Predictor feature columns 

Y = Target   # Predicted class (1, 0) 



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# 1 is just any random seed number



x_train.head()
print("{0:0.2f}% data is in training set".format((len(x_train)/len(cust_df.index)) * 100))

print("{0:0.2f}% data is in test set".format((len(x_test)/len(Target.index)) * 100))
print("Original Personal Loan Values of customer who bought : {0} ({1:2.2f}%)".format(n_true, (n_true / (n_true + n_false)) * 100 ))

print("Original Personal Loan Values of customer who didn't buy  : {0} ({1:2.2f}%)".format(n_false, (n_false / (n_true + n_false)) * 100))

print("")

print("Training Personal Loan Values of customer who bought    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))

print("Training Personal Loan Values of customer who didn't buy   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))

print("")

print("Test Personal Loan Values of customer who bought        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))

print("Test Personal Loan Values of customer who didn't buy       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))

print("")
def logistReg(x_train,y_train,solver="liblinear"):

    # Fit the model on train

    model = LogisticRegression(solver=solver)

    model.fit(x_train, y_train)

    #predict on test

    y_predict = model.predict(x_test)

    y_predictprob = model.predict_proba(x_test)



    coef_df = pd.DataFrame(model.coef_,columns=list(x_train.columns))

    coef_df['intercept'] = model.intercept_

    model_score = model.score(x_train, y_train)

    print(f"Accuracy of Training Data: {model_score}")

    model_score = model.score(x_test, y_test)

    print(f"Accuracy of Test Data: {model_score}")

    print(coef_df)

    print(metrics.classification_report(y_test,y_predict))

    cm=metrics.confusion_matrix(y_test, y_predict, labels=[1, 0])



    df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                      columns = [i for i in ["Predict 1","Predict 0"]])

    plt.figure(figsize = (8,5))

    sns.heatmap(df_cm, annot=True)

    plt.show()

    print("f1 score", metrics.f1_score(y_test,y_predict))

    print("Auc Roc Score: ",metrics.roc_auc_score(y_test,y_predict))

    return y_predictprob,y_predict
y_predProb,y_pred = logistReg(x_train,y_train)
X = cust_df.drop(["Personal Loan","Age","ZIP Code","CreditCard","ID","Online"],axis=1)     # Predictor feature columns 

Y = Target   # Predicted class (1, 0) 



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=46)

# 1 is just any random seed number



x_train.head()
y_predpob,y_pred=logistReg(x_train,y_train)
sns.countplot(y_train)

plt.show()
y_train.value_counts()
#Undersampling the majority

xtrain_resampled, ytrain_resampled = RandomUnderSampler(sampling_strategy=0.2,random_state=46).fit_resample(x_train,y_train)

sns.countplot(ytrain_resampled)

plt.show()

y_predProb,y_predict = logistReg(xtrain_resampled,ytrain_resampled)
fprLR, tprLR, threshLR = metrics.roc_curve(y_test, y_predProb[:,1], pos_label=1)
scores =[]

#OverSampling the minority to get the better results

xtrain_resampled, ytrain_resampled = SMOTE(sampling_strategy=1,random_state=46).fit_resample(x_train,y_train)

for k in range(1,50):

    NNH = KNeighborsClassifier(n_neighbors = k, weights = 'distance', metric='euclidean' )

    NNH.fit(xtrain_resampled, ytrain_resampled)

    scores.append(NNH.score(x_test, y_test))
plt.plot(range(1,50),scores)

plt.show()
NNH = KNeighborsClassifier(n_neighbors= 5 , weights = 'distance', metric='euclidean' )

NNH.fit(xtrain_resampled, ytrain_resampled)

y_predKnn = NNH.predict(x_test)
print(metrics.classification_report(y_test,y_predKnn))

cm=metrics.confusion_matrix(y_test, y_predKnn, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (8,5))

sns.heatmap(df_cm, annot=True)

plt.show()

print(f'Score of Knn Test Data : {NNH.score(x_test,y_test)}')

print(f'Score of Knn Train Data : {NNH.score(xtrain_resampled,ytrain_resampled)}')

print(f"Roc AUC score of KNN : {metrics.roc_auc_score(y_test,y_predKnn)}")

print(f"f1 score of KNN : {metrics.f1_score(y_test,y_predKnn)}\n")
pred_prob_NNH = NNH.predict_proba(x_test)

fprNNH, tprNNH, threshNNH = metrics.roc_curve(y_test, pred_prob_NNH[:,1], pos_label=1)
NBmodel = GaussianNB()

NBmodel.fit(xtrain_resampled,ytrain_resampled)

y_NBPred = NBmodel.predict(x_test)
print(metrics.classification_report(y_test,y_NBPred))

cm=metrics.confusion_matrix(y_test, y_NBPred, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (8,5))

sns.heatmap(df_cm, annot=True)

plt.show()

print(f'Score of NB Test Data : {NBmodel.score(x_test,y_test)}')

print(f'Score of NB Train Data : {NBmodel.score(xtrain_resampled,ytrain_resampled)}')

print(f"Roc AUC score of NB : {metrics.roc_auc_score(y_test,y_NBPred)}")

print(f"f1 score of NB : {metrics.f1_score(y_test,y_NBPred)}\n")
pred_prob_NB = NBmodel.predict_proba(x_test)

fprNB, tprNB, threshNB = metrics.roc_curve(y_test, pred_prob_NB[:,1], pos_label=1)
# plot roc curves

plt.figure(figsize=(15,10))

plt.plot(fprLR, tprLR, linestyle='--',color='orange', label='Logistic Regression')

plt.plot(fprNNH, tprNNH, linestyle='--',color='green', label='KNN')

plt.plot(fprNB, tprNB, linestyle='--', color='blue', label='Naive Bayes')

# title

plt.title('ROC curve')

# x label

plt.xlabel('False Positive Rate(1-True Positive Rate)')

# y label

plt.ylabel('True Positive rate')



plt.legend(loc='best')

plt.show();