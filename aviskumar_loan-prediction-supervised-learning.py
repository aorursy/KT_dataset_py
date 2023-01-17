import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



from scipy.stats import chisquare,chi2_contingency



from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



from sklearn import metrics

from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model



from statsmodels.formula.api import ols   

from sklearn.linear_model import SGDClassifier



from sklearn.linear_model import LogisticRegression



from sklearn.naive_bayes import GaussianNB



from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import roc_curve,roc_auc_score



from sklearn import svm
data=pd.read_csv('../input/personal-loan/Bank_Personal_Loan_Modelling-1.xlsx')

data.head()
data.info()
data.describe()
#Analysing the columns we see that the experinece column the data is entered wrongly like -3,-2,-1 



data['Experience'].replace({-3:3,-2:2,-1:1},inplace=True)



#Replacing 0 experience

#From the data we see that the people with 0 exp is between age 24 and 30



exp_mean=data['Experience'].loc[(data['Age']>=24) & (data['Age']<=30) ].mean()

exp_std=data['Experience'].loc[(data['Age']>=24) & (data['Age']<=30) ].std()

exp_zero_count=data['Experience'].loc[data['Experience']==0].value_counts()





blank_exp=np.random.randint(exp_mean-exp_std,exp_mean+exp_std,size=exp_zero_count)



data['Experience'].loc[data['Experience']==0]=blank_exp



#Thus Experience column is preprocessed
#Checking the normality of the columns



plt.figure(figsize=(20,15))

plt.subplot(3,3,1)

sns.distplot(data['Age'])

plt.subplot(3,3,2)

sns.distplot(data['Experience'])

plt.subplot(3,3,3)

sns.distplot(data['Income'])

plt.subplot(3,3,4)

sns.distplot(data['CCAvg'])

plt.subplot(3,3,5)

sns.distplot(data['Mortgage'].loc[data['Mortgage']!=0])
#Checking the outliers of the columns using boxplot



plt.figure(figsize=(20,15))

plt.subplot(3,3,1)

sns.boxplot(x='Personal Loan',y='Age',data=data)

plt.subplot(3,3,2)

sns.boxplot(x='Personal Loan',y='Experience',data=data)

plt.subplot(3,3,3)

sns.boxplot(x='Personal Loan',y='Income',data=data)

plt.subplot(3,3,4)

sns.boxplot(x='Personal Loan',y='CCAvg',data=data)

plt.subplot(3,3,5)

sns.boxplot(x='Personal Loan',y=data['Mortgage'].loc[data['Mortgage']!=0],data=data)
sns.pairplot(data)
linear_corr=data.corr()

fig, ax = plt.subplots(figsize=(15,10)) 

sns.heatmap(linear_corr,annot=True,ax=ax)



#From figure,we see that the Experience and Age are linearily dependent, So we can drop one of them
fig, ax = plt.subplots(figsize=(20,10)) 

sns.countplot(x='ZIP Code',hue='Personal Loan',data=data,ax=ax)
pd.crosstab(data['ZIP Code'],data['Personal Loan'])
sns.jointplot(x='ID',y='Personal Loan',data=data)
#From the above 3 cells, we can see thet the columns ID,ZIP Code can be dropped,Since they are not providing necessary info about the Personal Loan

#Since the Age and Experience are also Linearily dependent we can drop one of them. We can drop Experience column



drop_cols=['Experience','ID','ZIP Code']

data.drop(columns=drop_cols,inplace=True)

data.head()
print(data[['Personal Loan','Family']].groupby(['Family']).mean())

print(data[['Personal Loan','Education']].groupby(['Education']).mean())

print(data[['Personal Loan','Securities Account']].groupby(['Securities Account']).mean())
#From above cell,we see that the Family count has some dependency on the Personal Loan, Its better we can 



plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

sns.countplot(x='Family',hue='Personal Loan',data=data)

plt.subplot(1,2,2)

sns.countplot(x='Education',hue='Personal Loan',data=data)



#We see that if the family size is greater than 2, it makes people to apply loan

#So we can make a category column, Big_family - 0 means less than or equal to 2 ; 1 means greater than 2

data['Big_family']=data['Family'].replace({1:0,2:0,3:1,4:1})



#We also see that the people with more than 1 degree has same characteristics, so we can group em together,

data['Is_Educated']=data['Education'].replace({1:0,2:1,3:1})





#We see that most of people dont take Mortgage, So its better to convert wether Mortgage is taken or not,

data['IsMortgage']=data['Mortgage']

data[data['IsMortgage']!=0] = 1



#Thus we can drop Family,Education and Mortrage columns,

drop_cols=['Mortgage','Family','Education']

data.drop(columns=drop_cols,inplace=True)
plt.figure(figsize=(20,10))

plt.subplot(2,2,1)

sns.countplot(x='Securities Account',hue='Personal Loan',data=data)

plt.subplot(2,2,2)

sns.countplot(x='CD Account',hue='Personal Loan',data=data)

plt.subplot(2,2,3)

sns.countplot(x='Online',hue='Personal Loan',data=data)

plt.subplot(2,2,4)

sns.countplot(x='CreditCard',hue='Personal Loan',data=data)
#Check dependency among the categorical variables



cont1=pd.crosstab(data['Securities Account'],data['CD Account'])

print(chi2_contingency(cont1))



cont2=pd.crosstab(data['CreditCard'],data['CD Account'])

print(chi2_contingency(cont2))



cont3=pd.crosstab(data['CreditCard'],data['Online'])

print(chi2_contingency(cont3))
#Done for OLS method- formula



data.rename(index=str,columns={"Personal Loan":"Personal_Loan" , "Securities Account":"Securities_Account" , "CD Account" : "CD_Account"},inplace=True)

data.head()
#Splitting of Independent and Dependent variables



y=data['Personal_Loan']

X=data.drop(columns='Personal_Loan')
#Standardization of Data



def standardization(X_train,X_test):

    scaler=preprocessing.StandardScaler()

    X_train=scaler.fit_transform(X_train)

    X_test=scaler.transform(X_test)

    return X_train,X_test
#Linear regression method,



def linear_reg(X,y):

    

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)

    X_train,X_test=standardization(X_train,X_test)

    

    linear_reg=LinearRegression()

    linear_reg.fit(X_train,y_train)

    score=linear_reg.score(X_test,y_test)

    print("The linear model prediction is " + str(score*100) + "%")

    

    

    # make predictions

    expected = y_test

    predicted = linear_reg.predict(X_test).round()

    print("The confusion matrix is ")

    print(metrics.confusion_matrix(expected, predicted))

    

    roc=roc_auc_score(y_test, predicted)

    print("ROC value for linear model is "+ str(roc*100) + "%")

    

    

#OlS Linear Regression method



def linear_reg_ols(formula,data):

    model=ols(formula,data).fit()

    print(model.summary())

    



#Polynomial Regression model



def polynomial_reg(X,y):

    X_poly_train,X_poly_test,y_poly_train,y_poly_test=train_test_split(X,y,test_size=0.25,random_state=1)

    X_poly_train,X_poly_test=standardization(X_poly_train,X_poly_test)

    

    poly = PolynomialFeatures(degree=2, interaction_only=True)



    X1_poly_train=poly.fit_transform(X_poly_train)

    X1_poly_test=poly.fit_transform(X_poly_test)



    lin=linear_model.LinearRegression()

    lin.fit(X1_poly_train,y_poly_train)



    y_pred=lin.predict(X1_poly_test)



    poly_score=lin.score(X1_poly_test,y_poly_test)

    print("The polynomial model prediction is " + str(poly_score*100) + "%")

    

    # make predictions

    expected = y_poly_test

    predicted = lin.predict(X1_poly_test).round()

    print("The confusion matrix is ")

    print(metrics.confusion_matrix(expected, predicted))

    

    roc=roc_auc_score(expected, predicted)

    print("ROC value for linear model is "+ str(roc*100) + "%")

    

    

#Gradient Descent



def gradient_descent(X_train,y_train):

    gradient=SGDClassifier(max_iter=1000,tol=1e-3)

    gradient.fit(X_train,y_train)

    y_pred=gradient.predict(X_test)

    y_pred=y_pred.reshape(1250,1)

    grad_score=gradient.score(X_test,y_test)

    

    print(y_pred)

    print(y_test)

    print("The Gradient Descent model prediction is " + str(grad_score*100) + "%")

    

#Logistic regression



def logistic_reg(X,y):

    

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)

    X_train,X_test=standardization(X_train,X_test)

    

    logistic_reg=LogisticRegression()

    logistic_reg.fit(X_train,y_train)

    log_pred=logistic_reg.predict(X_test)

    log_score=logistic_reg.score(X_test,y_test)

    print("The Logistic model prediction is " + str(log_score*100) + "%")

    print("The confusion matrix is ")

    print(metrics.confusion_matrix(y_test, log_pred))

    print("the Classification report is")

    print(metrics.classification_report(y_test, log_pred))

    roc=roc_auc_score(y_test, log_pred)

    print("ROC value for logistic model is "+ str(roc*100) + "%")

    

    

#Naive Bayes



def naive_bayes(X,y):

    

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)

    X_train,X_test=standardization(X_train,X_test)

    

    naive_model=GaussianNB()

    naive_model.fit(X_train,y_train)

    naive_pred=naive_model.predict(X_test)

    naive_score=naive_model.score(X_test,y_test)

    print("The Naive Bayes model prediction is " + str(naive_score*100) + "%")

    print("The confusion matrix is ")

    print(metrics.confusion_matrix(y_test, naive_pred))

    print("the Classification report is")

    print(metrics.classification_report(y_test, naive_pred))

    roc=roc_auc_score(y_test, naive_pred)

    print("ROC value for linear model is "+ str(roc*100) + "%")

    

    

#KNN



def knn(X,y,n):

    

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)

    X_train,X_test=standardization(X_train,X_test)

    

    knn_model = KNeighborsClassifier(n_neighbors= n , weights = 'distance' )

    knn_model.fit(X_train, y_train)

    knn_predict=knn_model.predict(X_test)

    knn_score=knn_model.score(X_test,y_test)

    print("The KNN model prediction is " + str(knn_score*100) + "%")

    print("The confusion matrix is ")

    print(metrics.confusion_matrix(y_test,knn_predict))

    print("the Classification report is")

    print(metrics.classification_report(y_test,knn_predict))

    roc=roc_auc_score(y_test, knn_predict)

    print("ROC value for linear model is "+ str(roc*100) + "%")

    

    

#SVM



def svm_fun(X,y):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)

    X_train,X_test=standardization(X_train,X_test)

    

    clf = svm.SVC(gamma=0.025,C=3)

    #when C increases Marigin shrinks

    # gamma is a measure of influence of a data point. It is inverse of distance of influence. C is complexity of the model

    # lower C value creates simple hyper surface while higher C creates complex surface



    clf.fit(X_train,y_train)

    svm_pred=clf.predict(X_test)

    svm_score=clf.score(X_test,y_test)

    print("The KNN model prediction is " + str(svm_score*100) + "%")

    

    print("The confusion matrix is ")

    print(metrics.confusion_matrix(y_test,svm_pred))

    print("the Classification report is")

    print(metrics.classification_report(y_test,svm_pred))

    roc=roc_auc_score(y_test, svm_pred)

    print("ROC value for svm model is "+ str(roc*100) + "%")
#Linear

linear_reg(X,y)
#OLS linear

formula= ' Personal_Loan ~ Age + Income + CCAvg + Securities_Account + CD_Account + Online + CreditCard + Big_family + Is_Educated + IsMortgage '

linear_reg_ols(formula,data)
#Polynomial

polynomial_reg(X,y)
#Gradient Descent

#gradient_descent(X_train,y_train)
#SVM



svm_fun(X,y)
#Logistic Regression

logistic_reg(X,y)



#Since in the input data, we have more value for personal Loan as 0 than 1,

   #we must consider the 0 class level f1 score - Here it is 96% good that it would predict who would get the Personal loan
#Naive Bayes

naive_bayes(X,y)



#Since in the input data, we have more value for personal Loan as 0 than 1,

   #we must consider the 0 class level f1 score - Here it is 91% good that it would predict who would get the Personal loan
#KNN

knn(X,y,3)



#Since in the input data, we have more value for personal Loan as 0 than 1,

   #we must consider the 0 class level f1 score - Here it is 96% good that it would predict who would get the Personal loan