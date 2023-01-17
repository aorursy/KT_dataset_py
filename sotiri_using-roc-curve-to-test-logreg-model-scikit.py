from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import classification_report,confusion_matrix,roc_curve,roc_auc_score

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
#Dataframe:

df = pd.read_csv("../input/heart.csv")



# X = regressors, y= response

X = df.drop(columns='target', axis=1)

y = df.target



#Binary classification: 1 - Heart disease, 0 - No Disease

df.head()
#Check to see if age is normally distributed amongst those who have heart disease:

df_heart = df[df['target']==1].age

age_arr = np.array(df_heart)

print('Number of individuals with heart disease: {}'.format(df_heart.count()))

print('Mean (disease): {0} , Standard Deviation (disease): {1}'.format(np.mean(df_heart),np.std(df_heart)))



# No Disease:

df_no_disease = df[df['target']==0].age

age_nd_arr = np.array(df_no_disease)

print('Number of individuals with no disease: {}'.format(df_no_disease.count()))

print('Mean (nd): {0} , Standard Deviation (nd): {1}'.format(np.mean(df_no_disease),np.std(df_no_disease)))
#Check Cumulative density function:

#CDF function:

def ecdf(data):

    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n

    n = len(data)



    # x-data for the ECDF: x

    x = np.sort(data)



    # y-data for the ECDF: y

    y = np.arange(1, n+1) / n



    return x, y
#Unpack ecdf for plotting:

x,y = ecdf(age_arr)

x_nd,y_nd = ecdf(age_nd_arr)



#Plot:

sns.set()

_ = plt.plot(x,y,marker='.',linestyle='none')

_ = plt.plot(x_nd,y_nd,marker='.',linestyle='none')

_ = plt.xlabel('Age (years)')

_ = plt.ylabel('CDF')

_ = plt.title('Cumulative Density Function')



_ = plt.legend(['Disease','No Disease'],loc='lower right')



#We know as sample size n = 165 >30 by central limit theorem we can approx normal
#Focus on fitting a distribution to the ages with disease:

#Normal:

mu = np.mean(df_heart)

sigma = np.std(df_heart)

sample = np.random.normal(mu,sigma,size=10000)

x_theo,y_theo = ecdf(sample)

#Plot sample against actual results to compare:

_ = plt.plot(x_theo,y_theo)

_ = plt.plot(x,y,marker='.',linestyle='none')

_ = plt.xlabel('Age (Years)')

_ = plt.ylabel('CDF')



#Fairly normal:
#Plotting the bell curve:

_ = plt.hist(sample,bins=100,density=True,histtype='step')

_ = plt.xlabel('Age')

_ = plt.ylabel('Probability')

_ = plt.ylim(-0.01, 0.42)

plt.show()
#What is the probability of getting a heart attack at age<=30?

age_30 = np.sum(sample < 30)

age_30

#77 people out of 10000 is quite small:

print('The probability of someone getting heart disease under 30 is {}'.format(age_30/len(sample)))
# Now onto the predictive model:

lr = LogisticRegression(solver='saga',max_iter=10000)



#X_1 = features, y_1 = target:

X_1 = df.drop(columns='target',axis=1)

y_1 = df.target



#Train vs Test split:

X_train,X_test,y_train,y_test = train_test_split(X_1,y_1,test_size=0.3,random_state=42)
#Setup a hyperparameter grid to train C (the hyperparameter of LogReg)

#Grid of values using np.logspace:

#logspace is usually in base 10 (powers of 10) and are evenly divided between start & stop:

c_space = np.logspace(-5,8,)



#Parameter grid:

param_grid = {'C': c_space}
#Using GridsearchCV: Find the best possible value for C (hyper parameter)

logreg_cv = GridSearchCV(lr,param_grid,cv=5)
#Fit logreg_cv onto training data to find best hyper parameter C:

logreg_cv.fit(X_train,y_train)
#Print the tuned parameters and score:

best_param_c = logreg_cv.best_params_

print(best_param_c)



#Gives you R^2 with the associated best hyperparameter C

best_score = logreg_cv.best_score_

print(best_score)
#Predict based on new hyperparameter C = 0.59636

y_pred = logreg_cv.predict(X_test)
#Now to test the robustness of model use ROC:

conf_matrix = confusion_matrix(y_test,y_pred)

conf_matrix = pd.DataFrame(conf_matrix,index=['0 True','1 True'], columns=['0 pred','1 pred'])

conf_matrix
#Now looks at the classification report: Inspecting Recall and Precision

class_report = classification_report(y_test,y_pred)

print(class_report)
#ROC curve:

#Grab probabilities of datapoint predicting positive

y_pred_prob = logreg_cv.predict_proba(X_test)[:,1]



#Unpack fpr,tpr,threshold from roc_curve:

fpr,tpr,threshold = roc_curve(y_test,y_pred_prob)
#Plot ROC:

_ = plt.plot(fpr,tpr,color='red')

_ = plt.plot([0,1],[0,1],'k--')

_ = plt.xlabel('FPR')

_ = plt.ylabel('TPR')



plt.show()

#Model does a good job of observing true positives 
#Now looking at area under the curve to get a single statistic:

area = roc_auc_score(y_test,y_pred_prob)

print('The area under the ROC is {}'.format(area))