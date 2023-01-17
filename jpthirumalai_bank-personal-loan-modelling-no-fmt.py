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
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn.metrics as met

import sklearn.model_selection as model

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

import numpy as np
bank_df = pd.read_csv('/kaggle/input/bank-loan/Bank_Personal_Loan_Modelling.csv',error_bad_lines=False)
bank_df.columns
bank_df.shape
bank_df.dtypes
bank_df.info()
bank_df.apply(lambda val: sum(val.isnull()))
bank_df.describe().T
sns.pairplot(data=bank_df)
bank_df.corr()
import pylab as pl

pl.figure(figsize = (16,10))



sns.heatmap(bank_df.corr(), annot=True, fmt='0.2f')
bank_df.head(10)
bank_df[["Experience","Income","CCAvg", "Mortgage"]].apply(lambda x: pd.Series([(x < 0).sum(), (x > 0).sum(), (x==0).sum()]))

# 0 - count of negative values

# 1 - count of positive values

# 1 - count of zero values
sns.boxplot(x=bank_df.Family,y=bank_df.Income,hue=bank_df["Personal Loan"])
bank_df["Personal Loan"].unique()
bank_df.groupby(["Personal Loan"]).count()
bank_df = bank_df.drop(["ID"],axis=1)
#splitting training and testing data

X_train, X_test = train_test_split(bank_df , test_size=0.3, random_state=1)

y_train = X_train.pop("Personal Loan")

y_test = X_test.pop("Personal Loan")

modelcrorecard={} #defining a dictionary to comparing different model
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
def calculate_accuracy(confusion_matrix):

    accuracy = (confusion_matrix[0][0]+confusion_matrix[1][1])/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1])

    return accuracy
def get_predictions( X_testLR1, logitmodel ):

    y_pred_df = pd.DataFrame( { 'actual': y_test,

                               "predicted_prob": logitmodel.predict( sm.add_constant( X_testLR1 ) ) } )

    return y_pred_df
import statsmodels.api as sm



def do_statmodelLR(X_trainLR1,X_testLR1,y_trainLR1,y_testLR1):

    logit = sm.Logit( y_trainLR1, sm.add_constant( X_trainLR1 ) )

    logitmodel = logit.fit()

    y_pred_bank_df = get_predictions(X_testLR1, logitmodel )

    y_pred_bank_df["original"] = np.array(y_test)

    y_pred_bank_df['predicted'] = y_pred_bank_df.predicted_prob.map( lambda x: 1 if x > 0.6 else 0)

    return y_pred_bank_df,logitmodel
pred_y_df,lgmetric = do_statmodelLR(X_train,X_test,y_train,y_test)

con_matrix = confusion_matrix(pred_y_df.original,pred_y_df.predicted)
from sklearn import metrics

def draw_conmmat( actual, predicted, modelname): #creating this as function as I am planning to reuse

    cm = metrics.confusion_matrix( actual, predicted, [1,0] )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Buy Personal Loan", "No Buy Personal Loan"] , yticklabels = ["Buy Personal Loan", "No Buy Personal Loan"] )

    plt.ylabel('True label')

    plt.xlabel('Predicted label - '+str(modelname))

    plt.show()
draw_conmmat( pred_y_df.original, pred_y_df.predicted, "Logistic Regression using StatsModels")
acc11 = calculate_accuracy(con_matrix)

modelcrorecard["LRstatmodel"]=acc11

print(acc11)
bank_df.groupby(["Personal Loan"]).count()
#Total accuracy  (TP+TN)/(TP+FP+TN+FN)

accuracy = (80+1343)/(80+69+1343+8)

print("Accuracy :"+ str(accuracy))
from sklearn.linear_model import LogisticRegression



def do_LR(X_trainLR,y_trainLR,X_testLR,y_testLR):

    clf = LogisticRegression(fit_intercept = False, C = 1e9) 

    #scikit-learn's logistic regression performs regularization by default. To negate setting high value for C

    #did this and verified to verify whether scikit learn can do same as statsmodel

    clf.fit(X_trainLR,y_trainLR) #Model

    y_pred = clf.predict(X_testLR) #prediction

    scikit_score = clf.score(X_testLR,y_testLR)

    return scikit_score,y_pred
LRScore, LRpred_labels = do_LR(X_train,y_train,X_test,y_test)

print("Score scikit learn Logistic Regression : "+str(LRScore))

modelcrorecard["LRscikitlearn"]=LRScore
confusion_matrix(y_test,LRpred_labels)
draw_conmmat(y_test,LRpred_labels, "Logistic Regression using scikit learn")
from sklearn.neighbors import KNeighborsClassifier



def do_KNN(X_train,y_train,y_test, K_value): #defining this as function as I have to reuse

    NNH = KNeighborsClassifier(n_neighbors= K_value , weights = 'uniform', metric='euclidean' )

    NNH.fit(X_train, y_train)

    # For every test data point, predict it's label based on 5 nearest neighbours in this model. The majority class will 

    # be assigned to the test data point

    KNNpredict = NNH.predict(X_test)

    score = NNH.score(X_test, y_test)

    return score,KNNpredict
for i in range(1,21): #to find optimum K

    KNNscore,KNNpredict = do_KNN(X_train,y_train,y_test,i)# applying K values from 1 to 20

    print("When K value is "+str(i)+" score = "+str(KNNscore))
# applying K value as 3

optimum_kvalue = 3

KNNscore,KNNpredict = do_KNN(X_train,y_train,y_test,optimum_kvalue)

print("When K value is "+str(optimum_kvalue)+" score = "+str(KNNscore))

print("Confusion matrix for KNN is")

conf_matrix= confusion_matrix(y_test,KNNpredict)

modelcrorecard["KNN"] = KNNscore
draw_conmmat(y_test,KNNpredict,"KNN")
calculate_accuracy(conf_matrix)
acc_knn=(1333+21)/(1333+21+128+18) #Just to verify through manual calculation

print("KNN Accuracy="+str(acc_knn))
from sklearn.naive_bayes import GaussianNB 



def do_naive_bayes(X_trainNB,y_trainNB,X_testNB,y_testNB):

    

    naive_model = GaussianNB()

    naive_model.fit(X_trainNB, y_trainNB)



    NBpredict = naive_model.predict(X_testNB)

    NBscore = naive_model.score(X_testNB,y_testNB)

    return NBscore,NBpredict
NBscore, NBpredicted_labels = do_naive_bayes(X_train,y_train,X_test,y_test)

print("Score in Naive Bayes : ",NBscore)

modelcrorecard["Naive Bayes"]=NBscore
print("Confusion matrix for Naive Bayes : ")

print(y_test.shape,"\n\n",NBpredicted_labels.shape)

#con_matrix = confusion_matrix(y_test.tolist(), NBpredicted_labels.tolist())

draw_conmmat(y_test,NBpredicted_labels,"Naive Bayes")
bank_df_corrected = bank_df.copy() #Take a copy of bank_df keeping the original data as is

bank_df_corrected[bank_df_corrected[["Experience"]] < 0] = -1 # filling negative values as NaN
# After replacing negative values with NaN, if we check no non negative values

bank_df_corrected[["Experience"]].apply(lambda x: pd.Series([(x < 0).sum(), (x > 0).sum(), (x == 0).sum()]))

# 0 - count of negative values

# 1 - count of positive values

# 1 - count of zero values
#find mean of Experience with NaN

bank_df_corrected["Experience"].mean()
#Fill negative values with mean

exp_array = bank_df_corrected[bank_df_corrected["Experience"]!= -1]["Experience"]

bank_df_corrected["Experience"]=bank_df_corrected["Experience"].replace(-1,exp_array.mean())

print(exp_array.mean())
# After replacing negative values with NaN, if we check no non negative values

bank_df_corrected[["Experience"]].apply(lambda x: pd.Series([(x < 0).sum(), (x > 0).sum(), (x == 0).sum()]))

# 0 - count of negative values

# 1 - count of positive values

# 1 - count of zero values
for key, val in modelcrorecard.items():

    print(key,"",val)
#splitting training and testing data from the corrected Dataframe bank_df_corrected

X_train, X_test = train_test_split(bank_df_corrected , test_size=0.3, random_state=1)

y_train = X_train.pop("Personal Loan")

y_test = X_test.pop("Personal Loan")
modelcrorecard2={}



pred_y_df,lgmetric = do_statmodelLR(X_train,X_test,y_train,y_test)

con_matrix = confusion_matrix(pred_y_df.original,pred_y_df.predicted)

acc1= calculate_accuracy(con_matrix)

modelcrorecard2["Logistic Regression-statmodel"]=acc1

print(acc1)
LRScore, LRpred_labels = do_LR(X_train,y_train,X_test,y_test)

print("Score scikit learn Logistic Regression : "+str(LRScore))

con_matrix2 = confusion_matrix(y_test,LRpred_labels)

acc2 = calculate_accuracy(con_matrix2)

print(acc2)
modelcrorecard2["Logistic Regression-scikit learn"]=acc2
for i in range(1,20):

    KNNscore,KNNpredict = do_KNN(X_train,y_train,y_test,i)# applying K values from 1 to 20

    print("When K value is "+str(i)+" score = "+str(KNNscore))
optimum_kvalue2 = 3 #selection an ODD value

KNNscore,KNNpredict = do_KNN(X_train,y_train,y_test,optimum_kvalue2)

print("When K value is "+str(optimum_kvalue2)+" score = "+str(KNNscore))

con_matrix3 = confusion_matrix(y_test,KNNpredict)

acc3 = calculate_accuracy(con_matrix3)

modelcrorecard2["KNN"] = acc3

print(acc3)
NBscore, NBpredicted_labels = do_naive_bayes(X_train,y_train,X_test,y_test)

print("Score in Naive Bayes : ",NBscore)

con_matrix4 = confusion_matrix(y_test,NBpredicted_labels)

acc4 = calculate_accuracy(con_matrix4)

modelcrorecard2["Naive Bayes"]=acc4

print(acc4)
draw_conmmat(y_test,pred_y_df.predicted,"Logistic Regression using StatModel")

draw_conmmat(y_test,LRpred_labels,"Logistic Regression using scikit learn")

draw_conmmat(y_test,KNNpredict,"KNN")

draw_conmmat(y_test,NBpredicted_labels,"NBpredicted_labels")
for key, val in modelcrorecard2.items():

    print(key,"",val)