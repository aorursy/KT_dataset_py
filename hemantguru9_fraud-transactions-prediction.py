import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
# 1.1 Call libraries

import numpy as np

import pandas as pd

import seaborn as sns

# 1.2 For OS And TIME related operations

import os

import time 

# allow plots to appear directly in the notebook

import matplotlib.pyplot as plt

%matplotlib inline
from sklearn import preprocessing 

from sklearn.preprocessing import LabelEncoder

from ipykernel import kernelapp as app
# 1.3 Class for applying multiple data transformation jobs

from sklearn.compose import ColumnTransformer

# 1.4 Scale numeric data

from sklearn.preprocessing import StandardScaler as ss

# 1.5 One hot encode data--Convert to dummy

from sklearn.preprocessing import OneHotEncoder as ohe

from sklearn import metrics

from imblearn.over_sampling import SMOTE, ADASYN

# from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split

# Modeling by Decision Tree:

from sklearn.tree import DecisionTreeClassifier as dt

# 19.1 Also create confusion matrix using pandas dataframe

import scikitplot as skplt

from sklearn.metrics import confusion_matrix, average_precision_score

from sklearn.linear_model import LogisticRegression

from xgboost.sklearn import XGBClassifier

from bayes_opt import BayesianOptimization

from sklearn.pipeline import Pipeline,make_pipeline

from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV as GCV

from sklearn.model_selection import RandomizedSearchCV as RCV

from xgboost.sklearn import XGBClassifier
# 2.1 Change ipython options to display all data columns

pd.options.display.max_columns = 300

os.chdir("../input")

os.listdir()
df_train=pd.read_csv('datasetForFinalAssignment.csv')

df_train.head()
df_train.dtypes
df_train.drop(['Column 1'],inplace = True, axis ='columns')

df_train.columns
y=df_train['class']

#y.reset_index(drop=True)

y.head(20)

X = df_train.drop(['class'],axis ='columns')

#X=df_train

X.head()
sns.countplot(x="class", data=df_train)
#Rename Columns names for this section

X.rename(columns={'N[device_id]':'numberUse'},inplace=True)

X.rename(columns={'signup_time-purchase_time':'timeInBetween'},inplace=True)

X.head()
# removing not necessary Columns

X.drop(['device_id','signup_time','purchase_time','ip_address'],inplace = True, axis ='columns')
#Define the transformation function using columnTransformer, OHE and StandardScaler

def transform(categorical_columns,numerical_columns,df):

    cat = ('categorical', ohe() , categorical_columns  )

    num = ('numeric', ss(), numerical_columns)

    col_trans = ColumnTransformer([cat, num])

    #col_trans.fit(df)

    df_trans_scaled = col_trans.fit_transform(df)

    return df_trans_scaled
#Define the columns for transformations

categorical_columns = ['source', 'browser','sex']

numerical_columns = ['timeInBetween', 'purchase_value','age','numberUse']

#Define the columns for post transformation dataframe - makes referencing and understanding easier

columns = ['source_Direct','source_SEO','source_Ads','browser_Chrome','browser_FireFox','browser_IE','browser_Safari','browser_Opera','sex_M','sex_F'] + numerical_columns

columns
X_ts =transform(categorical_columns, numerical_columns, X)
X_ts = pd.DataFrame(X_ts, index=X.index, columns=columns)

print (X_ts.dtypes,"\n",X_ts.shape, "\n", X_ts.head())
y=pd.DataFrame(y)

type(y),y.head()

#y_train.head().drop(index)
#Split Data in Test_Train Set to start Modelling

X_train, X_test, y_train, y_test,indicies_tr,indicies_test = train_test_split(X_ts,y,np.arange(X_ts.shape[0]),test_size = 0.1,random_state=42)

print("X_train: shape = ",X_train.shape,"X_test Shape = ",X_test.shape,"Y_train Sample data :",y_train.shape," Y_test sample date =",y_test.shape)
Y_bal=np.sum(y_train)/len(y_train)

Y_bal
sm = SMOTE(random_state=42)

X_balance, y_res = sm.fit_resample(X_train,np.asarray(y_train).ravel())
print("The data is", (np.sum(y_res)/len(y_res)), "imbalance. which was eralier :", Y_bal)

X_balance.shape,type(X_balance),type(y_res)
#X_balance=pd.DataFrame(X_balance)

#y_res =pd.DataFrame(y_res)

#y_test=pd.DataFrame(y_test)

type(y_res),type(y_test),type(X_balance),y_res.shape,y_test.shape,y_test[:4],y_res[:4]
#Generate the image of test dataset pre-split using indicies_test.

#This will be used to capture the unscaled values of purchase_value for computing cost of model

X_Cost = df_train.iloc[indicies_test]

X_Cost.purchase_value.head()
def modelCost(test,out,df):

    #falsePositive: Cost is $8*count

    #non-fraudulent transactions (test '0') predicted as fraudulent by model (out '1')

    falsePositiveCost = df.purchase_value[(test==0) & (out==1)].count()*8

    print("falsePositive {:.0f}".format(df.purchase_value[(test==0) & (out==1)].count()))

    print("falsePositiveCost ${:.0f}".format(falsePositiveCost))

    #falseNegative: Cost is sum of purchase_value

    #fraudulent transactions (test '1') predicted as non-fraudulent by model (out '0')

    falseNegativeCost = df.purchase_value[(test==1) & (out==0)].sum()

    print("falseNegative {:.0f}".format(df.purchase_value[(test==1) & (out==0)].count()))

    print("falseNegativeCost ${:.0f}".format(falseNegativeCost))

    totalCost = falsePositiveCost + falseNegativeCost

    print("totalCost ${:.0f}".format(totalCost))

    return totalCost
from mlxtend.plotting import plot_confusion_matrix

def confusion_matrix_1(f):

    fig, ax = plot_confusion_matrix(conf_mat=f)

    plt.title("The Confusion Matrix Graph for ")

    plt.ylabel("Actual")

    plt.xlabel("Predicted")

    plt.show()

    print("The accuracy is "+str((f[1,1]+f[0,0])/(f[0,0] + f[0,1]+f[1,0] + f[1,1])*100)+ " %.")

    print("The recall from the confusion matrix is "+ str(f[1,1]/(f[1,0] + f[1,1])*100)+ " %.")

    print("The Precision from the confusion matrix is "+ str(f[1,1]/(f[0,1] + f[1,1])*100)+ " %.")
# set a seed for reproducibility

np.random.seed(3123)
#Running basic regression first to setup all checking and evaluation functions

log_reg = LogisticRegression(random_state=42)

start = time.time()

log_reg.fit(X_balance,y_res.ravel())

end = time.time()

(end - start) #0.31 seconds
out_LR = log_reg.predict(X_test)

log_reg.score(X_test, y_test) 
f_lr  = confusion_matrix( y_test, out_LR )

f_lr

# 19.2 Flatten 'f' now

tp,fp,fn,tn = f_lr.ravel()

tp,fp,fn,tn
confusion_matrix_1(f_lr)
#Proessing y_test for calculating cost

#out_LR=np.asarray(out_LR)

y_test=np.asarray(y_test)

y_test=np.reshape(y_test,len(y_test),)

y_test.shape,out_LR.shape
# Total cost to be paid for this prediction

totalCost_lr = modelCost(y_test,out_LR,X_Cost)
ct = dt( criterion="gini",    # Alternative 'entropy'

         splitter="best",     # Alternative 'random'

         max_depth=None)

# 3.2 Train our decision tree

c_tree = ct.fit(X_balance, y_res)
# 4.2 Now make prediction

out_DT = ct.predict(X_test)

ct.score(X_test,out_DT) 
out_DT.shape,y_test.shape, type(out_DT),type(y_test)
# 4.3 Get accuracy

accuracy_DT=np.sum(out_DT == y_test)/out_DT.size

print("accuracy = ", (np.sum(out_DT == y_test)/out_DT.size))
#Calculating Confusion Matrix

f_dt  = confusion_matrix( y_test, out_DT )

f_dt

# 19.2 Flatten 'f' now

tp,fp,fn,tn = f_dt.ravel()

tp,fp,fn,tn
# 19.3 Evaluate precision/recall Parameters of Accuracy 

precision_dt = tp/(tp+fp)

recall_dt = tp/(tp + fn)
confusion_matrix_1(f_dt)
# Total cost to be paid for this prediction

totalCost_DT = modelCost(y_test,out_DT,X_Cost)
steps_xg = [('sts', ss() ),

            ('pca', PCA()),

            ('xg',  XGBClassifier(silent = False,

                                  n_jobs=2)        # Specify other parameters here

            )

            ]

pipe_xg = Pipeline(steps_xg)
parameters = {'xg__learning_rate':  [0.875], 	# can be anything from 0.02 to 0.8

              'xg__n_estimators':   [206,207],  	# can be from 100 to 300

              'xg__max_depth':      [9,10], 			# which gives me best result

              'pca__n_components' : [9]			# how many N depth i want

              }                               # Total: 2 * 2 * 2 * 2

# Create Grid Search object first with all necessary prameters

clf = GCV(pipe_xg,            # pipeline object

          parameters,         # possible parameters

          n_jobs = 5,         # USe parallel cpu threads

          cv =3 ,             # No of folds => 2 means the data will be devided in 2 parts

          verbose =2,         # Higher the value, more the verbosity

          scoring = ['accuracy', 'roc_auc'],  # Metrics for performance

          refit = 'roc_auc'   # Refitting final model on what parameters?

          )
#Start fitting data to pipeline

start = time.time()

clf.fit(X_balance, y_res.ravel())

end = time.time()

(end-start)/60
#Start Prediction

out_gs=clf.predict(X_test)
print("Test Data",y_test.shape, "Predicted Size",out_gs.shape)
#Get Precision and Recall

f_gs = confusion_matrix( y_test, out_gs )

#tp,fp,fn,tn = f_gs

tp,fp,fn,tn

precision_gs = tp/(tp+fp)

recall_gs = tp/(tp + fn)

accuracy_gs = np.sum(out_gs == y_test)/out_gs.size

print("GS Precions :",precision_gs,"GS Recall :",recall_gs,"accuracy = ", (np.sum(out_gs == y_test)/out_gs.size))
confusion_matrix_1(f_gs)

# Total cost to be paid for this prediction

totalCost_GS = modelCost(y_test,out_gs,X_Cost)
totalCost_lr,totalCost_DT,totalCost_GS
# Read the test data file

df_test=pd.read_csv("datasetForFinalTest.csv")

df_test.head()
X_test=pd.DataFrame(df_test)

#Processing Data so that data file is ready for Predicting in Model.

X_test=X_test.drop(['Column 1','device_id','signup_time','purchase_time','ip_address'],inplace = True, axis =1)

X_test=pd.DataFrame(X_test)

type(X_test)
#Define the transformation function using columnTransformer, OHE and StandardScaler

def transform(categorical_columns,numerical_columns,df):

    cat = ('categorical', ohe() , categorical_columns  )

    num = ('numeric', ss(), numerical_columns)

    col_trans = ColumnTransformer([cat, num])

    df_trans_scaled = col_trans.fit_transform(df)

    return df_trans_scaled



#Define the columns for transformations

categorical_columns = ['source', 'browser','sex']

numerical_columns = ['signup_time-purchase_time', 'purchase_value','age','N[device_id]']

#Define the columns for post transformation dataframe - makes referencing and understanding easier

columns = ['source_Direct','source_SEO','source_Ads','browser_Chrome','browser_FireFox','browser_IE','browser_Safari','browser_Opera','sex_M','sex_F'] + numerical_columns

columns
#Rename Columns names for this section

X_test_ts =transform(categorical_columns,numerical_columns,df_test)

X_test_ts.shape
#Prediction on the Test Data set

Y_predi = log_reg.predict(X_test_ts)
Y_prediction=pd.DataFrame(Y_predi)

Y_prediction.head(), Y_prediction.shape
Final_Output = pd.concat([df_test, Y_prediction], axis=1)
Final_Output.rename(columns={0:'Prediction'},inplace=True)

Final_Output.head()
#os.mkdir("../FinalResult")

#os.chdir("../FinalResult")

Final_Output.to_csv('Result_submit.csv')