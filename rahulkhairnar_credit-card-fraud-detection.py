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
import scipy.stats as scipyStats

import numpy as np

import pandas as pd

import matplotlib.pyplot as mpl

from sklearn.preprocessing import StandardScaler, RobustScaler   #MODULE FOR SCALING DATA

import seaborn as sb

import imblearn

##IMPORTING ALL THE LIBRARIES WE WILL NEED FOR THIS PROJECT
df = pd.read_csv("../input/creditcardfraud/creditcard.csv") ## TO READ THE CSV FILE creditcard.csv WHICH CONTAINS OUR DATA
df.head()
df.info()
df.isnull().sum() ## TO CHECK THE TOTAL NUMBER OF NULL VALUES IN EVERY COLUMN. OUTPUT SAYS NO NULL VALUES
class_values = df["Class"].value_counts()

count_0 = class_values[0] ## STORE THE COUNT OF 0 IN THE VARIABLE

count_1 = class_values[1] ## STORE THE COUNT OF 1 IN THE VARIABLE

count_0_percent = ((count_0)/(count_0+count_1))*100 ##CALCULATE THE PERCENTAGES OF THE 0's AND 1's IN THE TOTAL DATA SET

count_1_percent = ((count_1)/(count_0+count_1))*100

class_values
fig = mpl.figure()

ax = fig.add_axes([0,0,1,1])

Class_labels = ["Fraud","Not Fraud"]

percentages = [count_1_percent,count_0_percent]

ax.bar(Class_labels,percentages)

mpl.title("Percentage of Fraudulent vs Non Fraudulent Transactions")

mpl.show()
amount_value = df["Amount"].value_counts()

mpl.style.use('ggplot')

mpl.hist(df["Amount"],bins=100)

mpl.title("Distribution of Transactions Amounts")

sb.distplot(df["Amount"])

mpl.show()
mpl.scatter(df["Class"].values, df["Amount"].values)

mpl.title("Transaction Amounts vs Class")

mpl.xlabel("Class-Fruadulent/Not Fraudulent")

mpl.ylabel("Transaction Amounts")

mpl.show
mpl.style.use('ggplot')

mpl.hist(df["Time"],bins=100)

mpl.title("Distribution of time of Transactions")

mpl.show()
df_without_time = df.iloc[:,1:31]  ##REMOVING THE TIME COLUMN BECAUSE DOESNOT GIVE ANY SIGNIFICANT INSIGHTS

df_without_time.head() 
df_corr = df.corr()

df_corr
map = sb.heatmap(df_corr, linewidth = 1.0)

mpl.title("Heat Map")

mpl.show()
scaler = RobustScaler().fit(df_without_time.iloc[:,:-1])

scaler.transform(df_without_time.iloc[:,:-1])

df_without_time.head()
df_without_time["Class"] = df_without_time["Class"].astype("category")

df_without_time["Class"] = df_without_time["Class"].cat.rename_categories({0:"Not Fraud",1:"Fraud"})

df_without_time["Class"]
from sklearn.model_selection import train_test_split
Output_para = df_without_time.iloc[:,:30] ## SEPARATING OUTPUT AND INPUT PARAMETER COLUMNS

Input_parameters = df_without_time.iloc[:,0:29]

Output_parameter = Output_para.iloc[:,-1]

Output_parameter.value_counts()
Data_train, Data_test, Class_train, Class_test = train_test_split(Input_parameters,Output_parameter, test_size=0.3, random_state=100)
print(Class_train.value_counts())
print("Test Data Size: ",Data_test.shape)

print("Training Data Size: ", Data_train.shape)

print("Training Data Output Data Size: ", Class_train.shape)

print("Testing Data Output Size: ", Class_train.shape)
from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler
random_under = RandomUnderSampler(sampling_strategy = 'auto')

Data_under_sampled, Class_under_sampled = random_under.fit_resample(Data_train, Class_train)

Data_under_sampled.shape
random_over = RandomOverSampler(sampling_strategy = 'auto')

Data_over_sampled, Class_over_sampled = random_over.fit_resample(Data_train, Class_train)

Class_over_sampled.shape
from imblearn.over_sampling import SMOTE
oversample = SMOTE()

Smote_Data_train,Smote_class_train = oversample.fit_resample(Data_train,Class_train) 

Smote_Data_train.shape
from imblearn.under_sampling import NearMiss
sample = NearMiss()

Data_NM_train,Class_NM_train = sample.fit_resample(Data_train,Class_train)

Class_NM_train.shape
from sklearn.metrics import roc_curve,roc_auc_score

from sklearn import tree 
auc_values = []

def calc_auc(clf):

    probs = clf.predict_proba(Data_test)

    probs = probs[:,1]

    auc_calc = round(roc_auc_score(Class_test, probs),2)

    auc_values.append(auc_calc)

    fpr, tpr, thresholds = roc_curve(Class_test,probs, pos_label='Not Fraud')

    mpl.plot(fpr, tpr, color='red', label='ROC')

    mpl.plot([0, 1], [0, 1], color='DarkBlue', linestyle='--')

    mpl.title("Receiver Operating Characteristic (ROC) Curve")

    mpl.xlabel("False Positive Rate")

    mpl.ylabel("True Positive Rate")

    mpl.legend()

    mpl.show()

    return auc_calc
clf = tree.DecisionTreeClassifier()

    #for original training dataset

clf = clf.fit(Data_train, Class_train)

print("AUC is: ",calc_auc(clf))
clf = tree.DecisionTreeClassifier()



#for original training dataset

clf = clf.fit(Data_over_sampled, Class_over_sampled)

auc = calc_auc(clf)

print("AUC is: ",auc)
clf = tree.DecisionTreeClassifier()

#for original training dataset

clf = clf.fit(Data_under_sampled, Class_under_sampled)

auc = calc_auc(clf)

print("AUC is: ",auc)
clf = tree.DecisionTreeClassifier()

#for original training dataset

clf = clf.fit(Smote_Data_train,Smote_class_train)

auc = calc_auc(clf)

print("AUC is: ",auc)
clf = tree.DecisionTreeClassifier()

#for original training dataset

clf = clf.fit(Data_NM_train,Class_NM_train)

auc = calc_auc(clf)

print("AUC is: ",auc)
from sklearn.linear_model import LogisticRegression #linear regression

from sklearn.svm import SVC #svc

from sklearn.neighbors import KNeighborsClassifier #knn

from sklearn.ensemble import RandomForestClassifier #random forest

from sklearn.metrics import classification_report 

import xgboost as xgb #XGBoost



##IMPORTING ALL THE ALGORITHMS
log_reg = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='ovr').fit(Data_under_sampled, Class_under_sampled)

auc = calc_auc(log_reg)

print("AUC is: ",auc)
random_forest = RandomForestClassifier().fit(Data_under_sampled,Class_under_sampled)

auc = calc_auc(random_forest)

print("AUC is: ",auc)
KNN_value = KNeighborsClassifier().fit(Data_under_sampled,Class_under_sampled)

auc = calc_auc(KNN_value)

print("AUC is: ",auc)
xgb_Boost = xgb.XGBClassifier().fit(Data_under_sampled,Class_under_sampled)

auc = calc_auc(xgb_Boost)

print("AUC is: ",auc)