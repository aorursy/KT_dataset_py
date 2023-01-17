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
#import Library

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



%matplotlib inline
data = pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")
data.head()
#Adjusting rows and columns

data.set_index('ID',inplace=True)

data.rename(columns = {"SEX":"GENDER","MARRIAGE":'MARITAL_STATUS', 'PAY_0' : 'PAY_1', "LIMIT_BAL" : "CREDIT_LIMIT", "default.payment.next.month" : "DEFAULT_STATUS"}, inplace = True)
print("For Data Description and Explaination of Variables, please refer to `APPENDIX`")

print()

data.info()
# creat a copy, in case we need variables in the format of number for analysis

data_original=data.copy()
data.describe()
data.GENDER.unique()
data["GENDER"]=data["GENDER"].apply(lambda x: "Male" if x==1 else "Female")
data.EDUCATION.unique()
data["EDUCATION"] = data["EDUCATION"].apply(lambda x: "Graduate School" if x==1 else ("Univeristy" if x==2 else ("High School" if x==3 else "Others")))
data.MARITAL_STATUS.unique()
data.PAY_1.unique()
for i in data.iloc[:,5:11]:

    data[i] = data[i].apply(lambda x: "No Delay" if x<0 else "Delay")
#Generating Frequency Distribution histograms of Numerical Variables



hist_num = data.iloc[:,:-1].hist(figsize=(20,15))

plt.suptitle('Frequency Distribution of Numerical Varibales', x=0.5, y=1.05, ha='center', fontsize='xx-large')

plt.tight_layout()
# Log the Credit Limit in new column

data["Log_limit"] = np.log(data.CREDIT_LIMIT)



# Log the Payment Amounts in new columns

index=1

for i in data.iloc[:,-8:-2]:

    log_pay_amt="Log_pay_amt"+str(index)

    data[log_pay_amt] = np.log(data[i]+1)

    index=index+1

    

data_log_transformed = data_original.copy()

data_log_transformed["CREDIT_LIMIT"] = data["Log_limit"]



# New data_log_transformed dataframe 

data_log_transformed = data_original.copy()

data_log_transformed["CREDIT_LIMIT"] = data["Log_limit"]

# Log the Payment Amounts in new columns

index=1

for i in data_log_transformed.iloc[:,-7:-1]:

    log_pay_amt="Log_pay_amt"+str(index)

    data_log_transformed[log_pay_amt] = np.log(data_log_transformed[i]+1)

    index=index+1

    

# Log the Bill Amounts in new columns

index=1

for i in data_log_transformed[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']]:

    log_pay_amt="Log_bill_amt"+str(index)

    data_log_transformed[log_pay_amt] = data_log_transformed[i].apply( lambda x: np.log1p(x) if (x>0) else 0 )

    index += 1

    

    

data_log_transformed = data_log_transformed.drop(

    columns=['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',

             'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',

            'CREDIT_LIMIT'])



#Check the result of logged distribution of Credit Limit and Payment Amount

hist_num = data.iloc[:,-7:].hist(figsize=(20,15))

plt.suptitle("Frequency Distribution of Log (Credit Limit) and Log(Payment Amount + 1) variables", x=0.5, y=1.05, ha='center', fontsize='xx-large')

plt.tight_layout()
fig, axes = plt.subplots(figsize=(15,15),nrows=3, ncols=3)

plt.suptitle('Frequency Distribution of Categorical Varibales', x=0.5, y=1.05, ha='center', fontsize='x-large')

data['GENDER'].value_counts().plot.bar(ax=axes[0,0],title="Gender")

data['EDUCATION'].value_counts().plot.bar(ax=axes[0,1],title="Education")

data['MARITAL_STATUS'].value_counts().plot.bar(ax=axes[0,2],title="Marital_Status")

i_row=1

i_col=0

count=1

for i in data.iloc[:,5:11]:

    data[i].value_counts().plot.bar(ax=axes[i_row,i_col],title="Pay_Status_"+str(count))

    count=count+1

    i_col=i_col+1

    if count>=4:

        i_row=2

    if count==4:

        i_col=0

# set title and axis labels

plt.tight_layout()

plt.show()

plt.suptitle('Frequency Distribution of Target Value', x=0.5, y=1.05, ha='center', fontsize='x-large')

data['DEFAULT_STATUS'].value_counts().plot.bar(title="Defaulted or not")
ave_defalt_rate = round(np.mean(data["DEFAULT_STATUS"]),3)*100

print("The average default rate of the dataset is " + str(ave_defalt_rate) +"%")
f, ax = plt.subplots(figsize = (20, 20))   

# this is to set fig size



correlational_matrix = data_original.corr()

#calculate correlation matrix and assign it



mask = np.triu(np.ones_like(correlational_matrix, dtype = np.bool))





cmap = sns.diverging_palette(500, 10, as_cmap = True) 

# this is just to set color range



# Range of correlational coefficients: -1 through 1



sns.heatmap(correlational_matrix,        

            mask = mask,

            cmap = cmap,                 #set color range

            vmax = 1,                    #affects the color range

            center = 0,                  #affects the color range

            square = True,               

            annot= True,                 #add annotation

            fmt=".1f",                   #set decimal place

            linewidths = 1,              #set line width

            cbar_kws = {"shrink": 0.5})  #shrink color bar by 0.5 times
data_log_transformed.head()
# Visualization of Credit Limite and Default Count using histogram

fig, ax = plt.subplots(figsize = (10, 6))

plt.hist(data.Log_limit, bins = 40, alpha = 1, color = "yellow", label="Total")

plt.hist(data.query('DEFAULT_STATUS == 0').Log_limit, bins = 40, alpha = 1, color = "orange", label = "Not Default")

plt.hist(data.query('DEFAULT_STATUS == 1').Log_limit, bins = 40, alpha = 1, color = "red", label = "Default")

plt.xlabel("Log of Credit Limit")

plt.ylabel("Count")

plt.title("Log value of Credit Limit VS Number of Defalt")

plt.grid()

plt.legend()

plt.show()
subset = data[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 

               'PAY_5', 'PAY_6', 'DEFAULT_STATUS']]



f, axes = plt.subplots(2, 3, figsize=(15, 13), facecolor='white')

f.suptitle('FREQUENCY OF DEFAULT (BY HISTORY OF DEFAULT)')



ax1 = sns.countplot(x="PAY_1", hue="DEFAULT_STATUS", data=subset, palette="Greens", ax=axes[0,0])

ax2 = sns.countplot(x="PAY_2", hue="DEFAULT_STATUS", data=subset, palette="Blues", ax=axes[0,1])

ax3 = sns.countplot(x="PAY_3", hue="DEFAULT_STATUS", data=subset, palette="Greens", ax=axes[0,2])

ax4 = sns.countplot(x="PAY_4", hue="DEFAULT_STATUS", data=subset, palette="Blues", ax=axes[1,0])

ax5 = sns.countplot(x="PAY_5", hue="DEFAULT_STATUS", data=subset, palette="Greens", ax=axes[1,1])

ax6 = sns.countplot(x="PAY_6", hue="DEFAULT_STATUS", data=subset, palette="Blues", ax=axes[1,2]);
print("Histogram by Age")



plt.figure(figsize = (8 , 6))

sns.distplot(data.query('DEFAULT_STATUS == 1').AGE, bins = 20, color="green")

mean_age = data.AGE.mean()

plt.axvline(mean_age,0,1, color = "blue")
#define a function to categorize age group



def get_group (age):

        if age < 40:

            return "Young"

        elif age >= 60:

            return "Old"

        else:

            return "Middle"
# apply to "AGE" and create a new column

data["Age_group"] = data["AGE"].apply(get_group)
print("Age Group VS Default Rate")



plt.figure(figsize = (5 , 5))

sns.barplot(x = 'Age_group', y = "DEFAULT_STATUS", data = data)

plt.show()
# Define a function to plot barplot between categorical variable and default status

def plot_cat(categorical_variable):

    sns.barplot(x = categorical_variable, y="DEFAULT_STATUS", data=data)

    plt.figure(figsize=(10,6))

    plt.show()
print("Default Rate VS Gender")

plot_cat("GENDER")
print("Default Rate VS Education Background")

plot_cat("EDUCATION")
print("Default Rate VS Marital Status")

plot_cat("MARITAL_STATUS")
print("Default Rate VS History of Delaying Payment in previous month (SEP)")

plot_cat("PAY_1")
print("Grouped age and history payment status vs Default Rate")



sns.barplot(x = 'Age_group', y = "DEFAULT_STATUS", data = data, hue = "PAY_1")

plt.show()
print("Grouped age and education background vs Default Rate")



sns.barplot(x = 'Age_group', y = "DEFAULT_STATUS", data = data, hue = "EDUCATION")

plt.show()
print("Ploting log of PAY_AMT Histogram")



plt.figure(figsize = (8 , 6))

sns.distplot(data.query('DEFAULT_STATUS == 1').Log_pay_amt1, bins = 20, color="green")

mean_amt1 = data.Log_pay_amt1.mean()

plt.axvline(mean_amt1,0,1, color = "blue")
# Define our function 

def log_amt(x):

    if x<2.5:

        return "low"

    elif x>=2.5 and x<9:

        return "medium"

    else:

        return "high"
# Apply get_amt function to SEP(PAY_AMT1) and create new column

data["log_amt1_group"]=data["Log_pay_amt1"].apply(lambda x:log_amt(x))
print("Default Status VS Payment Amount in Sep by Group")

plot_cat("log_amt1_group")
# Apply get_amt function to AUG(PAY_AMT2) and create new column

data["log_amt2_group"]=data["Log_pay_amt2"].apply(lambda x:log_amt(x))
print("Default Status VS Payment Amount in Aug by Group")

plot_cat("log_amt2_group")
# Apply the same formula to other months

data["log_amt3_group"]=data["Log_pay_amt3"].apply(lambda x:log_amt(x))

data["log_amt4_group"]=data["Log_pay_amt4"].apply(lambda x:log_amt(x))

data["log_amt5_group"]=data["Log_pay_amt5"].apply(lambda x:log_amt(x))

data["log_amt6_group"]=data["Log_pay_amt6"].apply(lambda x:log_amt(x))
f, axes = plt.subplots(2, 2, figsize=(15, 13), facecolor='white')



ax1 = sns.barplot(x = "log_amt3_group", y="DEFAULT_STATUS", data=data, ax=axes[0,0])

ax2 = sns.barplot(x = "log_amt4_group", y="DEFAULT_STATUS", data=data, ax=axes[0,1])

ax3 = sns.barplot(x = "log_amt5_group", y="DEFAULT_STATUS", data=data, ax=axes[1,0])

ax4 = sns.barplot(x = "log_amt6_group", y="DEFAULT_STATUS", data=data, ax=axes[1,1])

import sklearn

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
data_original.info()
y = data_original['DEFAULT_STATUS']

X = data_original.drop(columns=['DEFAULT_STATUS'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
log_reg = sm.Logit(y_train, X_train).fit() 
print(log_reg.summary()) 
yhat = log_reg.predict(X_test) 

prediction = list(map(round, yhat)) 



from sklearn.metrics import (confusion_matrix,  

                           accuracy_score) 

  

# confusion matrix 

cm = confusion_matrix(y_test, prediction)  

print ("Confusion Matrix : \n", cm)  

  

# accuracy score of the model 

print('Test accuracy = ', accuracy_score(y_test, prediction))

X_train=X_train[["CREDIT_LIMIT", "GENDER", "EDUCATION","MARITAL_STATUS","PAY_1", "PAY_2", "PAY_3", "BILL_AMT1", "PAY_AMT1", "PAY_AMT2"]]

X_test=X_test[["CREDIT_LIMIT", "GENDER", "EDUCATION","MARITAL_STATUS","PAY_1", "PAY_2", "PAY_3", "BILL_AMT1", "PAY_AMT1", "PAY_AMT2"]]
log_reg1 = sm.Logit(y_train, X_train).fit() 
print(log_reg1.summary()) 
yhat = log_reg1.predict(X_test) 

prediction = list(map(round, yhat)) 



from sklearn.metrics import (confusion_matrix,  

                           accuracy_score) 

  

# confusion matrix 

cm = confusion_matrix(y_test, prediction)  

print ("Confusion Matrix : \n", cm)  

  

# accuracy score of the model 

print('Test accuracy = ', accuracy_score(y_test, prediction))

data.info()
data_d=data.drop(columns=['CREDIT_LIMIT','AGE','PAY_AMT1','PAY_AMT1','PAY_AMT2','PAY_AMT3',

                          'PAY_AMT4','PAY_AMT5','PAY_AMT6','Log_pay_amt1','Log_pay_amt1','Log_pay_amt2',

                          'Log_pay_amt3','Log_pay_amt4','Log_pay_amt5','Log_pay_amt6'])
data_d=pd.get_dummies(data_d)

y = data_d['DEFAULT_STATUS']

X = data_d.drop(columns=['DEFAULT_STATUS'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
log_reg2 = sm.Logit(y_train, X_train).fit() 
print(log_reg2.summary()) 
yhat = log_reg2.predict(X_test) 

prediction = list(map(round, yhat)) 



from sklearn.metrics import (confusion_matrix,  

                           accuracy_score) 

  

# confusion matrix 

cm = confusion_matrix(y_test, prediction)  

print ("Confusion Matrix : \n", cm)  

  

# accuracy score of the model 

print('Test accuracy = ', accuracy_score(y_test, prediction))
X_train = X_train['Log_limit']

X_test = X_test['Log_limit']
log_reg3 = sm.Logit(y_train, X_train).fit()

print(log_reg3.summary())
yhat = log_reg3.predict(X_test) 

prediction = list(map(round, yhat)) 



from sklearn.metrics import (confusion_matrix,  

                           accuracy_score) 

  

# confusion matrix 

cm = confusion_matrix(y_test, prediction)  

print ("Confusion Matrix : \n", cm)  

  

# accuracy score of the model 

print('Test accuracy = ', accuracy_score(y_test, prediction))
from sklearn.utils import resample
data.columns
data_majority = data[data.DEFAULT_STATUS==0]

data_minority = data[data.DEFAULT_STATUS==1]



print(data_majority.DEFAULT_STATUS.count())

print("-----------")

print(data_minority.DEFAULT_STATUS.count())

print("-----------")

print(data.DEFAULT_STATUS.value_counts())
# Upsample minority class

data_minority_upsampled = resample(data_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=23364,    # to match majority class

                                 random_state=777) # reproducible results



# Combine majority class with upsampled minority class

data_upsampled = pd.concat([data_majority, data_minority_upsampled])

# Display new class counts

data_upsampled.DEFAULT_STATUS.value_counts()
# Downsample majority class

data_majority_downsampled = resample(data_majority, 

                                 replace=False,    # sample without replacement

                                 n_samples=6636,     # to match minority class

                                 random_state=777) # reproducible results



# Combine minority class with downsampled majority class

data_downsampled = pd.concat([data_majority_downsampled, data_minority])

# Display new class counts

data_downsampled.DEFAULT_STATUS.value_counts()
## remember to pip install imbalanced-learn



from imblearn.over_sampling import SMOTE
y = data_log_transformed['DEFAULT_STATUS']

X = data_log_transformed.drop(columns=['DEFAULT_STATUS'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
smote = SMOTE(random_state=123)

X_SMOTE, y_SMOTE = smote.fit_sample(X_train, y_train)

print(len(y_SMOTE))

print(y_SMOTE.sum())
y_SMOTE.value_counts()
log_reg_smote = sm.Logit(y_SMOTE, X_SMOTE).fit()

print(log_reg_smote.summary())
#-------------- 

# logistic regression 

#--------------

yhat = log_reg_smote.predict(X_test) 

prediction = list(map(round, yhat)) 



from sklearn.metrics import (confusion_matrix,  

                           accuracy_score) 

  

# confusion matrix 

cm = confusion_matrix(y_test, prediction)  

print ("Confusion Matrix : \n", cm)  

  

# accuracy score of the model 

print('Test accuracy = ', accuracy_score(y_test, prediction))
#-------------- 

# kernel SVM 

#--------------

from sklearn.svm import SVC

classifier1 = SVC(kernel="rbf")

classifier1.fit( X_SMOTE, y_SMOTE )

y_pred = classifier1.predict( X_test )



cm = confusion_matrix( y_test, y_pred )

print("Accuracy on Test Set for kernel-SVM = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))

# confusion matrix 

print ("Confusion Matrix : \n", cm) 
exit()