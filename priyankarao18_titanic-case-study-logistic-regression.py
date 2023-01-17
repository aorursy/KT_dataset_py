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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_train.head()
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_test.head()
df_train.info()
df_test.info()
df_train.shape
df_train.isnull().sum()
df_train.Age.isnull().sum()/df_train.Age.isnull().count()
df_train.Cabin.isnull().sum()/df_train.Cabin.isnull().count()
# Replacing the null values in age column with median score and dropping the cabin column as null values are greater than 50%

df_train.Age = df_train.Age.fillna(df_train.Age.median())
df_train.Embarked = df_train.Embarked.fillna(df_train.Embarked.mode()[0])

df_train.drop("Cabin" , axis=1 , inplace=True)
df_train.isnull().sum()
#Performing the above steps on test data

df_test.Age = df_test.Age.fillna(df_test.Age.median())
df_test.Fare = df_test.Fare.fillna(df_test.Fare.median())

df_test.drop("Cabin" , axis=1 , inplace=True)
df_test.isnull().sum()
#Creating a new feature varibale

df_train["Family"] = np.where(df_train["SibSp"]+df_train["Parch"] > 0 , 1 , 0)
df_train.drop("SibSp" , axis=1 , inplace=True)
df_train.drop("Parch" , axis=1 , inplace=True)

#Renaming the fields 

df_train.Pclass = df_train.Pclass.map({3 : "Lower" , 2 : "Middle" , 1 : "Upper"})
df_train.Age = pd.cut(df_train.Age , [0 , 16 , 24 , 55 , 75] , labels = [ "Minor" , "Youths" , "Adults" , "Senior Citizen"])
#Performing the above steps on test data

df_test["Family"] = np.where(df_test["SibSp"]+df_test["Parch"] > 0 , 1 , 0)
df_test.drop("SibSp" , axis=1 , inplace=True)
df_test.drop("Parch" , axis=1 , inplace=True)


df_test.Pclass = df_test.Pclass.map({3 : "Lower" , 2 : "Middle" , 1 : "Upper"})
df_test.Age = pd.cut(df_test.Age , [0 , 16 , 24 , 55 , 75] , labels = [ "Minor" , "Youths" , "Adults" , "Senior Citizen"])
plt.figure(figsize =(15,10))
plt.subplot(3,3,1)
sns.countplot(data= df_train , x = "Survived")
plt.xticks([0,1] , ["Drowned" , "Survived"])
plt.subplot(3,3,2)
sns.countplot(data= df_train , x = "Pclass")
plt.subplot(3,3,3)
sns.countplot(data= df_train , x = "Sex")
plt.subplot(3,3,4)
sns.countplot(data= df_train , x = "Age")
plt.subplot(3,3,5)
sns.countplot(data= df_train , x = "Embarked")
plt.subplot(3,3,6)
sns.countplot(data= df_train , x = "Family")
plt.xticks([0,1] , ["No" , "Yes"])
plt.show()
plt.figure(figsize = [15,10])
plt.subplot(2,2,1)
sns.countplot(x = "Embarked" , hue="Survived" , data = df_train )
plt.title("Port of Embarkation")

plt.subplot(2,2,2)
sns.countplot(x="Sex" , hue = "Survived" , data=df_train)
plt.title("Sex of Passenger")

plt.subplot(2,2,4)
sns.countplot(x = "Pclass" , hue="Survived" , data=df_train)
plt.title("Passenger class")

plt.subplot(2,2,3)
sns.countplot(x="Family" , hue="Survived" , data=df_train)
plt.xticks([0,1] , ["No" , "Yes"])
plt.title("Presence of family")
plt.show()
df_train.groupby("Age")["Survived"].value_counts(normalize =True).unstack().plot(kind = "bar" , stacked = True)
plt.title("Age-wise Distribution")
plt.ylabel("Number of passengers")
plt.show()
sns.catplot(x ="Age" , y ="Fare" , data=df_train , kind="box" )
plt.show()
sns.catplot(x ="Pclass" , y ="Fare" , data=df_train , kind="box" )
plt.show()
sns.catplot(x ="Embarked" , y ="Fare" , data=df_train , kind="box" )
plt.show()
#creating dummies for categorical variables with more than 2 categories

data_train = pd.get_dummies(df_train , columns = ["Pclass" , "Embarked" , "Sex" , "Age"] , drop_first =True )
final_data_train= data_train

#Dropping irrelavent columns

final_data_train.drop("Name" , axis = 1 , inplace=True)
final_data_train.drop("Ticket" , axis = 1 , inplace=True)
final_data_train.drop("PassengerId" , axis = 1 , inplace=True)
final_data_train.head()
#Performing the above steps on test data 
data_test = pd.get_dummies(df_test , columns = ["Pclass" , "Embarked" , "Sex" , "Age"] , drop_first = True)

final_data_test= data_test

final_data_test.drop("Name" , axis = 1 , inplace=True)
final_data_test.drop("Ticket" , axis = 1 , inplace=True)
final_data_test.drop("PassengerId" , axis = 1 , inplace=True)
final_data_test.head()
y = final_data_train["Survived"]
X = final_data_train[final_data_train.columns[1:]]
#Spliting the initial training data into test and train data set for modelling

X_train , X_test , y_train , y_test = train_test_split(X , y , train_size = 0.7 , test_size = 0.3 ,random_state = 100)
#Scaling the numerical column so that the values are comparable 
#fit and transform is performed on train data set 

scaler = MinMaxScaler()
X_train[["Fare"]] = scaler.fit_transform(X_train[["Fare"]])

#for test data set only tranform is performed
X_test[["Fare"]] = scaler.transform(X_test[["Fare"]])
plt.figure(figsize = (15,10))
sns.heatmap(final_data_train.corr() , cmap = "Greens" ,annot=True)
plt.show()
#Model 1 

mod1 = sm.GLM(y_train , (sm.add_constant(X_train)) , family = sm.families.Binomial())
mod1.fit().summary()
col = X_train.columns

#Calculating the VIF score for model 1

vif = pd.DataFrame()
vif["Features"] = X_train[col].columns
vif["VIF"] = [variance_inflation_factor(X_train[col].values , i) for i in range(X_train[col].shape[1])]
vif["VIF"] = round(vif["VIF"] ,2)
vif = vif.sort_values(by = "VIF" , ascending = False )
vif
col1 = col.drop("Fare" , 1)
#Model 2 

mod2 = sm.GLM(y_train , sm.add_constant(X_train[col1]) , family = sm.families.Binomial())
mod2.fit().summary()
vif = pd.DataFrame()
vif["Features"] = X_train[col1].columns
vif["VIF"] = [variance_inflation_factor(X_train[col1].values , i) for i in range(X_train[col1].shape[1])]
vif["VIF"] = round(vif["VIF"] ,2)
vif = vif.sort_values(by = "VIF" , ascending = False )
vif
col2 = col1.drop("Family" , 1)
#Model 3

mod3 = sm.GLM(y_train , sm.add_constant(X_train[col2]) , family = sm.families.Binomial())
mod3.fit().summary()
vif = pd.DataFrame()
vif["Features"] = X_train[col2].columns
vif["VIF"] = [variance_inflation_factor(X_train[col2].values , i) for i in range(X_train[col2].shape[1])]
vif["VIF"] = round(vif["VIF"] ,2)
vif = vif.sort_values(by = "VIF" , ascending = False )
vif
col3 = col2.drop("Embarked_Q" , 1)
#Model 4

mod4 = sm.GLM(y_train , sm.add_constant(X_train[col3]) , family = sm.families.Binomial())
mod4.fit().summary()
vif = pd.DataFrame()
vif["Features"] = X_train[col3].columns
vif["VIF"] = [variance_inflation_factor(X_train[col3].values , i) for i in range(X_train[col3].shape[1])]
vif["VIF"] = round(vif["VIF"] ,2)
vif = vif.sort_values(by = "VIF" , ascending = False )
vif
res = mod4.fit()
#predicting the survival proability

X_train_sm = sm.add_constant(X_train[col3])
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred_final = pd.DataFrame({"Survived_actual" : y_train.values , "Survival_Prob" : y_train_pred })
y_train_pred_final["Passenger ID"] = y_train.index
y_train_pred_final.head()
# Drawing the ROC curve to find if the model is good or  not


def draw_roc(actual , probs):
    fpr , tpr , thresholds = metrics.roc_curve(actual , probs, drop_intermediate=False)
    auc_score = metrics.roc_auc_score(actual,probs)
    plt.figure(figsize = [10,5])
    plt.plot(fpr , tpr , label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0 , 1.0])
    plt.ylim([0.0 , 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    
    return None
fpr , tpr , thresholds = metrics.roc_curve(y_train_pred_final.Survived_actual , y_train_pred_final.Survival_Prob , drop_intermediate = False)
print("False positive rate: " , fpr)
print("True Positive rate: " , tpr)
print("Threshold value :" , thresholds)
metrics.roc_auc_score(y_train_pred_final.Survived_actual , y_train_pred_final.Survival_Prob)
#Plot of TPR vs FPR

draw_roc(y_train_pred_final.Survived_actual , y_train_pred_final.Survival_Prob)
# Predicting for different possible threshold values 

numbers = [float(x/10) for x in range(10)]
for i in numbers :
    y_train_pred_final[i] = y_train_pred_final.Survival_Prob.map(lambda x : 1 if x > i else 0)
y_train_pred_final.head()
cutoff_df = pd.DataFrame(columns = ["prob" , "accuracy" , "sensitivity" , "specificity"])

for i in numbers:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived_actual , y_train_pred_final[i])
    accuracy = (cm1[0 ,0]+cm1[1,1])/sum(sum(cm1))
    sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    specificity = cm1[0,0]/(cm1[0,1] +cm1[0,0])
    cutoff_df.loc[i] = [i , accuracy, sensitivity , specificity]
    
cutoff_df.head()
#the point where the 3 values meet is taken as the optimal and best threshold value

cutoff_df.plot.line(x="prob" , y = ["accuracy" , "sensitivity" , "specificity"])
plt.show()
#As shown in the graph 0.4 is chosen as the optimal threshold value.

y_train_pred_final["predicted"] = y_train_pred_final.Survival_Prob.map(lambda x: 1 if x>0.3 else 0)
y_train_pred_final.head()
metrics.accuracy_score(y_train_pred_final.Survived_actual , y_train_pred_final.predicted)
conf_matrix = metrics.confusion_matrix(y_train_pred_final.Survived_actual , y_train_pred_final.predicted)

TN = conf_matrix[0,0]
FP = conf_matrix[0,1]
FN = conf_matrix[1,0]
TP = conf_matrix[1,1]
sensitivity = TP/float(TP+FN)
sensitivity
specificity = TN/float(FP+TN)
specificity
precision = TP/float(TN+TP)
precision
recall = TP/float(TP+FN)
recall
X_test_sm = sm.add_constant(X_test[col3])
X_test_sm.head()
y_test_pred = res.predict(X_test_sm)
y_test_data_pred = pd.DataFrame(y_test_pred)
y_test_data_pred["Passenger_id"] = df_train.PassengerId
y_test_data_pred.rename(columns = { 0 :"survival_prob"} , inplace =True)
y_test_data_pred.head()
y_test_data_pred["pred_test"] = y_test_data_pred.survival_prob.map(lambda x: 1 if x>0.3 else 0 )
y_test_data_pred["original"] = df_train.Survived
y_test_data_pred.head()
metrics.accuracy_score(y_test_data_pred.original , y_test_data_pred["pred_test"])
conf_matrix = metrics.confusion_matrix(y_test_data_pred.original , y_test_data_pred["pred_test"])

TN = conf_matrix[0,0]
FP = conf_matrix[0,1]
FN = conf_matrix[1,0]
TP = conf_matrix[1,1]
sensitivity = TP/float(TP+FN)
sensitivity
specificity = TN/float(TN+FP)
specificity
precision_score(y_test_data_pred.original , y_test_data_pred["pred_test"])
recall_score(y_test_data_pred.original , y_test_data_pred["pred_test"])
f1_score(y_test_data_pred.original , y_test_data_pred["pred_test"])
#first we transform the numerical variable 
final_data_test[["Fare"]] = scaler.transform(final_data_test[["Fare"]])
gender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
gender.head()
final_test_sm = sm.add_constant(final_data_test[col3])
final_test_pred = res.predict(final_test_sm)
final_result = pd.DataFrame(final_test_pred)

final_result["Passenger_id"] = gender.PassengerId
final_result.head()
final_result.rename(columns = { 0 :"survival_prob"} , inplace =True)
final_result["pred_test"] = final_result.survival_prob.map(lambda x: 1 if x>0.4 else 0 )
final_result["original"] = gender.Survived
final_result.head()
metrics.accuracy_score(final_result.original , final_result["pred_test"])
f1_score(final_result.original , final_result["pred_test"])
#Final Result

result = final_result
result.drop(["survival_prob" , "original"] ,axis =1 , inplace =True)
result.rename(columns = {"Passenger_id" : "PassengerId" , "pred_test" : "Survived"} , inplace =True)
result.head()
result.to_csv('./submission.csv' , index = False , header =True)
submission = pd.read_csv("submission.csv")
submission.head()