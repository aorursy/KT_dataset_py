import os

import pandas as pd

import numpy as np

import pandas_profiling as pp

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
#Reading the data



fulldataset=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
#####EDA



#profile report to understand which are the variables with multiple missing values



report= pp.ProfileReport(fulldataset, check_correlation = False)
report
#The data consists of more than 56.5% missing values
#checking the percentage of missing values per variable



fulldataset.isnull().sum()
nullinfo= pd.DataFrame(fulldataset.isnull().sum())

nullinfo['percentage']=(nullinfo[0]/fulldataset.shape[0])*100

nullinfo
#dropping the variables with missing value percentage greater than 30%

columns_to_drop=nullinfo.loc[nullinfo.percentage>=30].index

fulldataset.shape
fulldataset=fulldataset.drop(columns_to_drop, axis=1)

fulldataset.shape
fulldataset.isnull().sum()
pd.concat((pd.DataFrame(fulldataset.isnull().sum()),pd.DataFrame(fulldataset.dtypes), pd.DataFrame(fulldataset.nunique())), axis=1)
#dropping variables which give the same information



fulldataset=fulldataset.drop(['country_txt','region_txt','attacktype1_txt','targtype1_txt','targsubtype1_txt','natlty1_txt','weaptype1_txt','weapsubtype1_txt'],axis=1)

fulldataset.shape
fulldataset2=fulldataset.copy()
#Missing value imputation



for Col_Name in list(fulldataset2):

    if(fulldataset2[Col_Name].dtype == object):       

        Temp_Imputation_Val = fulldataset2[Col_Name].mode()[0]

        fulldataset2[Col_Name] = fulldataset2[Col_Name].fillna(Temp_Imputation_Val)

    elif(fulldataset2[Col_Name].dtype == float):

        Temp_Imputation_Val = round(fulldataset2[Col_Name].mean()) 

        fulldataset2[Col_Name].fillna(Temp_Imputation_Val, inplace= True)

    else:

        Temp_Imputation_Val = round(fulldataset2[Col_Name].median())

        fulldataset2[Col_Name].fillna(Temp_Imputation_Val, inplace= True)



fulldataset2.isnull().sum()
#missing value imputation for the variables imonth and iday as they cannot be 0

fulldataset2[['imonth','iday']].nunique()
#month and day cannot be 0 hence this value is imputated with the most occuring date in the dataset

fulldataset2['imonth']=np.where((fulldataset2.imonth==0),fulldataset2.imonth.mode(),fulldataset2.imonth).astype('int64')

fulldataset2['iday']=np.where((fulldataset2.iday==0),fulldataset2.iday.mode(),fulldataset2.iday).astype('int64')

fulldataset2[['imonth','iday']].nunique()
pd.concat((pd.DataFrame(fulldataset2.isnull().sum()),pd.DataFrame(fulldataset2.dtypes), pd.DataFrame(fulldataset2.nunique())), axis=1)
#outlier detection and correction



summary=fulldataset2.describe()

summary=summary.transpose()

summary
fulldataset3=fulldataset2.copy()

Cols_to_correct=['latitude','longitude','nkill','nwound']

#a function to detect the outliers in the continuous variables

def Outlier_correction(Data,Columns):

    for i in Columns:

        IQR=(Data[i].quantile(0.75))-(Data[i].quantile(0.25))

        positive_IQR=(Data[i].quantile(0.75))+ (1.5*IQR)

        negative_IQR=(Data[i].quantile(0.25))- (1.5*IQR)

        Data[i]=np.where(Data[i]>positive_IQR, positive_IQR, Data[i])

        Data[i]=np.where(Data[i]<negative_IQR, negative_IQR, Data[i])

    return(Data)

    

Outlier_correction(fulldataset3, Cols_to_correct)
pd.concat((pd.DataFrame(fulldataset3.isnull().sum()),pd.DataFrame(fulldataset3.dtypes), pd.DataFrame(fulldataset3.nunique())), axis=1)
#Encoding the categorical variables



# Step 1: Identify categorical vars

Categ_Vars = fulldataset3.loc[:,fulldataset3.dtypes == object].columns

Categ_Vars
# Step 2:Encoding the categorical variable



fulldataset3[Categ_Vars].dtypes
fulldataset3[Categ_Vars]=fulldataset3[Categ_Vars].apply(preprocessing.LabelEncoder().fit_transform)

fulldataset3[Categ_Vars].dtypes
#splitting the data



fulldataset3.shape

Train, Test = train_test_split(fulldataset3, test_size=0.2, random_state = 123)

Train.shape
## Validating if Train dataset has 80% rows from fulldataset3 and also that random rows have been selected

round(Train.shape[0]/fulldataset3.shape[0],1)
Train.index[1:10] # Random indexes or rows have been selected
#splitting the dependent and the independent variables

Train_X = Train.drop(['gname'], axis = 1).copy()

Train_Y = Train['gname'].copy()

Test_X = Test.drop(['gname'], axis = 1).copy()

Test_Y = Test['gname'].copy()
#model building



#1. Decision Tree



DT1 = DecisionTreeClassifier(random_state=100, min_samples_leaf=50)

DT1_Model = DT1.fit(Train_X, Train_Y)

Test_Pred_DT = DT1_Model.predict(Test_X)
Confusion_Mat_DT = confusion_matrix(Test_Y, Test_Pred_DT) # R, C format (Actual = Test_Y, Predicted = Test_Pred)

Confusion_Mat_DT
print(accuracy_score(Test_Y, Test_Pred_DT))

print(classification_report(Test_Y, Test_Pred_DT))
#2. RandomForestClassifier()



RF1 = RandomForestClassifier(random_state = 100)

RF1_Model = RF1.fit(Train_X, Train_Y)

Test_Pred_RF = RF1_Model.predict(Test_X)
Confusion_Mat_RF = confusion_matrix(Test_Y, Test_Pred_RF) # R, C format (Actual = Test_Y, Predicted = Test_Pred)

Confusion_Mat_RF
print(accuracy_score(Test_Y, Test_Pred_RF))

print(classification_report(Test_Y, Test_Pred_RF))
#The f1-score for the decision tree model is 0.81 and for Random Forest classifier is 0.87

#There are many categories in the dependent variable 'gname' which were not predicted and thus their f-1 score is set to 0.





#Overview about the analysis



#The dataset contains a lot of missing values(56.5%) and thus a lot of the information is unavailable for analysis.

#Thus the variables with more than 30% missing values were eliminated initially.

#The imputaiton of the missing values for the remainder of the data was based on the available information of the datatype of the variable.

#for example the variable 'weapsubtype1' has datatype 'float64' but based on the values it can be considered as 'int'.

#Also many variables have datatype as 'int' but are actually categorical variables. With Advice from the SME these variables can be treated as categorical and treated as per categorical variables.

#The algorithms for Decision Tree and Random Forests were selected as they enable better classification results as they are based on GINI value and entropy unlike a regression model which is based on probability.
