#Import basic packages
import numpy as np
import pandas as pd

#Package for data visualisation
import matplotlib.pyplot as plt

#Packages for preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

#Packages for modelling 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

#Package for evaluation
from sklearn.model_selection import cross_val_score
#Read training & testing dataset and store it as DataFrame 
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

#Check the shape of each of the dataset
print(df_train.shape)
print(df_test.shape)
#Assign a value and create a new column in testing set.
df_test['target'] = 10
#Combine both dataset and denote to (df_all)
df_all = df_train.append(df_test, sort = True)
#Take a look at the summary of each column
df_all.info()
#Replace (-1) with NaN
df_all = df_all.replace(-1,np.nan)
#Let's look at the summary again
df_all.info()
#Let's list down all the categorical variables that contain NaN. 
cat_na = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_05_cat', 'ps_car_07_cat', 'ps_car_09_cat']
#Bar chart plot regards on the frequency count of each category in each of the categorical variables that contain NaN. 
for i in cat_na:
    my_tab = pd.crosstab(index = df_all[i],columns="count")    
    my_tab.plot.bar()
    plt.show()
#Fill NaN with most frequently number
for i in cat_na:
    df_all[i] = df_all[i].fillna(df_all[i].mode()[0])
#List down all the continuous variables that contain NaN
cont_na = ['ps_reg_03', 'ps_car_11', 'ps_car_12', 'ps_car_14']
#It is not wise to have a frequency plot for continuous variable...give it a try if you want to find out. 
#Fill NaN with mean
for i in cont_na:
    df_all[i] = df_all[i].fillna(df_all[i].mean())
#Use nunique method to determine the number of unique values in each column
def count_unique_value(dataframe):
    df = pd.DataFrame()
    df['No. of unique value'] = dataframe.nunique()
    df['DataType'] = dataframe.dtypes
    return df

print(count_unique_value(df_all))
#Change datatype to 'Category' for the columns with number of unique value <= 20. 
def change_datatype(dataframe):
    col = dataframe.columns
    for i in col:
        if dataframe[i].nunique()<=20:
            dataframe[i] = dataframe[i].astype('category')
    
change_datatype(df_all)

#Change the datatype of target to int64. 
df_all['target'] = df_all['target'].astype('int64')
#Convert categorical variables to dummy variables
df_all_dummy = pd.get_dummies(df_all, drop_first = True)
#Split the combined dataset into training set & testing set
df_train_adj = df_all_dummy[df_all_dummy['target'] != 10]
df_test_adj = df_all_dummy[df_all_dummy['target'] == 10]
#Extract training data from training set
data_to_train = df_train_adj.drop(['target','id'], axis = 1)
#Extract labels from training set
labels_to_use = df_train_adj['target']
#Build different model

#Logistic Regression
logreg = make_pipeline(RobustScaler(), LogisticRegression())

#SGD Classifier
sgd = make_pipeline(RobustScaler(), SGDClassifier(loss="log"))

#Random Forest Classifier
rfc = make_pipeline(RobustScaler(), RandomForestClassifier(50))
def evaluation_auc(model):
    result= cross_val_score(model, data_to_train, labels_to_use, cv = 3, scoring = 'roc_auc')
    return(result)
#Score for Logistic Regression
score = evaluation_auc(logreg)
print("\nLogistic Regression Score: {:.5f} ({:.5f})\n".format(score.mean(), score.std()))
#Score for SGD Classifier
score = evaluation_auc(sgd)
print("\nSGD Classifier Score: {:.5f} ({:.5f})\n".format(score.mean(), score.std()))
#Score for Random Forest Classifier
score = evaluation_auc(rfc)
print("\nRandom Forest Classifier score: {:.5f} ({:.5f})\n".format(score.mean(), score.std()))
#Submission preparation
test_df_id = df_test['id']
test_df_x = df_test_adj.drop(['target', 'id'], axis = 1)
logreg.fit(data_to_train, labels_to_use)

#As we are predicting probability, use predict_proba instead of predict! 
test_df_y = logreg.predict_proba(test_df_x)[:,1]

submission = pd.DataFrame({'id': list(test_df_id), 'target': list(test_df_y)})
submission.to_csv('sgd_log.csv')