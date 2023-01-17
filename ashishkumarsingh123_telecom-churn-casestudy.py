# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime as dt



# For Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.decomposition import PCA

from sklearn import linear_model, metrics

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.feature_selection import RFE

from sklearn.metrics import r2_score

import os



# To Supress Warnings

import warnings

warnings.filterwarnings('ignore')

                        

pd.set_option("display.max_rows", 500)

pd.set_option("display.max_columns", 500)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Creating Function for identifying all info about all variables in Dataframe including the unique and Null info

def get_variable_type(var) :

    if var==0:

        return "Not Known"

    elif var < 20 and var!=0 :

        return "Categorical"

    elif var >= 20 and var!=0 :

        return "Contineous"



def predict_variable_type(metadata):

    metadata["Variable_Type"] = metadata["Unique_Values_Count"].apply(get_variable_type).astype(str)

    metadata["frequency"] = metadata["Null_Count"] - metadata["Null_Count"]

    metadata["frequency"].astype(int)

    return metadata



def get_meta_data(df) :

    metadata = pd.DataFrame({

                    'Datatype' : df.dtypes.astype(str), 

                    'Non_Null_Count': df.count(axis = 0).astype(int), 

                    'Null_Count': df.isnull().sum().astype(int), 

                    'Null_Percentage': df.isnull().sum()/len(df) * 100, 

                    'Unique_Values_Count': df.nunique().astype(int) 

                     })

    

    metadata = predict_variable_type(metadata)

    return metadata



          

def list_potential_categorical_type(dataframe,main) :

    header("Stats for potential Categorical datatype columns")

    metadata_matrix_categorical = dataframe[dataframe["Variable_Type"] == "Categorical"]

    # TO DO *** Add check to skip below if there is no Categorical values 

    length = len(metadata_matrix_categorical)

    if length == 0 :

        header_red("No Categorical columns in given dataset.")  

    else :    

        metadata_matrix_categorical = metadata_matrix_categorical.filter(["Datatype","Unique_Values_Count"])

        metadata_matrix_categorical.sort_values(["Unique_Values_Count"], axis=0,ascending=False, inplace=True)

        col_to_check = metadata_matrix_categorical.index.tolist()

        name_list = []

        values_list = []

        for name in col_to_check :

            name_list.append(name)

            values_list.append(main[name].unique())

        temp = pd.DataFrame({"index":name_list,"Unique_Values":values_list})

        metadata_matrix_categorical = metadata_matrix_categorical.reset_index()

        metadata_matrix_categorical = pd.merge(metadata_matrix_categorical,temp,how='inner',on='index')

        display(metadata_matrix_categorical.set_index("index")) 

   

def get_potential_categorical_type(dataframe,main,unique_count):

    metadata_matrix_categorical = dataframe[dataframe["Variable_Type"] == "Categorical"]

    metadata_matrix_categorical = dataframe[dataframe["Unique_Values_Count"] == unique_count]

    length = len(metadata_matrix_categorical)

    if length == 0 :

        header_red("No Categorical columns in given dataset.")  

    else :    

        metadata_matrix_categorical = metadata_matrix_categorical.filter(["Datatype","Unique_Values_Count"])

        metadata_matrix_categorical.sort_values(["Unique_Values_Count"], axis=0,ascending=False, inplace=True)

        col_to_check = metadata_matrix_categorical.index.tolist()

        name_list = []

        values_list = []

        for name in col_to_check :

            name_list.append(name)

            values_list.append(main[name].unique())

        temp = pd.DataFrame({"index":name_list,"Unique_Values":values_list})

        metadata_matrix_categorical = metadata_matrix_categorical.reset_index()

        metadata_matrix_categorical = pd.merge(metadata_matrix_categorical,temp,how='inner',on='index')

        display(metadata_matrix_categorical.set_index("index")) 

           

#Function for creating Heatmaps

def heatmap(x,y,df):

    plt.figure(figsize=(x,y))

    sns.heatmap(df.corr(),cmap="YlGn",annot=True)

    plt.show()



# create box plot for  6th, 7th and 8th month

def plot_box_chart(attribute,df):

    plt.figure(figsize=(20,16))

    plt.subplot(2,3,1)

    sns.boxplot(data=df, y=attribute+"_6",x="churn",

                showfliers=False)

    plt.subplot(2,3,2)

    sns.boxplot(data=df, y=attribute+"_7",x="churn",

                showfliers=False)

    plt.subplot(2,3,3)

    sns.boxplot(data=df, y=attribute+"_8",x="churn",

                showfliers=False)

    plt.show()

    

def plot(var, df):

    if df[var].dtype in ['int64','float64']:

        sns.distplot(df.var)

    elif df[var].dtype in ['object']:

        sns.countplot(df.var)
# lets import the dataset

telecom = pd.read_csv("/kaggle/input/telecom-churn-dataset/telecom_churn_data.csv")

telecom.head()
print(telecom.shape, telecom.info(verbose=1))
# Checking unique values, null percentage, variable type etc for each column

get_meta_data(telecom)
telecom.describe(percentiles=[.01,.99])
telecom['total_rech_data_6'].replace(np.NaN,0.0,inplace=True)

telecom['total_rech_data_7'].replace(np.NaN,0.0,inplace=True)

telecom['total_rech_data_8'].replace(np.NaN,0.0,inplace=True)

telecom['av_rech_amt_data_6'].replace(np.NaN,0.0,inplace=True)

telecom['av_rech_amt_data_7'].replace(np.NaN,0.0,inplace=True)

telecom['av_rech_amt_data_8'].replace(np.NaN,0.0,inplace=True)

telecom['max_rech_data_6'].replace(np.NaN,0.0,inplace=True)

telecom['max_rech_data_7'].replace(np.NaN,0.0,inplace=True)

telecom['max_rech_data_8'].replace(np.NaN,0.0,inplace=True)
# Total Data recharge can be calculated using total_rech_data_X and av_rech_amt_data_X feilds in the data

telecom['total_data_rech_amt_6'] = telecom.total_rech_data_6 * telecom.av_rech_amt_data_6

telecom['total_data_rech_amt_7'] = telecom.total_rech_data_7 * telecom.av_rech_amt_data_7



# We can now sum up data and non data recharge to get total recharge amount by customer



telecom['Tot_recharge_amt_6'] = telecom.total_data_rech_amt_6 + telecom.total_rech_amt_6

telecom['Tot_recharge_amt_7'] = telecom.total_data_rech_amt_7 + telecom.total_rech_amt_7



#Average recharge in June and July

telecom['AV_Tot_rech_6_7'] = (telecom.Tot_recharge_amt_6 + telecom.Tot_recharge_amt_7)/2



# Calculating 70th percentile of the Total average recharge amount

print("70th percentile is : ", telecom.AV_Tot_rech_6_7.quantile(0.7))



# fitler the data based on 70th percentile of total Average recharge value

telecom_High_value = telecom.loc[telecom.AV_Tot_rech_6_7 >= telecom.AV_Tot_rech_6_7.quantile(0.7), :]

telecom_High_value = telecom_High_value.reset_index(drop=True)



print("Dimensions of the filtered dataset:",telecom_High_value.shape)
# "Churn" would be the target variable to identify if a customer has churned, values would be either 1 (churn) or 0 (non-churn)

# we can calculate churn/non-churn based on the usage in the last month as mentioned in the problem statement

#Customers who do not have any incoming calls or outgoing calls or 2G or 3G data usage can be termed as Churned

telecom_High_value['churn'] = np.where(telecom_High_value[['total_ic_mou_9','total_og_mou_9','vol_2g_mb_9','vol_3g_mb_9']].

                                 sum(axis=1) == 0, 1,0)

telecom_High_value.head()
# Checking churn/non churn percentage

telecom_High_value['churn'].value_counts()/len(telecom_High_value)*100
# We can drop the _9 columns now, since we needed them only to calculate churn

#Extract list of _9 columns

_9_cols = telecom_High_value.columns[telecom_High_value.columns.str.contains('_9',regex=True)]

print("DF Shape before Dropping columns:",telecom_High_value.shape)

telecom_High_value.drop(_9_cols,axis=1,inplace=True)

print("DF Shape after Dropping columns:",telecom_High_value.shape)

print("Dropped columns:",_9_cols)
#Dropping columns with 1 unique value

def drop_unique_cols(df):

    Dropped_cols=[]

    print("DF Shape before Dropping columns:",df.shape)

    for col in df.columns:

        if df[col].nunique()==1:

            Dropped_cols.append(col)

            df.drop(col,axis=1, inplace=True)

    print("DF Shape after Dropping columns:",df.shape)

    print("Dropped Columns:\n",Dropped_cols)

drop_unique_cols(telecom_High_value)
# Checking unique values, null percentage, variable type etc for each column

get_meta_data(telecom_High_value)
#Coulmns having more than 40% missing values can be dropped

print("DF Shape before Dropping columns:",telecom_High_value.shape)

telecom_High_value = telecom_High_value.loc[:, telecom_High_value.isnull().mean() <0.4]

print("DF Shape after Dropping columns:",telecom_High_value.shape)
# Checking unique values, null percentage, variable type etc for each column

get_meta_data(telecom_High_value)
#We can also note that columns for onnet, ic and og etc have the same missing value numbers. This suggests that the customers

#did not use the services. We can umpute these columns with 0



#Also we can see that date of recharge columns are having missing values. Since there would be a lot of customers who

#do not recharge every month, but take multiple month package, we can assume that these customers did not recharge in the 

#particular month. Hence these values could also be imputed with 0



# since only above mentioned columns  are now having missing values in the data, we can replace missing values in all coulmns 

#with 0

telecom_High_value.replace(np.NaN,0.0,inplace=True)

# Checking unique values, null percentage, variable type etc for each column

get_meta_data(telecom_High_value)
# We can create tenure variable from the aon variable in data. Aon is age on network in days, we can create tenure on network

#in months

telecom_High_value['tenure']=telecom_High_value['aon']/30



#We can also create new variables. Since _8 variables are not used, 

#we can use them and see the difference between 6,7 months and 8 month



# We can calculate average values of _6 and _7 variables and then calculate the difference of the average with _8 variables

telecom_High_value['max_rech_amt_8_67_diff'] =((telecom_High_value.max_rech_amt_6 + telecom_High_value.max_rech_amt_7)/2)-telecom_High_value.max_rech_amt_8



telecom_High_value['total_rech_data_8_67_diff'] = ((telecom_High_value.total_rech_data_6 + telecom_High_value.total_rech_data_7)/2)-telecom_High_value.total_rech_data_8

telecom_High_value['max_rech_data_8_67_diff'] = ((telecom_High_value.max_rech_data_6 + telecom_High_value.max_rech_data_7)/2)-telecom_High_value.max_rech_data_8 



telecom_High_value['av_rech_amt_data_8_67_diff'] =((telecom_High_value.av_rech_amt_data_6 + telecom_High_value.av_rech_amt_data_7)/2)-telecom_High_value.av_rech_amt_data_8
# We can select the recharge columns that we need to see with respect to churn

recharge_columns =  telecom_High_value.columns[telecom_High_value.columns.str.contains('rech_amt')]

recharge_columns.tolist()
##lets check the boxplot with respect to Target Variable

i,j,k=5,3,1

plt.figure(figsize=[20,40])

for col in recharge_columns:

    plt.subplot(i,j,k)

    sns.boxplot(data=telecom_High_value, y=col,x="churn",

                showfliers=False)

    k+=1

plt.show()
plot_box_chart('offnet_mou',telecom_High_value)
plot_box_chart('onnet_mou',telecom_High_value)
sns.boxplot(data=telecom_High_value, y='tenure',x="churn", showfliers=False)
plot_box_chart('max_rech_amt',telecom_High_value)
plot_box_chart('max_rech_data',telecom_High_value)
plot_box_chart('arpu',telecom_High_value)
#checking distribution of variables

plt.figure(figsize=[15,5])

plt.subplot(1,3,1)

sns.distplot(telecom_High_value['max_rech_amt_8_67_diff'])

plt.subplot(1,3,2)

sns.distplot(telecom_High_value['total_rech_data_8_67_diff'])

plt.subplot(1,3,3)

sns.distplot(telecom_High_value['tenure'])
#We can remove the date columns

telecom_High_value.drop(['date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8'],axis=1,inplace=True)
for col in telecom_High_value.columns:

    if telecom_High_value[col].dtype in ['int64','float64']:

            percentiles=telecom_High_value[col].quantile([0.01,0.99]).values

            telecom_High_value[col][telecom_High_value[col] <= percentiles[0]] = percentiles[0]

            telecom_High_value[col][telecom_High_value[col] >= percentiles[1]] = percentiles[1]

telecom_High_value.describe([.01,.99])
#Checking Correlation

telecom_High_value.corr()
#Splitting Data into X and Y sets

y = telecom_High_value["churn"].astype('int64')

X = telecom_High_value.drop(["churn","mobile_number"],axis=1)



# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

print(X_train.shape,X_test.shape)
from sklearn.decomposition import PCA

pca=PCA(random_state=42)

pca.fit(X_train)



#Plotting Cumulative sum of explained variance

var_cumu=np.cumsum(pca.explained_variance_ratio_)



fig=plt.figure(figsize=[12,8])

plt.plot(var_cumu)

plt.show()
from sklearn.decomposition import IncrementalPCA

PCA_final=IncrementalPCA(n_components=20)

df_train_pca=PCA_final.fit_transform(X_train)

X_train_pca=pd.DataFrame(df_train_pca)

df_test_pca=PCA_final.transform(X_test)

X_test_pca=pd.DataFrame(df_test_pca)
from sklearn.linear_model import LogisticRegression

learner_pca=LogisticRegression(class_weight='balanced')

model_pca=learner_pca.fit(X_train_pca,y_train)



y_pred=model_pca.predict_proba(X_train_pca)

y_train_pred=pd.DataFrame(y_pred[:,1])

y_train_pred
#Checking the AUC score

metrics.roc_auc_score(y_train, y_train_pred)
#Checking ROC_AUC curve

def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None



draw_roc(y_train,y_train_pred)
y_train_pred.columns=['Churn_Prob']

y_train_pred.head()
# Creating columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred[i]= y_train_pred.Churn_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred.head()
# Calculating accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train, y_train_pred[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
y_train_pred['Predicted'] = y_train_pred.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred.head()
# Let's take a look at the confusion matrix

confusion = metrics.confusion_matrix(y_train,y_train_pred.Predicted)

confusion
#checking accuracy.

metrics.accuracy_score(y_train,y_train_pred.Predicted)
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives

print('Senstivity',TP / float(TP+FN))

print('Specificity',TN / float(TN+FP))

print('False postive rate',FP/ float(TN+FP))

print ('Negative predictive value',TN / float(TN+ FN))

print('Precision',confusion[1,1]/(confusion[0,1]+confusion[1,1]))

print('Recall',confusion[1,1]/(confusion[1,0]+confusion[1,1]))
#Predicting On Test Set

y_pred=model_pca.predict_proba(X_test_pca)

y_test_pred=pd.DataFrame(y_pred[:,1])

y_test_pred.columns=['Churn_Prob']



y_test_pred['Predicted'] = y_test_pred.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_test_pred.head()
# Let's take a look at the confusion matrix

confusion = metrics.confusion_matrix(y_test,y_test_pred.Predicted)

confusion
#checking accuracy.

metrics.accuracy_score(y_test,y_test_pred.Predicted)
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives

print('Senstivity',TP / float(TP+FN))

print('Specificity',TN / float(TN+FP))

print('False postive rate',FP/ float(TN+FP))

print ('Negative predictive value',TN / float(TN+ FN))

print('Precision',confusion[1,1]/(confusion[0,1]+confusion[1,1]))

print('Recall',confusion[1,1]/(confusion[1,0]+confusion[1,1]))
# lets create a decision tree now



# import decision tree libraries

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

from sklearn.model_selection import GridSearchCV



dt = DecisionTreeClassifier(random_state=42)



params = {'max_depth': [2,3,5,10, 20], 'min_samples_leaf':[5,10,20,50,100],'criterion':['gini','entropy'] }



grid_search=GridSearchCV(estimator=dt, param_grid=params,cv=4,n_jobs=-1,verbose=1,scoring='accuracy')



# lets create a decision tree with the default hyper parameters except max depth to make the tree readable



grid_search.fit(X_train, y_train)
cv_df=grid_search.cv_results_

cv_df
dt_best=grid_search.best_estimator_

dt_best
from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_model(dt_classifier):

    y_train_pred=dt_classifier.predict(X_train)

    y_test_pred=dt_classifier.predict(X_test)

    print("Train set performance")

    print(accuracy_score(y_train,y_train_pred))

    print(confusion_matrix(y_train,y_train_pred))

    print("."*50)

    print("Test set performance")

    print(accuracy_score(y_test,y_test_pred))

    print(confusion_matrix(y_test,y_test_pred))

evaluate_model(dt_best)
# Importing required packages for visualization

from sklearn.tree import export_graphviz

import graphviz



data=export_graphviz(dt_best, out_file=None, filled=True, rounded=True,

                feature_names=X.columns, 

                class_names=['No Churn', "Churn"])

graph = graphviz.Source(data)

graph

from sklearn.ensemble import RandomForestClassifier



# GridSearchCV to find optimal n_estimators

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



# parameters to build the model on

params = {'max_depth': [2,3,5,10,20], 'min_samples_leaf':[5,10,20,50,100],'criterion':['gini','entropy'] }



# random forest - the class weight is used to handle class imbalance - it adjusts the cost function

rf = RandomForestClassifier(class_weight='balanced')



grid_search=GridSearchCV(estimator=rf, param_grid=params,cv=4,n_jobs=-1,verbose=1,scoring='accuracy')



# fit model

grid_search.fit(X_train, y_train)
cv_rf=grid_search.cv_results_

cv_rf
rf_best=grid_search.best_estimator_

rf_best
evaluate_model(rf_best)
sample_tree = rf_best.estimators_[0]
data=export_graphviz(sample_tree, out_file=None, filled=True, rounded=True,

                feature_names=X.columns, 

                class_names=['No Churn', "Churn"])

graph = graphviz.Source(data)

graph
imp_df = pd.DataFrame({

    "Varname": X_train.columns,

    "Imp": rf_best.feature_importances_

})

imp_df.sort_values(by="Imp", ascending=False)
# GridSearchCV to find optimal n_estimators

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

#Importing XGBClassifier

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, accuracy_score



# parameters to build the model on

params = {'learning_rate': [0.1,0.2,0.3], 'subsample': [0.3,0.4,0.5]}



# XGB - the class weight is used to handle class imbalance - it adjusts the cost function

xgb = XGBClassifier(class_weight='balanced',max_depth=2, n_estimators=200)



grid_search=GridSearchCV(estimator=xgb, param_grid=params,cv=5,n_jobs=-1,verbose=1,scoring='accuracy',return_train_score=True)



# fit model

grid_search.fit(X_train, y_train)



#best Estimator

xgb_best=grid_search.best_estimator_

xgb_best



#Evaluate Model

evaluate_model(xgb_best)
#Feature importance

imp_df = pd.DataFrame({

    "Varname": X_train.columns,

    "Imp": xgb_best.feature_importances_

})

imp_df.sort_values(by="Imp", ascending=False)