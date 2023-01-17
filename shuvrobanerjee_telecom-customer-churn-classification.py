# Installing imbalanced-learn library for usage in the dataset -For us,it was not included in Jupyter notebook,if it's present in 
# any other environment following command will check its' installation and update if a new version is available
#!pip install imbalanced-learn
# Importing Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import re

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
# We will use this function for generic plotting purpose during EDA
def genericPlotter(df,data,plot,string):
    plt.figure(figsize=(8,4))
    if plot == 'box':
        sns.boxplot(y=df[data])
        plt.title("Distribution Status of "+ string)
    if plot == 'bar':
        data.plot(kind ='bar')
        plt.ylabel('Count')
        plt.xlabel(string)
        plt.title("Distribution Status of "+ string)
    if plot =='misc':
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        sns.distplot(df[data],kde_kws={'bw': 1.5})
        plt.subplot(132)
        sns.boxplot(y = df[data])
        plt.subplot(133)
        plt.hist(df[data])
        if(string == 'log'):
            plt.yscale('log')
        else:
            plt.title("Distribution Status of "+ string)
            
    plt.show()
# We will use this function for outlier treatment for this analysis
# Refer :https://medium.com/@prashant.nair2050/hands-on-outlier-detection-and-treatment-in-python-using-1-5-iqr-rule-f9ff1961a414
def outlierImputter(data,col_name,exp):
    IQR = data[col_name].quantile(0.75) - data[col_name].quantile(0.25)
    step = exp * IQR
    lowerbound = 0 if (data[col_name].quantile(0.25) - step) < 0 else (data[col_name].quantile(0.25) - step)
    upperbound = data[col_name].quantile(0.75) + step
    return lowerbound,upperbound
# Plot function for specific purpose -Segmented Univariate Analysis
def segmentedUnivariatePlotter(df,X_col,y_col,title_string,ylabel):
    plt.suptitle('Segmented Univariate Analysis (' + X_col + ')', fontsize=24)
    plt.rc("font", size=10)
    plt.figure(figsize=(20, 10))
    sns.barplot(x=df.index, y=y_col, data=df)
    plt.title(title_string)
    plt.xlabel(X_col,rotation=0, fontsize=20, labelpad=20)
    plt.xticks(rotation=45,fontsize=10)
    plt.ylabel(ylabel,rotation=90, fontsize=20, labelpad=20)
    plt.show()
    return
def binPlotter(df,column,new_bin_col_name,y_col_name,binval,showsize=10):
        df[new_bin_col_name] = pd.cut(df[column], binval)
        df_num_column_status=df.pivot_table(values=[y_col_name],index=[new_bin_col_name],aggfunc='mean')
        df_num_column_status.head(showsize)
        return df_num_column_status
def groupedBarChartPlotter(df_segment,label,showsize=10):
        colors = ['green', 'red']
        ax=df_segment.plot(kind='bar',figsize=(15,10),color=colors)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Churn", fontsize=12)
        plt.show()
        return
def binCreater(column,new_bin_col_name,binval):
    df[new_bin_col_name] = pd.cut(df[column], binval)
    return df
def CVPlotter(df,param,param_string):
    plt.figure(figsize=(10,5))
    plt.plot(df[param], 
             df["mean_train_score"], 
             label="training accuracy")
    plt.plot(df[param], 
             df["mean_test_score"], 
             label="test accuracy")
    plt.xlabel("min_samples_"+param_string)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
master_df =pd.read_csv('../input/telecom-churn-data/telecom_churn_data.csv',encoding='ISO-8859-1')
master_df.head(25)
master_df.shape
master_df.info()
master_df.describe(percentiles =[0.05,0.10,0.25,0.50,0.75,0.99])
# Checking for presence of null values in the dataset
# Let's have a look at missing values - how many are there & also get their percencentages
pd.options.display.float_format = '{:.2f}'.format
missing_values = pd.DataFrame((master_df.isnull().sum()/len(master_df))*100).reset_index().rename(columns = {'index': 'feature', 0: 'Missing%'}).sort_values('Missing%',ascending = False)
#pd.set_option('display.max_rows',100)
missing_values
delete_columns =[]
for i in master_df.columns:
    if master_df[i].nunique() == 1:
        delete_columns.append(i)
print(delete_columns)
master_df=master_df.drop(delete_columns,axis=1)
master_df.shape
# Filter on main dataset for finding columns having churn phase information
churn_phase_data =master_df.filter(regex='_9$', axis=1)
churn_phase_data.head()
# Having a closer look onto the churn_phase_columns
churn_phase_columns =churn_phase_data.columns
churn_phase_columns
# Creating a seperate dataframe to track churn phase information:
churn_predictor_columns =['total_ic_mou_9','total_og_mou_9','vol_2g_mb_9','vol_3g_mb_9']
churn_phase_data.loc[:,churn_predictor_columns].head()
# Checking for missing values in churn_data
round((churn_phase_data.loc[:,churn_predictor_columns].isnull().sum()/len(churn_phase_data.index))*100,2)
#Finding Out Actual Churners from Churn Phase -Creating a new dataset
churners =master_df.loc[(churn_phase_data.total_ic_mou_9 == 0.00) 
                            & (churn_phase_data.total_og_mou_9 == 0.00)
                            & (churn_phase_data.vol_2g_mb_9 == 0.00)
                            & (churn_phase_data.vol_3g_mb_9 == 0.00), ['mobile_number']
                           ]
# Adding a column to depict actual churners in churn_phase
churners['churn'] =1

# Displaying churners dataframe
churners.head()
# Merging Master_df and churners df 
master_df =pd.merge(master_df,churners,how ='outer',on ='mobile_number')
master_df['churn'] =master_df['churn'].fillna(0)
master_df['churn'] =master_df['churn'].astype('int')
master_df.head()
# Dropping Churn Phase Data from Master_df so that it contains only Good and Action Phase Data for model building and training
master_df=master_df.drop(master_df.loc[:,churn_phase_columns],axis=1)
# master_df -after removing churn phase information
master_df.head()
# Creating Derived Columns to find out high valued customer based on call recharge amount and data recharge amount
# of good phase,i.e.,month 6 and 7
master_df['Average_recharge_amnt_call_6-7']=round(master_df['total_rech_amt_6']+master_df['total_rech_amt_7']/2,2)
master_df['Average_recharge_amnt_data_6-7']=round(master_df['av_rech_amt_data_6']+master_df['av_rech_amt_data_7']/2,2)
# Let's create one derived variable to find out avearge user expenditure in good phase
master_df['Good_Phase_Avg_Spend']=master_df['Average_recharge_amnt_call_6-7']+master_df['Average_recharge_amnt_data_6-7']
# Creating Filter for HVC :
X_Percentile =0.70
target_X_for_calls =master_df['Average_recharge_amnt_call_6-7'].quantile(X_Percentile)
target_X_for_data =master_df['Average_recharge_amnt_data_6-7'].quantile(X_Percentile)

print('For HVC : Cut-off Amount for Calls:',target_X_for_calls)
print('For HVC : Cut-off Amount for Data:',target_X_for_data)
# Applying HVC filter in master dataset
master_df =master_df.loc[(master_df['Average_recharge_amnt_call_6-7'] >=target_X_for_calls) | (master_df['Average_recharge_amnt_data_6-7'] >=target_X_for_data) ]
master_df
# Deleting derived columns from dataset as filter is already applied
cols_to_drop=['Average_recharge_amnt_call_6-7','Average_recharge_amnt_data_6-7']
master_df=master_df.drop(cols_to_drop,axis=1)
#ARPU is a key column to find out user behaviour.Let's see if we have any negative value here:
master_df.loc[(master_df.arpu_6 < 0) & (master_df.arpu_7 < 0) & (master_df.arpu_8 <0)]
master_df =master_df.loc[(master_df.arpu_6 > 0) & (master_df.arpu_7 > 0) & (master_df.arpu_8 >0)]
master_df.corr()

cols_to_impute_nightpack =['night_pck_user_6','night_pck_user_7','night_pck_user_8']
cols_to_impute_fb_user= ['fb_user_6','fb_user_7','fb_user_8']

#majority of the data distribution for night pack users is zero hence we impute it with zero
for el in cols_to_impute_nightpack:
    master_df.loc[:,el].fillna(0).astype(int)

#majority of the data distribution for fb users is one hence we impute it with one
for el in cols_to_impute_fb_user:
    master_df.loc[:,el].fillna(1).astype(int)    

master_df.date_of_last_rech_data_6.fillna('6/30/2014',inplace=True)
master_df.date_of_last_rech_6.fillna('6/30/2014',inplace=True)
master_df.date_of_last_rech_data_7.fillna('7/31/2014',inplace=True)
master_df.date_of_last_rech_7.fillna('7/31/2014',inplace=True)
master_df.date_of_last_rech_data_8.fillna('8/31/2014',inplace=True)
master_df.date_of_last_rech_8.fillna('8/31/2014',inplace=True)
cols_to_rename ={'jun_vbc_3g':'vbc_3g_6','jul_vbc_3g':'vbc_3g_7','aug_vbc_3g':'vbc_3g_8','sep_vbc_3g':'vbc_3g_9'}

master_df.rename(columns =cols_to_rename,inplace=True)
# Dropping mobile number
master_df=master_df.drop('mobile_number',axis=1)
master_df
# Removing Nan/Null values from the rest of dataset
master_df.isnull().sum()

master_df=master_df.fillna(master_df.median())
master_df.describe()
# Checking for Any Null Values Presence Now
master_df.isnull().any().sum()
# data_density is defined as percentage of non zero values in each column
data_density = master_df.astype(bool).sum(axis=0)/master_df.shape[0] * 100
cols_to_drop = []
for el in master_df.columns:
    if data_density[el] < 5:
        cols_to_drop.append(el)
print(cols_to_drop)
cols_to_drop.extend(['og_others_6'])
master_df=master_df.drop(cols_to_drop, axis=1)
master_df.head()

master_df.shape

master_df["duration_of_last_rech_7-6"] = (pd.to_datetime(master_df.date_of_last_rech_7) - pd.to_datetime(master_df.date_of_last_rech_6)).dt.days
master_df["duration_of_last_rech_8-7"] = (pd.to_datetime(master_df.date_of_last_rech_8) - pd.to_datetime(master_df.date_of_last_rech_7)).dt.days

master_df["duration_of_last_rech_data_7-6"] = (pd.to_datetime(master_df.date_of_last_rech_data_7) - pd.to_datetime(master_df.date_of_last_rech_data_6)).dt.days
master_df["duration_of_last_rech_data_8-7"] = (pd.to_datetime(master_df.date_of_last_rech_data_8) - pd.to_datetime(master_df.date_of_last_rech_data_7)).dt.days
# list to contain all date columns which are going to be deleted.
cols_to_drop = []
for col in master_df.columns:
    if 'date' in col:
        cols_to_drop.append(col)

print ("These Date Columns will be Removed from Dataset:",cols_to_drop)        
master_df=master_df.drop(cols_to_drop, axis=1) 
# Checking for Any Null Values Presence Now
master_df=master_df.fillna(master_df.median())

master_df.isnull().any().sum()
master_df.shape
cols = set()
# get all the feature names (without the months in the end)
for col in master_df.columns:
    # only for columns that end with _<month>, create derived variables
    if bool(re.match(r".*_([6-8])$", col)) == True:
        cols.add(re.split(r"_([6-8])", col)[0])
new_cols = set()

# for all the features create two separate columns - the difference between month 6 and 7, and 7 and 8.
for col in cols:
    master_df[col + "_6-7"] = master_df[col + "_7"] - master_df[col + "_6"]
    master_df[col + "_7-8"] = master_df[col + "_8"] - master_df[col + "_7"]
    new_cols.add(col + "_6-7")
    new_cols.add(col + "_7-8")

print('All columns in the master dataframe')
print(list(master_df.columns))
print('\n')
print('Total count of columns')
print(len(list(master_df.columns)))
print('\n')
print('New columns that were added')
print(list(new_cols))
master_df['total_data_6']= master_df.vol_2g_mb_6 + master_df.vol_3g_mb_6
master_df['total_data_7']= master_df.vol_2g_mb_7 + master_df.vol_3g_mb_7
master_df['total_data_8']= master_df.vol_2g_mb_8 + master_df.vol_3g_mb_8
# average revenue per user in a single column and split by average vs first two months average 
master_df['arpu_2mths'] = (master_df['arpu_6']+master_df['arpu_7'])/2
master_df['arpu_3mths'] = (master_df['arpu_6']+master_df['arpu_7']+master_df['arpu_8'])/3
# average revenue per user in a single column and split by average vs first two months average 
master_df['avg_tot_rech_2mths'] = (master_df['total_rech_amt_6']+master_df['total_rech_amt_7'])/2
master_df['avg_tot_rech_3mths'] = (master_df['total_rech_amt_6']+master_df['total_rech_amt_7']+master_df['total_rech_amt_8'])/3
# Conversion of the number of months by 12
master_df['aon_years']=master_df['aon']/365
master_df['aon_years'] = master_df['aon_years'].astype(int)
# We can drop the original column as it will be no use going forward
master_df=master_df.drop('aon',axis=1)
# Let's check the shape of the cleaned dataset
master_df.shape
# Copying the dataset
hvc_df =master_df.copy()
hvc_df.dtypes[hvc_df.dtypes == 'object']
hvc_df.dtypes[hvc_df.dtypes == 'int']
hvc_df.dtypes[hvc_df.dtypes == 'float']
# Let's see the description and values of the dataset to find out trends.
hvc_df.describe(percentiles =[0.05,0.10,0.25,0.50,0.75,0.99])

#Plotting distribution of continuous columns
for col in hvc_df._get_numeric_data():
    genericPlotter(hvc_df,col,'box',col)


# Keeping the dependent 'churn' column out of consideration for now
cols_to_consider = hvc_df.columns.drop('churn')

for col in cols_to_consider:
    ranges = outlierImputter(hvc_df, col, 4)
    hvc_df[col][(hvc_df[col]<ranges[0])] = np.nan
    hvc_df[col][(hvc_df[col]>ranges[1])] = np.nan

print(hvc_df.isnull().sum(axis=1).nlargest(30))
# delete rows with more than 100 outlier values from non churn data points
hvc_df['No_of_Outliers'] = hvc_df.isnull().sum(axis=1)
hvc_df.drop(hvc_df[(hvc_df.No_of_Outliers > 100) & (hvc_df.churn == False)].index, inplace = True)
hvc_df.drop('No_of_Outliers', axis = 1, inplace = True)
hvc_df = hvc_df.reset_index(drop=True)
hvc_df.shape
# Imputing with median
for col in cols_to_consider:
    hvc_df[col].fillna((hvc_df[col].median()), inplace=True)

hvc_df
cols_to_drop =[]
for el in hvc_df.columns:
    if hvc_df[el].sum() == 0:
        cols_to_drop.append(el)       
print(cols_to_drop)
hvc_df=hvc_df.drop(cols_to_drop, axis=1)
hvc_df.shape 
#After removal of outlier,let's see the distribution of the continuous columns:
# Distribution as histograms
cols_for_EDA =hvc_df.columns.drop('churn')
for col in cols_for_EDA:
    genericPlotter(hvc_df,col,'misc',col)
plot_df =binPlotter(hvc_df,'aon_years','aon_group','arpu_2mths',10).sort_values(by='arpu_2mths',ascending=True)
print(plot_df)
segmentedUnivariatePlotter(plot_df,'aon_years','arpu_2mths','Age on Network vs Average Revenue','Ageon Network to Average Revenue')
plot_df =binPlotter(hvc_df,'aon_years','aon_group','arpu_3mths',10).sort_values(by='arpu_3mths',ascending=True)
print(plot_df)
segmentedUnivariatePlotter(plot_df,'aon_years','arpu_3mths','Age on Network vs Average Revenue','Ageon Network to Average Revenue')
correlation_data= hvc_df.corr()
correlation_data
correlation_df  = correlation_data.stack().reset_index().sort_values(by = 0, ascending = False)
correlation_df[((correlation_df[0] < 1) & (correlation_df[0] >= 0.4)) | ((correlation_df[0] <= -0.4) & (correlation_df[0] > -1))]
# Find features with correlation greater than 0.90
correlation_data= hvc_df.corr().abs()
# Select upper triangle of correlation matrix
upper = correlation_data.where(np.triu(np.ones(correlation_data.shape), k=1).astype(np.bool))
cols_to_drop = [column for column in correlation_data.columns if any(upper [column] > 0.90)]
print ("Columns to Drop Due to High Collinearity :" ,cols_to_drop)
# Deleting Columns 
hvc_df.drop(cols_to_drop,axis=1,inplace=True)
# Checking shape
hvc_df.shape
churn_count =hvc_df['churn'].value_counts()
churn_count
# Plotting Churn Distribution 
genericPlotter(hvc_df,churn_count,'bar','Churn')
# Copying hvc_df data to a new dataframe for model buildfing purpose
model_builder_df =hvc_df.copy()
model_builder_df =model_builder_df._get_numeric_data()
X =model_builder_df.drop('churn',axis=1)
y =model_builder_df['churn']
model_builder_df.drop('churn',axis=1,inplace=True)
# Apply Standard Scaler
scaler =preprocessing.StandardScaler().fit(X)
X =scaler.transform(X)
X
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size =0.3,train_size =0.7,random_state=1)
print ("Dimension of X_train :",X_train.shape)
print ("Dimension of X_test:",X_test.shape)
balancer =SMOTE(random_state=42)
X_balanced_train,y_balanced_train =balancer.fit_sample(X_train,y_train)
print("Dimension of X_balanced_train:",X_balanced_train.shape)
print("Dimension of y_balaned_train:",y_balanced_train.shape)
print("Checking for Imbalance in SMOTE transformed Training Set for Churn:",(y_balanced_train!=0).sum()/(y_balanced_train ==0).sum())
y_balanced_train.shape
# Selecting top 15 features according to RFE on the balanced dataset

logistic_model= LogisticRegression(random_state=1)
rfe =RFE(logistic_model,15)
rfe =rfe.fit(X_balanced_train,y_balanced_train)
rfe_selected_features =list(model_builder_df.columns[rfe.support_])
print("According to RFE 10 Most Important Features Are:",rfe_selected_features)
# Creating X and y for logistic regression and fitting model
X_rfe =pd.DataFrame(data=X_balanced_train).iloc[:,rfe.support_]
y_rfe =y_balanced_train
logistic_model =LogisticRegression(random_state=1)
logistic_model.fit(X_rfe,y_rfe)
# Applying model on test data
X_test_rfe =pd.DataFrame(data=X_test).iloc[:,rfe.support_]
y_pred =logistic_model.predict(X_test_rfe)
confusion_matrix =confusion_matrix(y_test,y_pred)
print("Confusion Matrix for Logistic Regression:",confusion_matrix)
print("Test Score Accuracy on Test Dataset:",logistic_model.score(X_test_rfe,y_test))
pca =PCA(random_state=100)
# Applying PCA on the balanced training data
#pca.fit(X_balanced_train)
X_balanced_train_pca =pca.fit_transform(X_balanced_train)
X_test_pca =pca.transform(X_test)
print ("Checking the Shape of X train post PCA tranformation",X_balanced_train_pca)
print ("Checking the Shape of X test post PCA transformation",X_test_pca)
logistic_model_pca =LogisticRegression(C=1e9)
logistic_model_pca.fit(X_balanced_train_pca,y_balanced_train)
# Making Predictions
y_pred_pca =logistic_model_pca.predict(X_test_pca)
y_pred_pca_df =pd.DataFrame(y_pred_pca)
y_pred_pca_df
y_test
# Printing Confusion Matrix
from sklearn.metrics import confusion_matrix
print ("Confusion Matrix After Using PCA in Logistic Regression",confusion_matrix(y_test,y_pred_pca))
print ("Accuracy of the Logistic Regression Model After Using PCA",accuracy_score(y_test,y_pred_pca))
cols_to_consider =model_builder_df.columns
pca_df =pd.DataFrame({'PC-1':pca.components_[0],'PC-2':pca.components_[1],'PC-3':pca.components_[2],'Feature':col})
pca_df.head()
# Let's check the no. of principal components responsible for higest percentage of variance
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
# Scree Plot to check variance described by principal components
plt.figure(figsize=(12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("No. of Principal Components")
plt.ylabel("Explained Variance -Cumulative")
plt.show()
selective_pca =PCA(n_components =31)
X_train_selective_pca =selective_pca.fit_transform(X_balanced_train)
X_test_selective_pca=selective_pca.transform(X_test)
print("Shape of the New PCA Generated Training Set for X:",X_train_selective_pca.shape)
print("Shape of New PCA Generated Test Set for X",X_test_selective_pca.shape)
# Fitting Model
logistic_model_selective_pca =LogisticRegression(C=1e9)
logistic_model_selective_pca.fit(X_train_selective_pca,y_balanced_train)

# Applying Model
y_pred_31_components =logistic_model_selective_pca.predict(X_test_selective_pca)

obj=confusion_matrix(y_test,y_pred_31_components)
# Confusion Matrix
print ("Confusion Matrix:",confusion_matrix(y_test,y_pred_31_components))
# Accuracy Score
print("Accuracy Score of Logistic Regression with 31 components:",accuracy_score(y_test,y_pred_31_components))
svc =LinearSVC(C=0.001,penalty ="l1",dual=False).fit(X_balanced_train,y_balanced_train)
svc_model =SelectFromModel(svc,prefit=True)
X_lasso =svc_model.transform(X_balanced_train)
position =svc_model.get_support(indices =True)

print("Lasso Optimized X train Dimension:",X_lasso.shape)
# Feature Selection for Decision Tree
selected_features =list(model_builder_df.columns[position])
print("Selected Features via Lasso for Decision Tree:",selected_features)
dTree =DecisionTreeClassifier(max_depth=5)
dTree.fit(X_lasso,y_balanced_train)
X_test_tree =pd.DataFrame(data=X_test).iloc[:,position]
y_pred_tree=dTree.predict(X_test_tree)

#Classification Report
print("Classification Report of the Model",classification_report(y_test,y_pred_tree))
print("****************************************")
# Confusion Matrix
#print ("Confusion Matrix for the model",confusion_matrix(y_test,y_pred_tree))
print("****************************************")
print ("Accuracy Score for the model",accuracy_score(y_test,y_pred_tree))

n_folds =5
parameters ={'min_samples_leaf':range(n_folds,200,20)}
dTree2 =DecisionTreeClassifier(criterion ='gini',random_state =100)
CV_tree =GridSearchCV(dTree2,parameters,cv =n_folds,scoring='accuracy',return_train_score=True)
CV_tree.fit(X_lasso,y_balanced_train)
my_score =CV_tree.cv_results_
pd.DataFrame(my_score).head()
# Plotting Accuracies 
# plotting accuracies with min_sample_leaf
print("Test and Train Accuracy of min_samples_leaf\n",CVPlotter(my_score,"param_min_samples_leaf","leaf"))
parameters ={'min_samples_split':range(n_folds,200,20)}
dTree3 =DecisionTreeClassifier(criterion='gini',random_state=100)

# Fit tree on training data
CV_tree =GridSearchCV(dTree3,parameters,cv=n_folds,scoring='accuracy',return_train_score=True)
CV_tree.fit(X_lasso,y_balanced_train)
# Publishing Score
score =CV_tree.cv_results_
pd.DataFrame(score).head()
# Plotting Accuracies 
# plotting accuracies with min_sample_leaf
CVPlotter(score,"param_min_samples_split","split")
parameters ={'max_depth':range(5,15,5),
             'min_samples_leaf':range(25,100,30),
             'min_samples_split':range(45,100,40),
             'criterion':["entropy","gini"]
            }

dTree4 =DecisionTreeClassifier()
CV_tree =GridSearchCV(estimator=dTree4,param_grid=parameters,cv=n_folds,verbose =1)

# Fitting Grid Search
CV_tree.fit(X_lasso,y_balanced_train)
cv_results =pd.DataFrame(CV_tree.cv_results_)
cv_results
# printing the optimal accuracy score and hyperparameters
print("Best Accuracy", CV_tree.best_score_)
print ("Best Estimator",CV_tree.best_estimator_)
# Creating Model with these estimators
my_final_tree =DecisionTreeClassifier(criterion ='gini',
                                      random_state=100,
                                      max_depth =5,
                                      min_samples_leaf=25,
                                      min_samples_split=45)
my_final_tree.fit(X_lasso,y_balanced_train)
y_pred_tree =my_final_tree.predict(X_test_tree)
#Classification Report
print("Classification Report of the Model",classification_report(y_test,y_pred_tree))
print("****************************************")
# Confusion Matrix
#print ("Confusion Matrix for the model",confusion_matrix(y_test,y_pred_tree))
print("****************************************")
print ("Accuracy Score for the model",accuracy_score(y_test,y_pred_tree))

# Fitting Model
rf_model =RandomForestClassifier()
rf_model.fit(X_lasso,y_balanced_train)

#Making predictions
# Generting test dataset from lasso selected features
X_test_rf =pd.DataFrame(data=X_test).iloc[:,position]
y_test_rf =rf_model.predict(X_test_rf)
#Classification Report
print("Classification Report of the Model",classification_report(y_test,y_test_rf))
print("****************************************")
# Confusion Matrix
#print ("Confusion Matrix for the model",confusion_matrix(y_test,y_test_rf))
print("****************************************")
print ("Accuracy Score for the model",accuracy_score(y_test,y_test_rf))
X =model_builder_df
features =X.columns.values
X = pd.DataFrame(scaler.transform(X))
X.columns =features
importances =rf_model.feature_importances_
X_values = []
for el in position:
    X_values.append(X.columns[el])
weights = pd.Series(importances,index=X_values)
weights.sort_values()[-10:].plot(kind='barh')