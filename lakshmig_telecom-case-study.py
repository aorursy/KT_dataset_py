# Import the Necessary Libraries

import numpy as np
import pandas as pd

# Import Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import the logistic regression 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Import the scaler, KMeans etc.,
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
import os

# Supress the warnings
import warnings
warnings.filterwarnings('ignore')

# Read the Csv file & Read the few coloumns and rows of the file. 

Churn = pd.read_csv('Downloads//telecom_churn_data.csv')
Churn.head()
# Let us check the data shape
Churn.shape
# We have 99999 rows and 226 coloumns. 
# Let us check then datatype 
Churn.info(verbose=1)
# Let us check the statistics 
Churn.describe()
# Let us see the NAN values % in ascending order wise
round(Churn.isnull().mean(axis=0).sort_values(ascending=False)*100,2)
# Lot of values is having more than 70% of the data null. We need to check the unique values first and deal it one by one.
# Let us check the unique values in sorting order and check the data / value first.
Churn.nunique().sort_values()
## Missing values percentage of all the coloumns in sorting fashion
round(Churn.isnull().mean(axis=0).sort_values(ascending=False)*100,2)
# As We see more than 40% of varaibles are needed for solving the business problem. We need to imute it rather than drop it.
# Let us make a custom function for the data cleaning part.
# Create a funciton and check the % values and print the missing nos. 
def checknull(per_cutoff):
    missing = round(100*(Churn.isnull().sum()/Churn.shape[0]))
    print("There are {} features having more than {}% missing value".format(len(missing.loc[missing > per_cutoff]),per_cutoff))
    return missing.loc[missing > per_cutoff]
# Let us also create the Imute functions so that we can imute it one by one. 
def imputenan(data,imputeColList=False,missingColList=False):
    # Function impute the nan with 0
    # argument: colList, list of columns for which nan is to be replaced with 0
    if imputeColList:
        for column in [x + y for y in ['_6','_7','_8','_9'] for x in imputeColList]:
            data[column].fillna(0, inplace=True)
    else:    
        for column in missingColList:
            data[column].fillna(0, inplace=True)
# Check the missing values again
checknull(70)
# We can see the KPI's are also there in the above sets. we need to imute it rather than drop it. Let us impute it with the zero.
imputeCol = ['av_rech_amt_data', 'arpu_2g', 'arpu_3g', 'count_rech_2g', 'count_rech_3g',
             'max_rech_data', 'total_rech_data','fb_user','night_pck_user']
imputenan(Churn,imputeCol)

# Check the data again now.
checknull(70)
# dropping the columns having more than 50% missing values
missingcol = list(checknull(70).index)
Churn.drop(missingcol,axis=1,inplace=True)
Churn.shape

# Check the head again
Churn.head()
# the Circle id is also of no use having a unique value in this 
Chrun = Churn.drop('circle_id',axis=1,inplace = True)
Churn.head()
# check again the missing values back in the data sets.
checknull(5)
# We can see still there are 29 features having more than missing values. let us check it first.
missingcol = list(checknull(5).index)
print ("There are %d customers missing values for %s"%(len(Churn[Churn[missingcol].isnull().all(axis=1)]),missingcol))
Churn[Churn[missingcol].isnull().all(axis=1)][missingcol].head()
# Let us immute it with zero of these customers as we have huge nos. 7745. 
imputenan(Churn,missingColList=missingcol)
# Check the missing null. 
Churn=Churn[~Churn[missingcol].isnull().all(axis=1)]
Churn.shape
# Let us check the data set again with more than 2% nan values
checknull(2)
# WE could see still there are 89 features are there. Let us check it first 
missingcol = list(checknull(2).index)
print ("There are %d customers missing values for %s"%(len(Churn[Churn[missingcol].isnull().all(axis=1)]),missingcol))
Churn[Churn[missingcol].isnull().all(axis=1)][missingcol].head()
# check the missing null and shape again.
Churn=Churn[~Churn[missingcol].isnull().all(axis=1)]
Churn.shape
# There are 381 customers having 89 features. we can imute it with zero.
missingcol.remove('date_of_last_rech_8')
missingcol.remove('date_of_last_rech_9')
imputenan(Churn,missingColList=missingcol)

# Let us check the missing nan again
# Let us check the data set again
checknull(0)
# Let us create the new data frame and store these features and run the uniqueness to impute it further .
columns = ['loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou','last_date_of_month_8','last_date_of_month_9','date_of_last_rech_6','date_of_last_rech_7', 'date_of_last_rech_8', 'date_of_last_rech_9']
for x in columns: 
    print("Unique values in column %s are %s" % (x,Churn[x].unique()))
# It seems are single value only. we can impute it with the same.Let us impute with mode first and proceed
columns = ['loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou','last_date_of_month_7','last_date_of_month_8','last_date_of_month_9']
for x in columns:
    print(Churn[x].value_counts())
    Churn[x].fillna(Churn[x].mode()[0], inplace=True)
print("All the above features take only one value. Lets impute the missing values in these features with the mode")
# We have impute the variables 
checknull(0)

# All these features are missing together
missingcol = list(checknull(0).index)
print ("There are %d rows in total having missing values for these variables."%(len(Churn[Churn[missingcol].isnull().all(axis=1)])))
# Let us impute it with the dates frequently coming. 
Churn[Churn['date_of_last_rech_6'].isnull()]['date_of_last_rech_6'] = '6/30/2014'
Churn[Churn['date_of_last_rech_7'].isnull()]['date_of_last_rech_7'] = '7/31/2014'
Churn[Churn['date_of_last_rech_8'].isnull()]['date_of_last_rech_8'] = '8/31/2014'
Churn[Churn['date_of_last_rech_9'].isnull()]['date_of_last_rech_9'] = '9/30/2014'
# Now we can see that the data is completely cleaned. Let us check the head.
Churn.head()
# We have lot of zeros in the coloumns as a single value. let us drop it and proceed.

Single_value =Churn.columns[(Churn == 0).all()]
print ("There are {} features only having zero as values. These features are \n{}".format(len(Single_value),Single_value))
# We can drop these features. 
Churn.drop(Single_value,axis=1,inplace=True)
# Percentage of data left after removing the missing values.
print("Percentage of data remaining after treating missing values: {}%".format(round(Churn.shape[0]/99999 *100,2)))
print ("Number of customers: {}".format(Churn.shape[0]))
print ("Number of features: {}".format(Churn.shape[1]))
Churn.reset_index(inplace=True,drop=True)
# list of all columns which store date
date_columns = list(Churn.filter(regex='date').columns)
date_columns
# Converting dtype of date columns to datetime
for col in date_columns:
    Churn[col] = pd.to_datetime(Churn[col], format='%m/%d/%Y')
# Percentage of data left after removing the missing values.
Churn.shape
# Let us create the new features. Also Filter high-value customers
# Defining high-value customers as follows:
#Those who have recharged with an amount more than or equal to X, where X is the 70th percentile of the average recharge amount in the first two months (the good phase).
rech_col = Churn.filter(regex=('count')).columns
Churn[rech_col].head()


# Creating new feature: avg_rech_amt_6,avg_rech_amt_7,avg_rech_amt_8,avg_rech_amt_9
for i in range(6,10):
    Churn['avg_rech_amt_'+str(i)] = round(Churn['total_rech_amt_'+str(i)]/Churn['total_rech_num_'+str(i)]+1,2)
# impute the NAN values

imputenan(Churn,missingColList=['avg_rech_amt_6','avg_rech_amt_7','avg_rech_amt_8','avg_rech_amt_9'])
# Let us create the total recharge amounts of all the months and store it.
# total recharge amount = count of recharge 2g + count of recharge of 3g of all months and convert to integer data.
for i in range(6,10):
    Churn['total_rech_num_data_'+str(i)] = (Churn['count_rech_2g_'+str(i)]+Churn['count_rech_3g_'+str(i)]).astype(int)
# let us store the total recharge amount data = total rechage number data * average rechage amount data
for i in range(6,10):
    Churn['total_rech_amt_data_'+str(i)] = Churn['total_rech_num_data_'+str(i)] * Churn['av_rech_amt_data_'+str(i)]
# Another new feature : total month recharge = total recharge amount + total recharge data for each customer each month
for i in range(6,10):
    Churn['total_month_rech_'+str(i)] = Churn['total_rech_amt_'+str(i)]+Churn['total_rech_amt_data_'+str(i)]
Churn.filter(regex=('total_month_rech')).head()

# calculating the avegare of first two months (good phase) total monthly recharge amount
Good_Phase_avg =(Churn.total_month_rech_6 + Churn.total_month_rech_7)/2
# finding the cutoff which is the 70th percentile of the good phase average recharge amounts
Cut_off = np.percentile(Good_Phase_avg,70)
# Filtering the users whose good phase avg. recharge amount >= to the cutoff of 70th percentile.
Highvalu_users = Churn[Good_Phase_avg >= Cut_off]
# Reset the index.
Highvalu_users.reset_index(inplace=True,drop=True)

print("Number of High-Value Customers in the Dataset: %d\n"% len(Highvalu_users ))
print("Percentage High-value users in data : {}%".format(round(len(Highvalu_users )/Churn.shape[0]*100),2))
## We need the tag the churners. 
#Tagging Churners. Now tag the churned customers (churn=1, else 0) based on the fourth month as follows:

#Those who have not made any calls (either incoming or outgoing) AND have not used mobile internet even once in the churn phase. The attributes we need to use to tag churners are:

#total_ic_mou_9
#total_og_mou_9
#vol_2g_mb_9
#vol_3g_mb_9

def getChurnStatus(data,Churn_Phase_Month=9):
    # Function to tag customers as churners (churn=1, else 0) based on 'vol_2g_mb_','vol_3g_mb_','total_ic_mou_','total_og_mou_'
    #argument: churnPhaseMonth, indicating the month number to be used to define churn (default= 9)
    Churn_features= ['vol_2g_mb_','vol_3g_mb_','total_ic_mou_','total_og_mou_']
    flag = ~data[[s + str(Churn_Phase_Month) for s in Churn_features ]].any(axis=1)
    flag = flag.map({True:1, False:0})
    return flag
# Check the Churn and not churn data for high value customers.
Highvalu_users['Churn'] = getChurnStatus(Highvalu_users,9)
print(" We have {} users tagged as churners out of {} High Value Customers.".format(len(Highvalu_users[Highvalu_users.Churn == 1]),Highvalu_users.shape[0]))
print("High-value Churn Percentage : {}%".format(round(len(Highvalu_users[Highvalu_users.Churn == 1])/Highvalu_users.shape[0] *100,2)))
Highvalu_users.head()
Highvalu_users.shape
# As we see the REvenue is one of the important parameter. Let us create the total revenues. 
#To find the Average Revenue per unit ARU  = total Revenue / Average Subscribers
# To get the Total Revenue = Average Revenue per unit ARU * Average Subscribers(5000 nos. of the total)

# Another new feature : total month recharge = total recharge amount + total recharge data for each customer each month
for i in range(6,10):
    Highvalu_users['Total_revenue_'+str(i)] = Highvalu_users['arpu_'+str(i)] * 5000

Highvalu_users.filter(regex=('Total_revenue_')).head()
# Total Revenue in the Phase wise manner good Phase - first two months, action phase is third month and fourth month is churn phase.
# Let us see the revenue in the Phase wise manner

Highvalu_users['Good_Phase'] = Highvalu_users['Total_revenue_6'] + Highvalu_users['Total_revenue_7']
Highvalu_users['Action_Phase'] =  Highvalu_users['Total_revenue_8']
Highvalu_users['Chrun_Phase'] = Highvalu_users['Total_revenue_9']
#After tagging churners, remove all the attributes corresponding to the churn phase (all attributes having ‘ _9’, etc. in their names).
Chrun_Phase = Highvalu_users.filter(regex='_9', axis=1)
Highvalu_users.drop(Chrun_Phase,axis=1,inplace=True)
Highvalu_users.shape
# Let's see the correlation matrix 
plt.figure(figsize = (20,20))        # Size of the figure
sns.heatmap(Highvalu_users.corr(),annot = True, fmt = ".2f", cmap = "GnBu")
# correlation matrix
corr_matrix = Highvalu_users.corr().abs()

# Selecting the upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# feature columns with correlation greater than 0.80
high_corr_feat = [column for column in upper.columns if any(upper[column] > 0.80)]

print("HIGHLY CORRELATED FEATURES IN DATA SET:{}\n\n{}".format(len(high_corr_feat), high_corr_feat))
# Inferences : Around 66 Variables are highly correlated each other more than 80% 
# Let us check only these variables correlation and find the inferences and proceed for the univarate and bi-variate analysis.
new_corrs = Highvalu_users[['onnet_mou_8', 'loc_og_t2t_mou_8', 'loc_og_t2m_mou_8', 'loc_og_t2f_mou_7', 'loc_og_mou_6', 'loc_og_mou_7', 'loc_og_mou_8', 'std_og_t2t_mou_6', 'std_og_t2t_mou_7', 'std_og_t2t_mou_8', 'std_og_t2m_mou_6', 'std_og_t2m_mou_7', 'std_og_t2m_mou_8', 'isd_og_mou_7', 'isd_og_mou_8', 'total_og_mou_6', 'total_og_mou_7', 'total_og_mou_8', 'loc_ic_t2t_mou_7', 'loc_ic_t2t_mou_8', 'loc_ic_t2m_mou_8', 'loc_ic_mou_6', 'loc_ic_mou_7', 'loc_ic_mou_8', 'std_ic_mou_6', 'std_ic_mou_7', 'std_ic_mou_8', 'total_ic_mou_6', 'total_ic_mou_7', 'total_ic_mou_8', 'total_rech_amt_6', 'total_rech_amt_7', 'total_rech_amt_8', 'count_rech_2g_6', 'count_rech_2g_7', 'count_rech_2g_8', 'av_rech_amt_data_8', 'arpu_3g_6', 'arpu_3g_7', 'arpu_3g_8', 'arpu_2g_6', 'arpu_2g_7', 'arpu_2g_8', 'sachet_2g_6', 'sachet_2g_7', 'sachet_2g_8', 'monthly_3g_6', 'monthly_3g_7', 'monthly_3g_8', 'sachet_3g_6', 'sachet_3g_7', 'sachet_3g_8', 'avg_rech_amt_7', 'avg_rech_amt_8', 'total_rech_num_data_6', 'total_rech_num_data_7', 'total_rech_num_data_8', 'total_month_rech_6', 'total_month_rech_7', 'total_month_rech_8', 'Total_revenue_6', 'Total_revenue_7', 'Total_revenue_8', 'Good_Phase', 'Action_Phase', 'Chrun_Phase']]
# Let's see the correlation matrix 
plt.figure(figsize = (20,20))        # Size of the figure
sns.heatmap(new_corrs.corr(),annot = True, fmt = ".2f", cmap = "GnBu")
# Check the churn in a graphical way 

C_IT = sns.catplot("Churn", data = Highvalu_users, aspect=1.5, kind="count", color="b")
C_IT.set_xticklabels(rotation=30)
plt.show()
Highvalu_users.head()
#check the list of all the columns. 
list(Highvalu_users.columns)
# the date coloumns are not needed as we have only unique values as well as the mobile nos. Let us drop and proceed
Highvalu_users.drop(['mobile_number','last_date_of_month_6','last_date_of_month_7','last_date_of_month_8','date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8'],axis=1,inplace=True)
# sns.boxplot(y='arpu_6', data=tel)
cont_cols = [col for col in Highvalu_users.columns if col not in ['Churn']]
for col in cont_cols:
    plt.figure(figsize=(5, 5))
    sns.boxplot(y=col, data= Highvalu_users)
# Let us check the Chrun as the target with the other variables 
cont_cols = [col for col in Highvalu_users.columns if col not in ['Churn']]
for col in cont_cols:
    plt.figure(figsize=(5, 5))
    sns.barplot(x='Churn', y=col, data=Highvalu_users)
    plt.show()
# Check the Age of the Network used by the operator Vs the other variables expect churn.
cont_cols = [col for col in Highvalu_users.columns if col not in ['Churn']]
for col in cont_cols:
    plt.figure(figsize=(5, 5))
    sns.jointplot(x='aon',y = col,data= Highvalu_users)
    plt.show()                                                
# Check the Age of the Network used by the operator Vs the other variables expect churn.
# rug = True
# plotting only a few points since rug takes a long while
sns.distplot(Highvalu_users['aon'][:200], rug=True)
plt.show()
### Inferences : The data is normally distributed with outliers with respect to age on network.
# Do the X train , Y train test and seperate the data sets. 
# import needed model building sklearn parameters.
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#putting features variables in X
X = Highvalu_users.drop(['Churn'], axis=1)

#putting response variables in Y
y = Highvalu_users['Churn']    

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)
### Scaling before PCA 
#Rescaling the features before PCA as it is sensitive to the scales of the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# fitting and transforming the scaler on train
X_train = scaler.fit_transform(X_train)
# transforming the train using the already fit scaler
X_test = scaler.transform(X_test)

# Check the sample before for 0 and 1 :
print("Before sample the counts of label '1': {}".format(sum(y_train==1)))
print("Before sample counts of label '0': {} \n".format(sum(y_train==0)))
print("Before sample churn event rate : {}% \n".format(round(sum(y_train==1)/len(y_train)*100,2)))
# Import the SMOTE and Joblib for solving this issue of imbalance. 
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE
from scipy import sparse


sm = SMOTE(random_state=12, ratio = 1)
X_train_re, y_train_re = sm.fit_sample(X_train, y_train)
# Check the sample after for 0 and 1 :
print("After sample the counts of label '1': {}".format(sum(y_train_re==1)))
print("After sample counts of label '0': {} \n".format(sum(y_train_re==0)))
print("After sample churn event rate : {}% \n".format(round(sum(y_train_re==1)/len(y_train_re)*100,2)))
# Now we can see that the data is now 50% balanced. Let us also do the outlier analysis and proceed further.
# Let us check the outlier present in the data before doing modelling and remove the outliers and proceed with the modelling.

prev_box = list(Highvalu_users.columns)
for i in Highvalu_users[prev_box]:
    plt.figure(1,figsize=(15,5))
    sns.boxplot(Highvalu_users[i])
    plt.xticks(rotation = 90,fontsize =10)
    plt.show()
# Removal of the data sets from the outliers.Let us take the lower and upper limits are 0.05 and .9 on the upper. so that we could avoid loosing more data.
test_box_Df2 = list(Highvalu_users.columns) 
new_copy = Highvalu_users[test_box_Df2]
for i in new_copy.columns:
    Q1 = new_copy[i].quantile(0.05)
    Q3 = new_copy[i].quantile(0.90)

    IQR = Q3 - Q1
    
    lower_fence = Q1 - 1.5*IQR
    upper_fence = Q3 + 1.5*IQR

    new_copy[i][new_copy[i] <= lower_fence] = lower_fence
    new_copy[i][new_copy[i] >= upper_fence] = upper_fence
    
    print("oUTLIERS:",i,lower_fence,upper_fence)
    
    plt.figure(1,figsize=(10,5))
    sns.boxplot(new_copy[i])
    plt.xticks(rotation =90,fontsize =10)
    plt.show()
## Inferences : We have cleaned the data and now we can proceed for the modelling which we defined as we took lower ranges of the quartiles due to avoid loss of data
# We have stored the cleaned data in newcopy. Let us check the head again
new_copy.head()
# check the data shape again before modelling.
new_copy.shape
# Let us do the PCA to reduce the dimentionality of the data and proceed with the Logistic regression first and move forward to highermodels 

#Improting the PCA module
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)
# apply the PCA
pca.fit(X_train_re)
#List of PCA components.It would be the same as the number of variables
pca.components_
#Let's check the variance ratios
pca.explained_variance_ratio_[:50]
# Import the matplot and visualse the pca variance ratio in a bar graph.
import matplotlib.pyplot as plt
plt.figure(figsize = (6,5))        # Size of the figure

plt.bar(range(1,len(pca.explained_variance_ratio_[:50])+1), pca.explained_variance_ratio_[:50])
# most of the data in the 0 to 2 ranges.  let us see the cummulative vairance ratio.
var_cumu = np.cumsum(pca.explained_variance_ratio_)
# We can see 0 to 10 most of the data are lying. 
# Make the scree plots cleary for choosing the no. of PCA 
plt.figure(figsize=(8,6))
plt.title('Scree plots')
plt.xlabel('No. of Components')
plt.ylabel('Cummulative explained variance')

plt.plot(range(1,len(var_cumu)+1), var_cumu)
plt.show()
# Looks. we will take 35 components for desctribe the 95% of the varaince in the datasets.
#Using incremental PCA for efficiency - saves a lot of time on larger datasets
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=40)
# let us fit the data
X_train_pca = pca_final.fit_transform(X_train_re)
X_train_pca.shape
#creating correlation matrix for the principal components
corrmat = np.corrcoef(X_train_pca.transpose())
# 1s ----> 0s in diagonals
corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)
# we see that correlations are indeed very close to 0
# Seems there is no correlation int the data sets between any two variables in PCA. We dont have any multicolinearity.
#Applying selected components to the test data - 35 components
X_test_pca = pca_final.transform(X_test)
X_test_pca.shape
#Import needed libraries
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lr = LogisticRegression(class_weight='balanced')
#Fit the algorithm on the data
def model_fit(alg, X_train, y_train, performCV=True, cv_folds=5):
    alg.fit(X_train, y_train)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, X_train, y_train, cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print ("\nModel Summary:")
    print ("Accuracy : %.4g" % metrics.roc_auc_score(y_train, dtrain_predictions))
    print ("Recall/Sensitivity : %.4g" % metrics.recall_score(y_train, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
    
    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
# Let us fit the model. 
model_fit(lr, X_train_pca, y_train_re)
# Define the Modelmetrics .
def Model_metrics(Actual_churn=False,Predict_churn=False):

    confusion = metrics.confusion_matrix(Actual_churn, Predict_churn)

    TP = confusion[1,1] # true positive 
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] # false positives
    FN = confusion[1,0] # false negatives

    print("Roc_auc_score : {}".format(metrics.roc_auc_score(Actual_churn, Predict_churn)))
    # Let's see the sensitivity of our logistic regression model
    print('Sensitivity/Recall : {}'.format(TP / float(TP+FN)))
    # Let us calculate specificity
    print('Specificity: {}'.format(TN / float(TN+FP)))
    # Calculate false postive rate - predicting churn when customer does not have churned
    print('False Positive Rate: {}'.format(FP/ float(TN+FP)))
    # positive predictive value 
    print('Positive predictive value: {}'.format(TP / float(TP+FP)))
    # Negative predictive value
    print('Negative Predictive value: {}'.format(TN / float(TN+ FN)))
    # sklearn precision score value 
    print('sklearn precision score value: {}'.format(metrics.precision_score(Actual_churn, Predict_churn)))
# predictions on Test data
pred_probs_test = lr.predict(X_test_pca)
Model_metrics(y_test,pred_probs_test)
# check the accuracy, recall and precision again,
print("Accuracy : {}".format(metrics.accuracy_score(y_test,pred_probs_test)))
print("Recall : {}".format(metrics.recall_score(y_test,pred_probs_test)))
print("Precision : {}".format(metrics.precision_score(y_test,pred_probs_test)))
#Making prediction on the test data
pred_probs_train = lr.predict_proba(X_train_pca)[:,1]
print("roc_auc_score(Train) {:2.2}".format(metrics.roc_auc_score(y_train_re, pred_probs_train)))
# Let us define for the cutoff values
#Function to find the optimal cutoff for classifing as churn/non-churn
# Let's create columns with different probability cutoffs 
def Optimal_Cutoff(df):
    #Function to find the optimal cutoff for classifing as churn/non-churn
    # Let's create columns with different probability cutoffs 
    numbers = [float(x)/10 for x in range(10)]
    for i in numbers:
        df[i] = df.churn_Prob.map( lambda x: 1 if x > i else 0)
    #print(df.head())
    
    # Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
     # TP = confusion[1,1] # true positive 
    # TN = confusion[0,0] # true negatives
    # FP = confusion[0,1] # false positives
    # FN = confusion[1,0] # false negatives
    
    cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
    
    from sklearn.metrics import confusion_matrix
    
    num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for i in num:
        cm1 = metrics.confusion_matrix(df.churn, df[i] )
        total1=sum(sum(cm1))
        accuracy = (cm1[0,0]+cm1[1,1])/total1
        
        speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
        cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
    print(cutoff_df)
    
    # Let's plot accuracy sensitivity and specificity for various probabilities.
    cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
    plt.show()
def predictChurnWithProb(model,X,y,prob):
    # Funtion to predict the churn using the input probability cut-off
    # Input arguments: model instance, x and y to predict using model and cut-off probability
    
    # predict
    pred_probs = model.predict_proba(X)[:,1]
    
    y_df= pd.DataFrame({'churn':y, 'churn_Prob':pred_probs})
    # Creating new column 'predicted' with 1 if Churn_Prob>0.5 else 0
    y_df['final_predicted'] = y_df.churn_Prob.map( lambda x: 1 if x > prob else 0)
    # Let's see the head
    Model_metrics(y_df.churn,y_df.final_predicted)
    return y_df
cut_off_prob=0.5
y_train_df = predictChurnWithProb(lr,X_train_pca,y_train_re,cut_off_prob)
y_train_df.head()
# Let us define the roc curveL
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 6))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds
# Check the ROC curve
draw_roc(y_train_df.churn, y_train_df.final_predicted)
# the ROC curve should be on the top left. showing good fit.

#draw_roc(y_pred_final.Churn, y_pred_final.predicted)
print("roc_auc_score : {:2.2f}".format(metrics.roc_auc_score(y_train_df.churn, y_train_df.final_predicted)))

# finding cut-off with the right balance of the metrices
# sensitivity vs specificity trade-off
Optimal_Cutoff(y_train_df)
# Note that the cutoff is between the 0.5 to 0.6. we choose to take the 0.52. at this point there will be balance of accuracy, sensitivity and specificity

# predicting with the choosen cut-off on train
cut_off_prob = 0.52
A = predictChurnWithProb(lr,X_train_pca,y_train_re,cut_off_prob)
A.head()
# Let us predict on the test set

# predicting with the choosen cut-off on test
B = predictChurnWithProb(lr,X_test_pca,y_test,cut_off_prob)
B.head()
# let us apply the Decision tree with PCA and Hyperparameters and fit.

Dt = DecisionTreeClassifier(class_weight='balanced',
                             max_features='auto',
                             min_samples_split=100,
                             min_samples_leaf=100,
                             max_depth=6,
                             random_state=10)
model_fit(Dt, X_train_pca, y_train_re)
# make predictions
pred_probs_test = Dt.predict(X_test_pca)
#Let's check the model metrices.
Model_metrics(Actual_churn=y_test,Predict_churn=pred_probs_test)

# Create the parameter grid based on the results of random search 
param_grid = {'max_depth': range(5,15,3),'min_samples_leaf': range(100, 400, 50),'min_samples_split': range(100, 400, 100),
    'max_features': [8,10,15]}
# Create a based model
dt = DecisionTreeClassifier(class_weight='balanced',random_state=10)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = dt, param_grid = param_grid, 
                          cv = 3, n_jobs = 4,verbose = 1,scoring="f1_weighted")
# Fit the grid search to the data
grid_search.fit(X_train_pca, y_train_re)
# printing the optimal accuracy score and hyperparameters
print('Recall score',grid_search.best_score_)
# model with the best hyperparameters
dt_final = DecisionTreeClassifier(class_weight='balanced',
                             max_depth=14,
                             min_samples_leaf=100, 
                             min_samples_split=100,
                             max_features=15,
                             random_state=10)
# fit th model and get the model summary:
model_fit(dt_final,X_train_pca,y_train_re)
# make predictions in the test data
pred_probs_test = dt_final.predict(X_test_pca)
#Let's check the model metrices.
Model_metrics(Actual_churn=y_test,Predict_churn=pred_probs_test)
# classification report
print(classification_report(y_test,pred_probs_test))
# After tunning aslo, we can see we get 71% of the churn.
# predicting churn with default cut-off 0.5
cut_off_prob = 0.5
y_train_df = predictChurnWithProb(dt_final,X_train_pca,y_train_re,cut_off_prob)
y_train_df.head()

# finding cut-off with the right balance of the metrices
Optimal_Cutoff(y_train_df)
# We can see that the cut off value is between 0.5 to 0.6. Let us chck again the cutoff as 0.52 and run again.
cut_off_prob = 0.5
y_train_df = predictChurnWithProb(dt_final,X_train_pca,y_train_re,cut_off_prob)
y_train_df.head()
# At 0.52 we can see there is a balance in the sensitivity, specificity and acccuracy.


#Lets see how it performs on test data.
y_test_df= predictChurnWithProb(dt_final,X_test_pca,y_test,cut_off_prob)
y_test_df.head()
# Again let us appy the Random forest pca and hypertuning with maxdepth
parameters = {'max_depth': range(10, 30, 5)}
rfc = RandomForestClassifier()
rfgs = GridSearchCV(rfc, parameters,n_jobs=-1,
                    cv=5, 
                   scoring="f1")
rfgs.fit(X_train_pca,y_train_re)
scores = rfgs.cv_results_
# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# We can see that the max_depth after 20th both the curves there were no significant change observed.
# Let us tune with the n_estimators 
parameters = {'n_estimators': range(50, 150, 25)}
rf1 = RandomForestClassifier(max_depth=20,n_jobs=-1,random_state=10)
rfgs = GridSearchCV(rf1, parameters, 
                    cv=3, 
                   scoring="recall")
# AGain fit the parameters

rfgs.fit(X_train_pca,y_train_re)
# plotting accuracies with max_depth
def plt_traintest_acc(score,param):
    scores = score
    plt.figure()
    plt.plot(scores["param_"+param], 
    scores["mean_train_score"], 
    label="training accuracy")
    plt.plot(scores["param_"+param], 
    scores["mean_test_score"], 
    label="test accuracy")
    plt.xlabel(param)
    plt.ylabel("f1")
    plt.legend()
    plt.show()
# check the plot
plt_traintest_acc(rfgs.cv_results_,'n_estimators')
# We can see between 70 to 80, let us take 80 as our n_estimators 
# Max features tuning with 80 
parameters = {'max_features': [4, 8, 14, 20, 24]}
rf3 = RandomForestClassifier(max_depth=20,n_estimators=80,n_jobs=-1,random_state=10)
rfgs = GridSearchCV(rf3, parameters,cv=5,scoring="f1")
# Let us fit again and check 
rfgs.fit(X_train_pca,y_train_re)
# check the acc. max features
plt_traintest_acc(rfgs.cv_results_,'max_features')
# As we see that at the 7.5 and later it is declining. 
# Let us tune the min sample leaf
parameters = {'min_samples_leaf': range(100, 400, 50)}
rf4 = RandomForestClassifier(max_depth=20,n_estimators=80,max_features=8,n_jobs=-1,random_state=10)
rfgs = GridSearchCV(rf4, parameters,cv=3, scoring="f1")

# Let us fit again and plt the curve 
rfgs.fit(X_train_pca,y_train_re)
plt_traintest_acc(rfgs.cv_results_,'min_samples_leaf')
# As we see that the Min samples leaf is 100.
# let us tune min samples split.
parameters = {'min_samples_split': range(50, 300, 50)}
rf5 = RandomForestClassifier(max_depth=20,n_estimators=80,max_features=8,n_jobs=-1,min_samples_leaf=100,random_state=10)
rfgs = GridSearchCV(rf5, parameters,cv=3,scoring="f1")
## Let us fit again and plt the curve 
rfgs.fit(X_train_pca,y_train_re)
plt_traintest_acc(rfgs.cv_results_,'min_samples_split')
# the min samples split is 50 later almost flat and it is declining at 200.
# Final - tunned in all aspects 

rf_final = RandomForestClassifier(max_depth=20,n_estimators=80,n_jobs=-1,max_features=8,min_samples_leaf=100,min_samples_split=50,random_state=10)
# check the train set.
model_fit(rf_final,X_train_pca,y_train_re)
# predict on test data
predictions = rf_final.predict(X_test_pca)
# Check the model metrics in test predictions
Model_metrics(y_test,predictions)
# After fine tunning we can see the recall is for the final Random forest is only 73% with 80% accuracy.
# Let us check the cutoff optimal value as we did for the other models
# predicting churn with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictChurnWithProb(rf_final,X_train_pca,y_train_re,cut_off_prob)
y_train_df.head()
# Let us see the cut-off with the metrics of accuracy, sensitvity and specificity
Optimal_Cutoff(y_train_df)
# We can see the optimal value is lying in 0.5 to 0.6. let us take as 0.52
cut_off_prob=0.52
A = predictChurnWithProb(rf_final,X_train_pca,y_train_re,cut_off_prob)
A.head()
# Make the prediction on the test:
y_test_df= predictChurnWithProb(rf_final,X_test_pca,y_test,cut_off_prob)
y_test_df.head()
# Gradient boosing classifier with PCA and hypertuning 
# Impor the needed libraries
from sklearn.ensemble import GradientBoostingClassifier  

# Fitting the default GradientBoostingClassifier
Gb = GradientBoostingClassifier(random_state=10)
model_fit(Gb, X_train_pca, y_train_re)
# Let us tune the n_estimators
param_test_1 = {'n_estimators':range(20,150,10)}
gsearch_1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test_1, scoring='f1',n_jobs=4,iid=False, cv=3)
gsearch_1.fit(X_train_pca, y_train_re)
# Let us check the best n_estimators first
print('gsearch_1.best_params_\ngsearch1.best_score_')
# check the score and n_estimators
print("The best n_estimators: {}".format(gsearch_1.best_params_))
print("The best score: {}".format(gsearch_1.best_score_))

# Let us use the n_estimators =140 and tune the hyperparameters of max depth min samples spli & Learning rate is taken as 0.1.
param_test_2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch_2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=140, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test_2, scoring='f1',n_jobs=4,iid=False, cv=3)
gsearch_2.fit(X_train_pca, y_train_re)
# check the score and hyper tune parameters
print("The best max depth & min samples split: {}".format(gsearch_2.best_params_))
print("The best score: {}".format(gsearch_2.best_score_))

# Let us tune the hyperparameters of  min samples leaf.
param_test_3 = {'min_samples_leaf':range(30,71,10)}
gsearch_3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,n_estimators=140,max_depth = 15,min_samples_split =200, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test_3, scoring='f1',n_jobs=4,iid=False, cv=3)
gsearch_3.fit(X_train_pca, y_train_re)
# check again the score and hyper tune parameter min_samples_leaf
print("The best min_samples_leaf: {}".format(gsearch_3.best_params_))
print("The best score: {}".format(gsearch_3.best_score_))
# Check again the Max_features hypertuning and do the gridsearch.
param_test_4 = {'max_features':range(7,20,2)}
gsearch_4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,n_estimators=140,max_depth=15, min_samples_split=200, min_samples_leaf=30, subsample=0.8, random_state=10),
param_grid = param_test_4, scoring='f1',n_jobs=4,iid=False, cv=3)
gsearch_4.fit(X_train_pca, y_train_re)
# check again the score and hyper tune parameter 
print("The best max_features: {}".format(gsearch_4.best_params_))
print("The best score: {}".format(gsearch_4.best_score_))
# The Final Model for Gradient boosting with max_features =19 
Gb_final = GradientBoostingClassifier(learning_rate=0.1,n_estimators=140,max_features=19,max_depth=15, min_samples_split=200, min_samples_leaf=40, subsample=0.8, random_state=10)
model_fit(Gb_final, X_train_pca, y_train_re)
# predictions on Test data & check the metrics on test data
test_predict = Gb_final.predict(X_test_pca)
Model_metrics(y_test,test_predict)
# Let us do the predict the churn for the default cutoff and fine tune later.
cut_off_prob=0.5
y_train_df = predictChurnWithProb(Gb_final,X_train_pca,y_train_re,cut_off_prob)
y_train_df.head()
# let see the optimal cutoff:
Optimal_Cutoff(y_train_df)
# We can see that the optimal cutoff points between 0.2 to 0.3. let us take as 0.2 and proceed.
cut_off_prob=0.2
A = predictChurnWithProb(Gb_final,X_train_pca,y_train_re,cut_off_prob)
A.head()
# Let us do predict in the test data:
y_test_df= predictChurnWithProb(Gb_final,X_test_pca,y_test,cut_off_prob)
y_test_df.head()
# Import the needed libraries
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
# Fitting the XGBClassifier
XGb = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,
                    objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)
# Model fit and performance on Train data
model_fit(XGb, X_train_pca, y_train_re)
# Let us tune the hyperparameters one by one
param_test_1 = {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)}
gsearch_1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
             min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
            param_grid = param_test_1, scoring='f1',n_jobs=4,iid=False, cv=3)
gsearch_1.fit(X_train_pca, y_train_re)
# check again the score and hyper tune parameter 
print("The best ma_depth & min_child_weight: {}".format(gsearch_1.best_params_))
print("The best score: {}".format(gsearch_1.best_score_))
# Let us tune the hyperparameters one by one
param_test_2 = {'gamma':[i/10.0 for i in range(0,5)]}
gsearch_2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=9,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
             objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test_2, scoring='f1',n_jobs=4,iid=False, cv=3)
gsearch_2.fit(X_train_pca, y_train_re)
# check again the score and hyper tune parameter 
print("The best gamma: {}".format(gsearch_2.best_params_))
print("The best score: {}".format(gsearch_2.best_score_))
# Final XGb boosing with gamma 0.2 & fit the model.
XGb = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=9,min_child_weight=1, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
     objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)
model_fit(XGb, X_train_pca, y_train_re)
# Predict on the test data and do the model metrics.
test_predict = XGb.predict(X_test_pca)
Model_metrics(y_test,test_predict)
# Take the default cutoff and check again
# predicting churn with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictChurnWithProb(XGb,X_train_pca,y_train_re,cut_off_prob)
y_train_df.head()
# Finding optimal cut-off probability
Optimal_Cutoff(y_train_df)
# the optimal cutoff point will be between 0.3 to 0.5, taken as 0.4
cut_off_prob=0.4
A = predictChurnWithProb(XGb,X_train_pca,y_train_re,cut_off_prob)
A.head()
# Let us predict in the test data
y_test_df= predictChurnWithProb(XGb,X_test_pca,y_test,cut_off_prob)
y_test_df.head()
# Let us try doing the model test in SVC and check. https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
# note that we are using cost C=1
from sklearn.svm import SVC 

Sv = SVC(C = 1)

# fit
Sv.fit(X_train_pca, y_train_re)

# predict on train
S_predict = Sv.predict(X_train_pca)

# check the model metrics
Model_metrics(y_train_re,S_predict)
# Predict on test data
S_predict = Sv.predict(X_test_pca)
Model_metrics(y_test,S_predict)
# Let us do the hyper tunning 
# specify range of parameters (C) as a list
params = {"C": [0.1, 1, 10, 100, 1000]}

Svm= SVC()

# set up grid search and fit on the train set.

model_cv = GridSearchCV(estimator = Svm, param_grid = params, scoring= 'f1',cv = 5, verbose = 1,n_jobs=4,return_train_score=True) 
model_cv.fit(X_train_pca, y_train_re)
# check again the score and hyper tune parameter 
print("The best params: {}".format(model_cv.best_params_))
print("The best score: {}".format(model_cv.best_score_))
# use the C as 1000 and run again
Sv_final = SVC(C = 1000)

# fit
Sv_final.fit(X_train_pca, y_train_re)

# predict on train
S_predict = Sv_final.predict(X_train_pca)
# check the model metrics
Model_metrics(y_train_re,S_predict)
# Check in the test set
S_predict = Sv_final.predict(X_test_pca)
Model_metrics(y_test,S_predict)
# Let us see the top variables which contributes more to the chrun rather than false postives.
# Random forest is used to derive the churn

# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [8,10,12],
    'min_samples_leaf': range(100, 400, 200),
    'min_samples_split': range(200, 500, 200),
    'n_estimators': [100,200, 300], 
    'max_features': [12, 15, 20]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = 4,verbose = 1)
# Fit the grid search to the data
grid_search.fit(X_train_re, y_train_re)
# printing the optimal accuracy score and hyperparameters
print('Accuracy is',grid_search.best_score_,'using',grid_search.best_params_)
# Apply rf calssifier with hyper tuning and fit the model
rf = RandomForestClassifier(max_depth=12,max_features=20,min_samples_leaf=100,min_samples_split=200,n_estimators=300,random_state=10)
rf.fit(X_train_re, y_train_re)
# Plot the bar graph
plt.figure(figsize=(10,40))
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(len(X.columns)).sort_values().plot(kind='barh', align='center')
