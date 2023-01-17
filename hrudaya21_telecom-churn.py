import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import time
import datetime
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/telecom_churn_data.csv')
# Function to Return Monthwise ColumnsList. Returns arrays of columns belonging to 6,7,8,9 month separately.
# Also returns an array of columns that are not month specific as common columns.
def returnColumnsByMonth(df):
    column_Month_6 = []
    column_Month_7 = []
    column_Month_8 = []
    column_Month_9 = []
    column_Common = []
    for eachColumns in df.columns:
        if((eachColumns.find("_6") >=0) | (eachColumns.find("jun_") >=0)):
            column_Month_6.append(eachColumns)
        elif((eachColumns.find("_7") >=0) | (eachColumns.find("jul_") >=0)):
            column_Month_7.append(eachColumns)
        elif((eachColumns.find("_8") >= 0) | (eachColumns.find("aug_") >=0)):
            column_Month_8.append(eachColumns)
        elif((eachColumns.find("_9") >=0) | (eachColumns.find("sep_") >=0)):
            column_Month_9.append(eachColumns)
        else:
            column_Common.append(eachColumns)
    return column_Month_6, column_Month_7, column_Month_8, column_Month_9, column_Common

# Function to Get Columns Based on Null %. 
#Returns columns that have % of null values higher or lower than nullPercentLimit
def getColumnsBasedOnNullPercent(df, nullPercentLimit, limitType = 'Upper'):
    col2NullPercent_df = pd.DataFrame(round((df.isnull().sum()/len(df.index))* 100, 2), columns=['NullPercent'])
    col2NullPercent_df = pd.DataFrame(round((df.isnull().sum()/len(df.index))* 100, 2), columns=['NullPercent'])
    if(limitType == 'Upper'):
        columnsList = np.array(col2NullPercent_df.apply(lambda x: x['NullPercent'] > nullPercentLimit , axis=1))
    if(limitType == 'Lower'):
        columnsList = np.array(col2NullPercent_df.apply(lambda x: ((x['NullPercent'] < nullPercentLimit) & (x['NullPercent'] > 0)) , axis=1))
    return np.array(df.loc[:, columnsList].columns)

# Function to get Days Since Last Recharge for 6/7/8/9 months
def daysSinceLastRechargeMonthwise(df, month):
    if(month == 6):
        return pd.to_datetime(df['last_date_of_month_6']) - pd.to_datetime(df['date_of_last_rech_6'])
    elif(month == 7):
        return pd.to_datetime(df['last_date_of_month_7']) - pd.to_datetime(df['date_of_last_rech_7'])
    elif(month == 8):
        return pd.to_datetime(df['last_date_of_month_8']) - pd.to_datetime(df['date_of_last_rech_8'])
    elif(month == 9):
        return pd.to_datetime(df['last_date_of_month_9']) - pd.to_datetime(df['date_of_last_rech_9'])

def plotCategoricalChurn_NotChurn(df, columnsList, flag = 0):
    for eachMonth in columnsList:
    #flag = 1        
    #eachMonth = "days_from_LastRechage_6"
        col = eachMonth
        X1 = df.groupby('churn')[col].agg(['mean']).reset_index()
        X1.rename(columns={'mean':col}, inplace=True)
        if(flag == 1):
            seventhMonth = eachMonth[:-1] + "7"
            X2 = df.groupby('churn')[seventhMonth].agg(['mean']).reset_index()    
            X2.rename(columns={'mean':seventhMonth}, inplace=True)
            X2 = pd.merge(X1,X2, on = ['churn'])
            newCol = eachMonth[:-1] + "goodPeriod_Avg"
            print(newCol)
            X2[newCol] = (X2[eachMonth] + X2[seventhMonth])/2
            p = sns.barplot(x='churn', y=newCol, data=X2)
            p.set_xticklabels(['Not-Churn','churn'], fontsize= 12)
            plt.ylabel(newCol,fontsize = 12)
            plt.xlabel('churn', fontsize = 12)
            plt.show()
            X2.head()

        else:
            print(eachMonth)
            p = sns.barplot(x='churn', y=col, data=X1)
            p.set_xticklabels(['Not-Churn','churn'], fontsize= 12)
            plt.ylabel(col,fontsize = 12)
            plt.xlabel('churn', fontsize = 12)
            plt.show()
            X1.head()

#Function to Show Howmuch % usage done for Churn Subscriber with respect to total usage on that month, for tht particular feature
# e.g: arpu (Averag Revenue Per User) ==> How much is the average arpu for churn and non-churn subscriber in month 6 7 and 8.
# Then check for churn subscriber how much % usage on total mean Usage in each month
def churnSubscriberUsageChangePercentage():
    for count, eachFeature in enumerate(column_Month_6):
        col = eachFeature
        X1 = df.groupby(['churn'])[col].agg(['mean']).reset_index()
        X1.rename(columns={'mean': "mean_"+col}, inplace=True)
        if(col == 'jun_vbc_3g'):
            col = 'jul_vbc_3g'
        else:
            col = col[:-1] + "7"
        X2 = df.groupby(['churn'])[col].agg(['mean']).reset_index()
        X2.rename(columns={'mean': "mean_"+col}, inplace=True)
        if(col == 'jul_vbc_3g'):
            col = 'aug_vbc_3g'
        else:
            col = col[:-1] + "8"
        X3 = df.groupby(['churn'])[col].agg(['mean']).reset_index()
        X3.rename(columns={'mean': "mean_"+col}, inplace=True)

        X1 = pd.merge(X1, X2, on = ['churn'])
        X1 = pd.merge(X1,X3, on = ['churn'])
        X1.head()
        X1 = X1.transpose().reset_index()
        X1 = X1.loc[1:]
        X1.columns = ['Feature', 'Not-Churn', 'Churn']
        #X1.head()
        X1['Usage%_During_Churn'] = round((X1['Churn']/(X1['Not-Churn'] + X1['Churn']))*100,2)
        print(X1.head())
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)
        p = sns.barplot(x='Feature', y='Usage%_During_Churn', data=X1)
        p.set_xticklabels(p.get_xticklabels(),rotation=45)
        plt.title('Churn subscriber usage to Total Usage % for {}'.format(col[:-2]), fontsize = 12)
        X1.rename(columns={'Usage%_During_Churn':'Churn_Subscriber_Usage_Trend'}, inplace=True)
        plt.plot(X1['Churn_Subscriber_Usage_Trend'], 'r-')
        #plt.title(title, fontsize = 12)
        ax.legend(loc='upper center', bbox_to_anchor=(0.8, 1.00), shadow=True, ncol=2, fontsize = 10)
        plt.grid(True)
        plt.show()
        
# Method to draw AUC curve
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
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return fpr, tpr, thresholds

# Common Method for Hyperparameter Tuning using Random Forest
def randomforestHyperparameterTuning(parameters, X_train, y_train, n_folds = 5, n_jobs = 4):
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestClassifier

    # instantiate the model
    param = list(parameters.keys())[0]
    if((param == 'max_features') | (param == 'n_estimators')):
        rfc = RandomForestClassifier(max_depth=4)
    else:
        rfc = RandomForestClassifier()
    # fit tree on training data
    rfc = GridSearchCV(rfc, parameters, 
                        cv=n_folds, 
                       scoring="accuracy",
                      return_train_score=True,
                      n_jobs = n_jobs)
    rfc.fit(X_train, y_train)
    #print("Best Parameter ==> {}".format(rfc.best_params_))
    # printing the optimal accuracy score and hyperparameters
    print('We can get accuracy of',rfc.best_score_,'using Best Parameter',rfc.best_params_)
    
    if(len(list(parameters.keys())) == 1):
        scores = rfc.cv_results_
        scoreParam = "param_" + list(parameters.keys())[0]

        plt.figure()
        plt.plot(scores[scoreParam], 
                 scores["mean_train_score"], 
                 label="training accuracy")
        plt.plot(scores[scoreParam], 
                 scores["mean_test_score"], 
                 label="test accuracy")
        plt.xlabel(param)
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

def convertCategorical(contVal, threshold):
    if(contVal > threshold):
        return 1
    else:
        return 0
convertCategorical = np.vectorize(convertCategorical)

# Function to Check the outlier in each Feature for Non-Churn and Churn Subscribers
def featurewiseOutlierBetweenChurnAndNonChurn():
    columnList = (list(df.columns[3:]))
    columnList.remove('churn')
    for count, eachColumn in enumerate(columnList):
        if(count%2 == 0):
            plt.figure(count, figsize=(18,6))
            p = plt.subplot(121)
            sns.boxplot(y = df[eachColumn], x = df['churn'])
            p.set_xticklabels(['Not-Churn','churn'], fontsize= 12)
            p.grid(True)
        else:
            q = plt.subplot(122)
            sns.boxplot(y = df[eachColumn], x = df['churn'])
            q.set_xticklabels(['Not-Churn','churn'], fontsize= 12)
            q.grid(True)

# Common Function to Do the Model Evalution
def modelEvaluation(y_test, y_pred, model, flag = 0):
    print(confusion_matrix(y_test,y_pred))
    print("Accuracy Score ==> {}".format(accuracy_score(y_test,y_pred)))
    print("AUC Score ==> {}".format(roc_auc_score(y_test,y_pred)))
    if flag == 1: #For PCA
        pred_probs_test = model.predict_proba(df_test_pca)[:,1]
    elif flag == 2: #For XGBoost For Imbalance Data
        pred_probs_test = model.predict_proba(np.array(X_test))[:,1]
    elif flag == 3: #For Lasso
        pred_probs_test = lasso.predict(X_test)
    else:
        pred_probs_test = model.predict_proba(X_test)[:,1]
    print("ROC_AUC Score ==> {:2.2}".format(metrics.roc_auc_score(y_test, pred_probs_test)))
    TP = (confusion_matrix(y_test,y_pred))[0][0]
    FP = (confusion_matrix(y_test,y_pred))[0][1]
    FN = (confusion_matrix(y_test,y_pred))[1][0]
    TN = (confusion_matrix(y_test,y_pred))[1][1]
    print("Not-Churn Accuracy Rate:(Specificity) ==> {}".format(TP/(TP+FP)))
    print("Churn Accuracy Rate:(Sensitivity) ==> {}".format(TN/(TN+FN)))
    draw_roc(y_test, y_pred)
    
df.head()
df.columns.values
column_Month_6, column_Month_7, column_Month_8, column_Month_9, column_Common = returnColumnsByMonth(df)

print("Month 6 Columns Count ==> {}".format(len(column_Month_6)))
print("Month 7 Columns Count ==> {}".format(len(column_Month_7)))
print("Month 8 Columns Count ==> {}".format(len(column_Month_8)))
print("Month 9 Columns Count ==> {}".format(len(column_Month_9)))
print("Common Columns Count ==> {}".format(len(column_Common)))
# All Months are having same type of columns So lets see the columns in general
print ("\nMonth based Columns:\n \t\t==> {}".format(np.array(column_Month_6)))
print ("\nCommon Columns:\n \t\t==> {}".format(np.array(column_Common)))
df['Total_Recharge_Amount'] = df['total_rech_amt_6'] + df['total_rech_amt_7']

# Get 70% of "Total Recharge Amount" to identify the recharge Amount Range for High value customer
print(df['Total_Recharge_Amount'].describe(percentiles = [0.7]))
print("\n70% of Total Recharge Amount of first 2 months are {}".format(df['Total_Recharge_Amount'].describe(percentiles = [0.7])[5]))
df = df[df['Total_Recharge_Amount'] > 737].reset_index(drop=True)
print("\nTotal High Value Customer Count ==> {}".format(df.shape[0]))
df.drop(columns=['Total_Recharge_Amount'], inplace=True)
#Get Null Percentage in dataFrame and Filter
nullPercentageLimit = 50
columns_More_Than_50_PercentNull = getColumnsBasedOnNullPercent(df,nullPercentageLimit)
#Drop Columns with More than 50% NUll
df = df.loc[:, ~df.columns.isin(columns_More_Than_50_PercentNull)]

print("\nColumn List Dropped with More than 50% of Null Value:==>\n {}\n".format(columns_More_Than_50_PercentNull))
# Get Columns which have only one value for all the rows.
singleCategoryColumns = df.loc[:, np.array(df.apply(lambda x: x.nunique() == 1))].columns

#Print these single value column names, and the value that they contain.
for eachSingleCatgory in singleCategoryColumns:
    print("{}: {}".format(eachSingleCatgory, df[eachSingleCatgory].unique()))
print("\n<=== Drop Single Category Columns, Other than last_date_of_month_6/7/8/9, as it will be used for Derive Columns ===>\n")
singleCategoryColumns = [x for x in singleCategoryColumns if x not in list(['last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8', 'last_date_of_month_9'])]
singleCategoryColumns = np.array(singleCategoryColumns)
df = df.loc[:, ~df.columns.isin(singleCategoryColumns)]

# Fill the NA value of the last_date_of_month column
df['last_date_of_month_7'] = df['last_date_of_month_7'].fillna('7/31/2014')
df['last_date_of_month_8'] = df['last_date_of_month_8'].fillna('8/31/2014')
df['last_date_of_month_9'] = df['last_date_of_month_9'].fillna('9/30/2014')
#Get the columns where the number of null valued rows are less than 50%
columns_Less_Than_50_PercentNull = getColumnsBasedOnNullPercent(df,nullPercentageLimit, limitType='Lower')
df_temp = df.loc[:, columns_Less_Than_50_PercentNull]

#Get the % of rows that are having null values in each of these columns
round(df_temp.isnull().sum()/len(df_temp.index) * 100,2)
column_Month_6, column_Month_7, column_Month_8, column_Month_9, column_Common = returnColumnsByMonth(df_temp)

print("Month 6 Columns Count ==> {}".format(len(column_Month_6)))
print("Month 7 Columns Count ==> {}".format(len(column_Month_7)))
print("Month 8 Columns Count ==> {}".format(len(column_Month_8)))
print("Month 9 Columns Count ==> {}".format(len(column_Month_9)))
print("Common Columns Count ==> {}".format(len(column_Common)))
print("==> All Months are having same columns with less% of Null Value")
print(np.array(column_Month_7))
df_temp.loc[:, column_Month_7].head()
# 4 Derive Columns for each month, which will tell before how many days from month end, 
#recharge happened by subscriber.
df['days_from_LastRechage_6'] = daysSinceLastRechargeMonthwise(df, 6).apply(lambda x: x.days)
df['days_from_LastRechage_7'] = daysSinceLastRechargeMonthwise(df, 7).apply(lambda x: x.days)
df['days_from_LastRechage_8'] = daysSinceLastRechargeMonthwise(df, 8).apply(lambda x: x.days)
df['days_from_LastRechage_9'] = daysSinceLastRechargeMonthwise(df, 9).apply(lambda x: x.days)
df['days_from_LastRechage_6'] = df['days_from_LastRechage_6'].fillna(30)
df['days_from_LastRechage_7'] = df['days_from_LastRechage_7'].fillna(30)
df['days_from_LastRechage_8'] = df['days_from_LastRechage_8'].fillna(30)
df['days_from_LastRechage_9'] = df['days_from_LastRechage_9'].fillna(30)
#Drop the last Recharge Date and End date of the month columns for all the months
dateColumns = ['last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8','last_date_of_month_9',
              'date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8','date_of_last_rech_9']
df = df.loc[:, ~df.columns.isin(dateColumns)]

df = df.fillna(0).reset_index()
# Label churn and non-churn customers
df['churn'] = np.where(
            (
                (df['total_ic_mou_9'] == 0.0) | 
                (df['total_og_mou_9'] == 0.0)
            ) & 
            (
                (df['vol_2g_mb_9'] == 0.0) & 
                (df['vol_3g_mb_9'] == 0.0)
            ),1,0
        )
# Remove columns with '9'
df = df.drop(df.filter(like = '9').columns, axis=1)
df.groupby(['churn'])['churn'].count()
df['loc_og_mou_Percent_6'] = round((df['loc_og_mou_6']/df['total_og_mou_6']) * 100,2)
df['std_og_mou_Percent_6'] = round((df['std_og_mou_6']/df['total_og_mou_6']) * 100,2)
df['spl_og_mou_Percent_6'] = round((df['spl_og_mou_6']/df['total_og_mou_6']) * 100,2)
df['og_others_Percent_6'] = round((df['og_others_6']/df['total_og_mou_6']) * 100,2)
df['loc_ic_mou_Percent_6'] = round((df['loc_ic_mou_6']/df['total_ic_mou_6']) * 100,2)
df['std_ic_mou_Percent_6'] = round((df['std_ic_mou_6']/df['total_ic_mou_6']) * 100,2)
df['spl_ic_mou_Percent_6'] = round((df['spl_ic_mou_6']/df['total_ic_mou_6']) * 100,2)
df['ic_others_Percent_6'] = round((df['ic_others_6']/df['total_ic_mou_6']) * 100,2)

df['loc_og_mou_Percent_7'] = round((df['loc_og_mou_7']/df['total_og_mou_7']) * 100,2)
df['std_og_mou_Percent_7'] = round((df['std_og_mou_7']/df['total_og_mou_7']) * 100,2)
df['spl_og_mou_Percent_7'] = round((df['spl_og_mou_7']/df['total_og_mou_7']) * 100,2)
df['og_others_Percent_7'] = round((df['og_others_7']/df['total_og_mou_7']) * 100,2)
df['loc_ic_mou_Percent_7'] = round((df['loc_ic_mou_7']/df['total_ic_mou_7']) * 100,2)
df['std_ic_mou_Percent_7'] = round((df['std_ic_mou_7']/df['total_ic_mou_7']) * 100,2)
df['spl_ic_mou_Percent_7'] = round((df['spl_ic_mou_7']/df['total_ic_mou_7']) * 100,2)
df['ic_others_Percent_7'] = round((df['ic_others_7']/df['total_ic_mou_7']) * 100,2)

df['loc_og_mou_Percent_8'] = round((df['loc_og_mou_8']/df['total_og_mou_8']) * 100,2)
df['std_og_mou_Percent_8'] = round((df['std_og_mou_8']/df['total_og_mou_8']) * 100,2)
df['spl_og_mou_Percent_8'] = round((df['spl_og_mou_8']/df['total_og_mou_8']) * 100,2)
df['og_others_Percent_8'] = round((df['og_others_8']/df['total_og_mou_8']) * 100,2)
df['loc_ic_mou_Percent_8'] = round((df['loc_ic_mou_8']/df['total_ic_mou_8']) * 100,2)
df['std_ic_mou_Percent_8'] = round((df['std_ic_mou_8']/df['total_ic_mou_8']) * 100,2)
df['spl_ic_mou_Percent_8'] = round((df['spl_ic_mou_8']/df['total_ic_mou_8']) * 100,2)
df['ic_others_Percent_8'] = round((df['ic_others_8']/df['total_ic_mou_8']) * 100,2)

# Fill All Nan Value because of 0 division set to 0.
df = df.fillna(0).reset_index()
column_Month_6, column_Month_7, column_Month_8, column_Month_9, column_Common = returnColumnsByMonth(df)

print("Month 6 Columns Count ==> {}".format(len(column_Month_6)))
print("Month 7 Columns Count ==> {}".format(len(column_Month_7)))
print("Month 8 Columns Count ==> {}".format(len(column_Month_8)))
print("Month 9 Columns Count ==> {}".format(len(column_Month_9)))
print("Common Columns Count ==> {}".format(len(column_Common)))
print("==> All Months are having same columns with less% of Null Value")
print(np.array(column_Month_7))
df.loc[:, column_Month_7].head()
churnSubscriberUsageChangePercentage()
X1 = df.groupby('churn')['aon'].agg(['mean']).reset_index()
p = sns.barplot(x='churn', y='mean', data=X1)
p.set_xticklabels(['Not-Churn', 'Churn'],rotation=30)
p.set_ylabel('Average Age in Network')
plt.title('Average Age in Network between Churn and Not-Churn subscriber')
plt.show()
featurewiseOutlierBetweenChurnAndNonChurn()
df = df.drop(df.loc[(df['churn'] == 0) & (
    (df['arpu_6'] > 15000) | (df['arpu_7'] > 20000) | (df['arpu_8'] > 20000) | (df['onnet_mou_8'] > 8000) | 
    (df['offnet_mou_7'] > 9000) | (df['offnet_mou_8'] > 10000) | (df['loc_og_t2t_mou_6'] > 6000) | (df['loc_og_t2t_mou_7'] > 5000) |
    (df['loc_og_t2t_mou_8'] > 6000) | (df['loc_og_t2m_mou_6'] > 4000) | (df['loc_og_t2f_mou_7'] > 600) | (df['loc_og_t2f_mou_8'] > 600) |
    (df['loc_og_t2c_mou_8'] > 250) | (df['loc_og_mou_6'] > 8000) | (df['loc_og_mou_7'] > 6000) | (df['loc_og_mou_8'] > 6000) |
    (df['std_og_t2m_mou_8'] > 8000) | (df['std_og_t2f_mou_6'] > 400) | (df['std_og_t2f_mou_7'] > 400) | (df['std_og_t2f_mou_8'] > 400) |
    (df['std_og_mou_8'] > 10000) | (df['spl_og_mou_7'] > 800) | (df['spl_og_mou_8'] > 600) |(df['total_og_mou_8'] > 8000) |
    (df['loc_ic_t2m_mou_8'] > 3000) | (df['loc_ic_t2f_mou_6'] > 1000) |(df['loc_ic_t2f_mou_8'] > 1000) |(df['loc_ic_mou_8'] > 4000) |
    (df['std_ic_t2m_mou_8'] > 3000) | (df['std_ic_t2f_mou_8'] > 800) | (df['std_ic_mou_8'] > 3000) | (df['total_ic_mou_8'] > 4000) |
    (df['isd_ic_mou_7'] > 3000) | (df['isd_ic_mou_8'] > 2000) | (df['ic_others_8'] > 400) | (df['sachet_2g_8'] > 30) | (df['sachet_3g_8'] > 30)
)].index)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.figure(figsize = (40,20))        # Size of the figure
sns.heatmap(df.corr(),annot = True)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
# Importing classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn import metrics
X = (df.iloc[:,3:])
X = X.loc[:,X.columns != 'churn']
y = df.loc[:, 'churn']

#Standardization of Data
scaler = StandardScaler()
scaler.fit(X)
#Using a Train : Test Split of 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print("Number of Features ==> {}".format(len(X.columns)))
lr = LogisticRegression()
lr.fit(X_train, y_train) #Use Balanced Data for Logistic Regression
y_pred = lr.predict(X_test)
modelEvaluation(y_test, y_pred, lr)
#Improting the PCA module
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)
#Doing the PCA on the train data
pca.fit(X_train)

#Making the screeplot - plotting the cumulative variance against the number of components
%matplotlib inline
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title("Scree Plot")
plt.show()
pca.components_
#Using incremental PCA for efficiency
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=25)
df_train_pca = pca_final.fit_transform(X_train)
#creating correlation matrix for the principal components
corrmat = np.corrcoef(df_train_pca.transpose())
#plotting the correlation matrix
%matplotlib inline
plt.figure(figsize = (20,10))
sns.heatmap(corrmat,annot = True)
corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:==> {}".format(corrmat_nodiag.max()), "\n min corr: ==> {}".format(corrmat_nodiag.min()))
df_test_pca = pca_final.transform(X_test)
lr = LogisticRegression()
model_pca = lr.fit(df_train_pca, y_train)

# Making prediction on the test data
y_pred = model_pca.predict(df_test_pca)
# draw_roc(y_test, y_pred)
modelEvaluation(y_test, y_pred, model_pca, 1)
pca_again = PCA(0.95)
df_train_pca = pca_again.fit_transform(X_train)
df_train_pca.shape
learner_pca = LogisticRegression()
model_pca = learner_pca.fit(df_train_pca,y_train)

df_test_pca = pca_again.transform(X_test)
df_test_pca.shape
# #Making prediction on the test data
y_pred = model_pca.predict(df_test_pca)
# draw_roc(y_test, y_pred)
modelEvaluation(y_test, y_pred, model_pca, 1)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
# Making predictions
y_pred = rfc.predict(X_test)
modelEvaluation(y_test, y_pred, rfc)
parameters = {'max_depth': range(2,30, 5)}
randomforestHyperparameterTuning(parameters, X_train, y_train)
# Number of Trees
parameters = {'n_estimators': range(100, 1500, 400)}
#randomforestHyperparameterTuning(parameters)
randomforestHyperparameterTuning(parameters, X_train, y_train)
# Maximum Features to split in a node
parameters = {'max_features': [4, 8, 14, 20, 24, 28, 32]}
#randomforestHyperparameterTuning(parameters)
randomforestHyperparameterTuning(parameters, X_train, y_train)
parameters = {'min_samples_leaf': range(50, 400, 40)}
#randomforestHyperparameterTuning(parameters)
randomforestHyperparameterTuning(parameters, X_train, y_train)
parameters = {'min_samples_split': range(200, 500, 50)}
#randomforestHyperparameterTuning(parameters)
randomforestHyperparameterTuning(parameters, X_train, y_train)
# param_grid = {
#     'max_depth': [4,8,10],
#     'min_samples_leaf': [30, 50 , 70],
#     'min_samples_split': [150, 170, 200],
#     'n_estimators': [100,200, 300], 
#     'max_features': [15,20,25]
# }
# # Create a based model
# rf = RandomForestClassifier()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1,verbose = 1)
# grid_search.fit(X_train, y_train)
# # printing the optimal accuracy score and hyperparameters
# print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=8,
                             min_samples_leaf=50, 
                             min_samples_split=150,
                             max_features=20,
                             n_estimators=100)
# fit
rfc.fit(X_train,y_train)

# Making predictions
y_pred = rfc.predict(X_test)
modelEvaluation(y_test, y_pred, rfc)
from sklearn.svm import SVC
# #SVM
# from sklearn.svm import SVC

# folds = KFold(n_splits = 5, shuffle = True, random_state = 4)
# # specify range of parameters (C) as a list
# params = {"C": [0.1, 1, 10]}

# model = SVC()

# # set up grid search scheme
# # note that we are still using the 5 fold CV scheme we set up earlier
# model_cv = GridSearchCV(estimator = model, param_grid = params, 
#                         scoring= 'accuracy', 
#                         cv = folds, 
#                         verbose = 1,
#                        return_train_score=True,
#                        n_jobs = -1)   
# # fit the model - it will fit 5 folds across all values of C
# model_cv.fit(X_train, y_train)  

# best_score = model_cv.best_score_
# best_C = model_cv.best_params_['C']

# print(" The highest test accuracy is {0} at C = {1}".format(best_score, best_C))
# model_svm = SVC(C = 0.1, probability=True)
# model_svm.fit(X_train, y_train)
# y_pred = model_svm.predict((X_test))
# modelEvaluation(y_test, y_pred, model_svm)
print("Non-Churn Percentage ==> {}".format(df[df['churn'] == 0].shape[0]/df.shape[0]))
print("Churn Percentage ==> {}".format(df[df['churn'] == 1].shape[0]/df.shape[0]))
import xgboost as xgb
from xgboost import XGBClassifier

# hyperparameter tuning with XGBoost

# creating a KFold object 
folds = 3

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


# specify model
xgb_model = XGBClassifier(max_depth=2, n_estimators=200)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True,
                        n_jobs = -1)    
model_cv.fit(X_train, y_train)
model_cv.best_params_
#Execute XGBoost using the best parameter value got from GridSearch Cross  Validation
params = {'learning_rate': 0.2,
          'max_depth': 2, 
          'n_estimators':200,
          'subsample':0.9,
         'objective':'binary:logistic'}

# fit model on training data
model = XGBClassifier(params = params)
model.fit(X_train, y_train)

# # Making predictions
y_pred = model.predict((X_test))
modelEvaluation(y_test, y_pred, model)
from imblearn.over_sampling import SMOTE

print('Before OverSampling, the shape of train_X: {}'.format(X_train.shape))
print('Before OverSampling, the shape of train_y: {} \n'.format(y_train.shape))
print("Before OverSampling, y_train count: '1': Churn ==> {}".format(sum(y_train == 1)))
print("Before OverSampling, y_train count: '0': Not-Churn ==> {}".format(sum(y_train == 0)))


sm = SMOTE()
X_train_sam, y_train_sam = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_sam.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_sam.shape))

print("After OverSampling, count: '1': Churn ==> {}".format(sum(y_train_sam==1)))
print("After OverSampling, count: '0': Not-Churn ==>{}".format(sum(y_train_sam==0)))
lr = LogisticRegression()
lr.fit(X_train_sam, y_train_sam) #Use Balanced Data for Logistic Regression
y_pred = lr.predict(X_test)
modelEvaluation(y_test, y_pred,lr)
#Use the Same Hyper parameter got in last execution
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=8,
                             min_samples_leaf=50, 
                             min_samples_split=150,
                             max_features=20,
                             n_estimators=100)
# fit
rfc.fit(X_train_sam,y_train_sam)

# Making predictions
y_pred = rfc.predict(X_test)
modelEvaluation(y_test, y_pred,rfc)
# model_svm = SVC(C = 0.1, probability=True)
# model_svm.fit(X_train_sam, y_train_sam)
# y_pred = model_svm.predict((X_test))
# modelEvaluation(y_test, y_pred,model_svm)
pca_again = PCA(0.99)
df_train_pca = pca_again.fit_transform(X_train_sam)
df_train_pca.shape
learner_pca = LogisticRegression()
model_pca = learner_pca.fit(df_train_pca,y_train_sam)

df_test_pca = pca_again.transform(X_test)
df_test_pca.shape
# #Making prediction on the test data
y_pred = model_pca.predict(df_test_pca)
modelEvaluation(y_test, y_pred, model_pca, 1)
from sklearn.linear_model import Lasso
# hide warnings
import warnings
warnings.filterwarnings('ignore')

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = 5, 
                        return_train_score=True,
                        verbose = 1,
                        n_jobs = -1)            

model_cv.fit(X_train, y_train)
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
print("Best Alpha Value ==> {} ".format(model_cv.best_params_))
alpha =0.01
lasso = Lasso(alpha=alpha)
lasso.fit(X_train, y_train)


y_pred = lasso.predict(X_test)
y_pred = convertCategorical(y_pred, 0.5)
modelEvaluation(y_test, y_pred, lasso, 3)
#Get Features list and co-efficient values from the Lasso Regression and make a single dataframe
s1 = pd.DataFrame(np.insert(np.array(X.columns),0,"constant"), columns=['feature'])
s1.reset_index(drop = True, inplace = True)

s2 = pd.DataFrame(np.insert(np.array(lasso.coef_), 0, lasso.intercept_), columns=['Values'])
s2.reset_index(drop = True, inplace= True)

s2['Values'] = s2['Values'].apply(lambda x: round(x,3))
drivingFeaturedf = pd.concat([s1,s2], axis=1)
drivingFeaturedf = drivingFeaturedf.iloc[1:]
drivingFeaturedf.reset_index(drop= True, inplace = True)
# Draw the complete date frame of features and coefficient.
import matplotlib.pyplot as plt
plt.figure(figsize=(26,10))
plt.subplot(111)
ax1 = sns.barplot(x = drivingFeaturedf['feature'], y = drivingFeaturedf['Values'])
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90, fontsize= 9)
ax1.set_yticklabels(ax1.get_yticklabels(), fontsize = 15)
plt.ylabel('Co-efficient', fontsize = 10)
plt.show()
#Drop features with 0 coefficient.
drivingFeaturedf = drivingFeaturedf.loc[drivingFeaturedf['Values'] != 0]
#Draw the plot for the features with non-zero co-efficient
import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
plt.subplot(111)
ax1 = sns.barplot(x = drivingFeaturedf['feature'], y = drivingFeaturedf['Values'])
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90, fontsize=10)
plt.ylabel('Co-efficient')
plt.show()
print("Driving Feature List: ==>\n \t\t\t{}".format(np.array(drivingFeaturedf['feature'])))
