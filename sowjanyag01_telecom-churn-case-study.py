# Ignoring warning messages

import warnings

warnings.filterwarnings('ignore')



# Import the required library

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_columns',230)
# reading the input data and preview

churn= pd.read_csv(r'../input/telecom_churn_data.csv')

churn.head()
print (churn.shape)

print (churn.info())

churn.describe()
#  unique number of custormers from the data

print ("Unique customers in the dataset: %d"%len(churn.mobile_number.unique()))
#Columns in the dataset

pd.DataFrame(churn.columns)
def getMissingValuesPercentage(missingCutoff):

    missing = round(100*(churn.isnull().sum()/churn.shape[0]))

    print("There are {} features having more than {}% missing values/entries".format(len(missing.loc[missing > missingCutoff]),missingCutoff))

    missing_churn_data_df =missing.loc[missing > missingCutoff]

    return missing_churn_data_df
# Function impute the NAN with 0

# argument: colList, list of columns for which nan is to be replaced with 0

def imputeNan(data,imputeColList=False,missingColList=False): 

    if imputeColList:

        for col in [y + s for s in ['_6','_7','_8','_9'] for y in imputeColList]:

            data[col].fillna(0, inplace=True)

    else:    

        for col in missingColList:

            data[col].fillna(0, inplace=True)
# Missing values per column expressed as % of total number of values

getMissingValuesPercentage(50)
#av_rech_amt_data_* features can be important for getting the high-value customers,

#Impute the missing av_rech_amt_data_* with 0

imputeCol = ['av_rech_amt_data', 'arpu_2g', 'arpu_3g', 'count_rech_2g', 'count_rech_3g',

             'max_rech_data', 'total_rech_data','fb_user','night_pck_user']

imputeNan(churn,imputeCol)
getMissingValuesPercentage(50)
# dropping the columns having more than 50% missing values

missingcol = list(getMissingValuesPercentage(50).index)

churn.drop(missingcol,axis=1,inplace=True)

churn.shape
# More than 5 % missing values

getMissingValuesPercentage(5) 
# checking if all these above features go missing together since they have the same 8% missing values in each feature.

missingcol = list(getMissingValuesPercentage(5).index)

print ("There are %d customers having missing values for %s together"%(len(churn[churn[missingcol].isnull().all(axis=1)]),missingcol))

churn[churn[missingcol].isnull().all(axis=1)][missingcol].head()
imputeNan(churn,missingColList=missingcol)
churn=churn[~churn[missingcol].isnull().all(axis=1)]

churn.shape
getMissingValuesPercentage(2)
missingcol = list(getMissingValuesPercentage(2).index)

print ("There are %d customers having missing values for %s together"%(len(churn[churn[missingcol].isnull().all(axis=1)]),missingcol))

churn[churn[missingcol].isnull().all(axis=1)][missingcol].head()
churn=churn[~churn[missingcol].isnull().all(axis=1)]

churn.shape
missingcol.remove('date_of_last_rech_8')

missingcol.remove('date_of_last_rech_9')

imputeNan(churn,missingColList=missingcol)
getMissingValuesPercentage(0)
col = ['loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou','last_date_of_month_7','last_date_of_month_8','last_date_of_month_9', 'date_of_last_rech_7', 'date_of_last_rech_8', 'date_of_last_rech_9']

for c in col: 

    print("Unique values in column %s are %s" % (c,churn[c].unique()))

                                                 
#Some of these features take only one value. Lets impute their missing values in these features with the mode

col = ['loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou','last_date_of_month_7','last_date_of_month_8','last_date_of_month_9']

for c in col:

    print(churn[c].value_counts())

    churn[c].fillna(churn[c].mode()[0], inplace=True)

print("Impute the missing values mode value")

getMissingValuesPercentage(0)
# All these features are missing together

missingcol = list(getMissingValuesPercentage(0).index)

print ("There are %d rows in total having missing values for these variables."%(len(churn[churn[missingcol].isnull().all(axis=1)])))
churn[churn['date_of_last_rech_6'].isnull()]['date_of_last_rech_6'] = '6/30/2014'

churn[churn['date_of_last_rech_7'].isnull()]['date_of_last_rech_7'] = '7/31/2014'

churn[churn['date_of_last_rech_8'].isnull()]['date_of_last_rech_8'] = '8/31/2014'

churn[churn['date_of_last_rech_9'].isnull()]['date_of_last_rech_9'] = '9/30/2014'
zero_columns=churn.columns[(churn == 0).all()]

print ("There are {} features which has only 0 as values. These features are \n{}".format(len(zero_columns),zero_columns))
# All take a same value '0'. 

churn.drop(zero_columns,axis=1,inplace=True)
#Final dataset after treating Missing values

churn.shape
churn.reset_index(inplace=True,drop=True)

# list of all columns which store date

date_columns = list(churn.filter(regex='date').columns)

date_columns
# Converting dtype of date columns to datetime

for col in date_columns:

    churn[col] = pd.to_datetime(churn[col], format='%m/%d/%Y')
churn.info()
# renaming columns,

#'jun_vbc_3g' : 'vbc_3g_6'

#'jul_vbc_3g' : 'vbc_3g_7'

#'aug_vbc_3g' : 'vbc_3g_8'

#'sep_vbc_3g' : 'vbc_3g_9'

churn.rename(columns={'jun_vbc_3g' : 'vbc_3g_6', 'jul_vbc_3g' : 'vbc_3g_7', 'aug_vbc_3g' : 'vbc_3g_8',

                      'sep_vbc_3g' : 'vbc_3g_9'}, inplace=True)
#Creating new feature: 'vol_data_mb_6', 'vol_data_mb_7', 'vol_data_mb_8', 'vol_data_mb_9',

for i in range(6,10):

    churn['vol_data_mb_'+str(i)] = (churn['vol_2g_mb_'+str(i)]+churn['vol_3g_mb_'+str(i)]).astype(int)
rechcol = churn.filter(regex=('count')).columns

churn[rechcol].head()
# Creating new feature: avg_rech_amt_6,avg_rech_amt_7,avg_rech_amt_8,avg_rech_amt_9

for i in range(6,10):

    churn['avg_rech_amt_'+str(i)] = round(churn['total_rech_amt_'+str(i)]/churn['total_rech_num_'+str(i)]+1,2)
imputeNan(churn,missingColList=['avg_rech_amt_6','avg_rech_amt_7','avg_rech_amt_8','avg_rech_amt_9'])
#Creating new feature: total_rech_num_data_6,total_rech_num_data_7,total_rech_num_data_8,total_rech_num_data_9

for i in range(6,10):

    churn['total_rech_num_data_'+str(i)] = (churn['count_rech_2g_'+str(i)]+churn['count_rech_3g_'+str(i)]).astype(int)
#Creating new feature: total_rech_amt_data_6,total_rech_amt_data_7,total_rech_amt_data_8,total_rech_amt_data_9

for i in range(6,10):

    churn['total_rech_amt_data_'+str(i)] = churn['total_rech_num_data_'+str(i)]*churn['av_rech_amt_data_'+str(i)]
#Creating new feature: total_mon_rech_6,total_mon_rech_7,total_mon_rech_8,total_mon_rech_9

for i in range(6,10):

    churn['total_month_rech_'+str(i)] = churn['total_rech_amt_'+str(i)]+churn['total_rech_amt_data_'+str(i)]

churn.filter(regex=('total_month_rech')).head()
# calculating the avegare of first two months (good phase) total monthly recharge amount

avg_goodPhase =(churn.total_month_rech_6 + churn.total_month_rech_7)/2

# finding the cutoff which is the 70th percentile of the good phase average recharge amounts

hv_cutoff= np.percentile(avg_goodPhase,70)

# Filtering the users whose good phase avg. recharge amount >= to the cutoff of 70th percentile.

hv_users = churn[avg_goodPhase >=  hv_cutoff]

hv_users.reset_index(inplace=True,drop=True)



print("Number of High-Value Customers in the Dataset: %d\n"% len(hv_users))

print("Percentage High-value users in data : {}%".format(round(len(hv_users)/churn.shape[0]*100),2))
def getChurnStatus(data,churnPhaseMonth=9):

    # Function to tag customers as churners (churn=1, else 0) based on 'vol_2g_mb_','vol_3g_mb_','total_ic_mou_','total_og_mou_'

    #argument: churnPhaseMonth, indicating the month number to be used to define churn (default= 9)

    churn_features= ['vol_2g_mb_','vol_3g_mb_','total_ic_mou_','total_og_mou_']

    flag = ~data[[s + str(churnPhaseMonth) for s in churn_features ]].any(axis=1)

    flag = flag.map({True:1, False:0})

    return flag
hv_users['churn'] = getChurnStatus(hv_users,9)

print("There are {} users tagged as churners out of {} High-Value Customers.".format(len(hv_users[hv_users.churn == 1]),hv_users.shape[0]))

print("High-value Churn Percentage : {}%".format(round(len(hv_users[hv_users.churn == 1])/hv_users.shape[0] *100,2)))

def plot_hist(dataset,col,binsize):

    fig, ax = plt.subplots(figsize=(20,4))

    counts, bins, patches = ax.hist(dataset[col],bins=range(0,dataset[col].max(),round(binsize)), facecolor='lightgreen', edgecolor='gray')

    

    ax.set_xticks(bins)

    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for count, x in zip(counts, bin_centers):

        # Label the percentages

        percent = '%0.0f%%' % (100 * float(count) / counts.sum())

        ax.annotate(percent, xy=(x,0.2), xycoords=('data', 'axes fraction'),

        xytext=(0, -32), textcoords='offset points', va='top', ha='center')

    

    ax.set_xlabel(col.upper())

    ax.set_ylabel('Count')

    plt.show()

    
def plot_avgMonthlyCalls(pltType,data,calltype,colList):

    # style

    plt.style.use('seaborn-darkgrid')

    # create a color palette

    palette = plt.get_cmap('Set1')

    

    if pltType == 'multi':

        #Create dataframe after grouping on AON with colList features

        total_call_mou = pd.DataFrame(data.groupby('aon_bin',as_index=False)[colList].mean())

        total_call_mou['aon_bin']=pd.to_numeric(total_call_mou['aon_bin'])

        total_call_mou

        # multiple line plot

        num=0

        fig, ax = plt.subplots(figsize=(15,8))

        for column in total_call_mou.drop('aon_bin', axis=1):

            num+=1

            ax.plot(total_call_mou['aon_bin'] , total_call_mou[column], marker='', color=palette(num), linewidth=2, alpha=0.9, label=column)

         

        ## Add legend

        plt.legend(loc=2, ncol=2)

        ax.set_xticks(total_call_mou['aon_bin'])

        

        # Add titles

        plt.title("Avg.Monthly "+calltype+" MOU  V/S AON", loc='left', fontsize=12, fontweight=0, color='orange')

        plt.xlabel("Aon (years)")

        plt.ylabel("Avg. Monthly "+calltype+" MOU")

    elif pltType == 'single':

        fig, ax = plt.subplots(figsize=(8,4))

        ax.plot(data[colList].mean())

        ax.set_xticklabels(['Jun','Jul','Aug','Sep'])

        

        # Add titles

        plt.title("Avg. "+calltype+" MOU  V/S Month", loc='left', fontsize=12, fontweight=0, color='orange')

        plt.xlabel("Month")

        plt.ylabel("Avg. "+calltype+" MOU")

        

    plt.show()
def plot_byChurnMou(colList,calltype):

    fig, ax = plt.subplots(figsize=(7,4))

    df=hv_users.groupby(['churn'])[colList].mean().T

    plt.plot(df)

    ax.set_xticklabels(['Jun','Jul','Aug','Sep'])

    ## Add legend

    plt.legend(['Non-Churn', 'Churn'])

    # Add titles

    plt.title("Avg. "+calltype+" MOU  V/S Month", loc='left', fontsize=12, fontweight=0, color='orange')

    plt.xlabel("Month")

    plt.ylabel("Avg. "+calltype+" MOU")
def plot_byChurn(data,col):

    # per month churn vs Non-Churn

    fig, ax = plt.subplots(figsize=(7,4))

    colList=list(data.filter(regex=(col)).columns)

    colList = colList[:3]

    plt.plot(hv_users.groupby('churn')[colList].mean().T)

    ax.set_xticklabels(['Jun','Jul','Aug','Sep'])

    ## Add legend

    plt.legend(['Non-Churn', 'Churn'])

    # Add titles

    plt.title( str(col) +" V/S Month", loc='left', fontsize=12, fontweight=0, color='orange')

    plt.xlabel("Month")

    plt.ylabel(col)

    plt.show()

    # Numeric stats for per month churn vs Non-Churn

    return hv_users.groupby('churn')[colList].mean()
# Filtering the common monthly columns for each month

comcol = hv_users.filter(regex ='_6').columns

monthlycol = [item.strip('_6') for item in comcol]

monthlycol
# getting the number of monthly columns and profile columns

print ("Total number of columns in data :", hv_users.shape[1] )

print ("Number of columns for each month : ",len(monthlycol))

print ("Total monthly columns among the orignal columns (%d*4): %d"%(len(monthlycol), len(monthlycol) * 4))

print ("Columns other than monthly columns :", hv_users.shape[1] - (len(monthlycol) * 4))
#Remove all the attributes corresponding to the churn phase (all attributes having ‘ _9’, etc. in their names).

col_9List = hv_users.filter(regex=('_9')).columns

hv_users.drop(col_9List,axis=1,inplace=True)
# list of all the monthly columns 6,7,8,9

allmonthlycol = [x + s for s in ['_6','_7','_8'] for x in monthlycol]

allmonthlycol
# list of column which are not monthly columns

nonmonthlycol = [col for col in hv_users.columns if col not in allmonthlycol]

nonmonthlycol
# Getting the distinct circle_id's in the data

hv_users.circle_id.value_counts()
hv_users.drop('circle_id',axis=1,inplace=True)
# Customers distribution of the age on network

print(hv_users.aon.describe())

plot_hist(hv_users,'aon',365)
#Create Derived categorical variable

hv_users['aon_bin'] = pd.cut(churn['aon'], range(0,churn['aon'].max(),365), labels=range(0,int(round(churn['aon'].max()/365))-1))
hv_users['aon_bin'].head()
# Plotting Avg. total monthly incoming MOU vs AON

ic_col = hv_users.filter(regex ='total_ic_mou').columns

plot_avgMonthlyCalls('single',hv_users,calltype='incoming',colList=ic_col)

plot_avgMonthlyCalls('multi',hv_users,calltype='incoming',colList=ic_col)
# Plotting Avg. total monthly outgoing MOU vs AON

og_col = hv_users.filter(regex ='total_og_mou').columns

plot_avgMonthlyCalls('single',hv_users,calltype='outgoing',colList=og_col)

plot_avgMonthlyCalls('multi',hv_users,calltype='outgoing',colList=og_col)
ic_col = ['total_ic_mou_6','total_ic_mou_7','total_ic_mou_8']

og_col = ['total_og_mou_6','total_og_mou_7','total_og_mou_8']

plot_byChurnMou(ic_col,'Incoming')

plot_byChurnMou(og_col,'Outgoing')
# Avg.Incoming MOU per month churn vs Non-Churn

hv_users.groupby(['churn'])['total_ic_mou_6','total_ic_mou_7','total_ic_mou_8'].mean()
# Avg. Outgoing MOU per month churn vs Non-Churn

hv_users.groupby(['churn'])['total_og_mou_6','total_og_mou_7','total_og_mou_8'].mean()
#Creating new feature: og_to_ic_mou_6, og_to_ic_mou_7, og_to_ic_mou_8

# adding 1 to denominator to avoid dividing by 0 and getting nan values.

for i in range(6,9):

    hv_users['og_to_ic_mou_'+str(i)] = (hv_users['total_og_mou_'+str(i)])/(hv_users['total_ic_mou_'+str(i)]+1)
plot_byChurn(hv_users,'og_to_ic_mou')
#Create new feature: loc_og_to_ic_mou_6, loc_og_to_ic_mou_7, loc_og_to_ic_mou_8

# adding 1 to denominator to avoid dividing by 0 and getting nan values.

for i in range(6,9):

    hv_users['loc_og_to_ic_mou_'+str(i)] = (hv_users['loc_og_mou_'+str(i)])/(hv_users['loc_ic_mou_'+str(i)]+1)
plot_byChurn(hv_users,'loc_og_to_ic_mou')
plot_byChurn(hv_users,'vol_data_mb')
plot_byChurn(hv_users,'total_month_rech')
plot_byChurn(hv_users,'max_rech_amt')
plot_byChurn(hv_users,'arpu')
#Create new feature: Total_loc_mou_6,Total_loc_mou_7,lTotal_loc_mou_8

for i in range(6,9):

    hv_users['Total_loc_mou_'+str(i)] = (hv_users['loc_og_mou_'+str(i)])+(hv_users['loc_ic_mou_'+str(i)])
plot_byChurn(hv_users,'Total_loc_mou_')
#Create new feature: Total_roam_mou_6,Total_roam_mou_7,Total_roam_mou_8

for i in range(6,9):

    hv_users['Total_roam_mou_'+str(i)] = (hv_users['roam_ic_mou_'+str(i)])+(hv_users['roam_og_mou_'+str(i)])
plot_byChurn(hv_users,'Total_roam_mou')
plot_byChurn(hv_users,'last_day_rch_amt')
import sklearn.preprocessing

from sklearn import metrics

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
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
def getModelMetrics(actual_churn=False,pred_churn=False):



    confusion = metrics.confusion_matrix(actual_churn, pred_churn)



    TP = confusion[1,1] # true positive 

    TN = confusion[0,0] # true negatives

    FP = confusion[0,1] # false positives

    FN = confusion[1,0] # false negatives



    print("Roc_auc_score : {}".format(metrics.roc_auc_score(actual_churn,pred_churn)))

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

    print('sklearn precision score value: {}'.format(metrics.precision_score(actual_churn, pred_churn )))

    

    
def predictChurnWithProb(model,X,y,prob):

    # Funtion to predict the churn using the input probability cut-off

    # Input arguments: model instance, x and y to predict using model and cut-off probability

    

    # predict

    pred_probs = model.predict_proba(X)[:,1]

    

    y_df= pd.DataFrame({'churn':y, 'churn_Prob':pred_probs})

    # Creating new column 'predicted' with 1 if Churn_Prob>0.5 else 0

    y_df['final_predicted'] = y_df.churn_Prob.map( lambda x: 1 if x > prob else 0)

    # Let's see the head

    getModelMetrics(y_df.churn,y_df.final_predicted)

    return y_df
def findOptimalCutoff(df):

    #Function to find the optimal cutoff for classifing as churn/non-churn

    # Let's create columns with different probability cutoffs 

    numbers = [float(x)/10 for x in range(10)]

    for i in numbers:

        df[i] = df.churn_Prob.map( lambda x: 1 if x > i else 0)

    #print(df.head())

    

    # Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

    cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

    from sklearn.metrics import confusion_matrix

    

    # TP = confusion[1,1] # true positive 

    # TN = confusion[0,0] # true negatives

    # FP = confusion[0,1] # false positives

    # FN = confusion[1,0] # false negatives

    

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
def modelfit(alg, X_train, y_train, performCV=True, cv_folds=5):

    #Fit the algorithm on the data

    alg.fit(X_train, y_train)

        

    #Predict training set:

    dtrain_predictions = alg.predict(X_train)

    dtrain_predprob = alg.predict_proba(X_train)[:,1]

    

    #Perform cross-validation:

    if performCV:

        cv_score = cross_val_score(alg, X_train, y_train, cv=cv_folds, scoring='roc_auc')

    

    #Print model report:

    print ("\nModel Report")

    print ("Accuracy : %.4g" % metrics.roc_auc_score(y_train, dtrain_predictions))

    print ("Recall/Sensitivity : %.4g" % metrics.recall_score(y_train, dtrain_predictions))

    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))

    

    if performCV:

        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

        
# creating copy of the final hv_user dataframe

hv_users_PCA = hv_users.copy()

# removing the columns not required for modeling

hv_users_PCA.drop(['mobile_number', 'aon_bin'], axis=1, inplace=True)
# removing the datatime columns before PCA

dateTimeCols = list(hv_users_PCA.select_dtypes(include=['datetime64']).columns)

print(dateTimeCols)

hv_users_PCA.drop(dateTimeCols, axis=1, inplace=True)
from sklearn.model_selection import train_test_split



#putting features variables in X

X = hv_users_PCA.drop(['churn'], axis=1)



#putting response variables in Y

y = hv_users_PCA['churn']    



# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# fitting and transforming the scaler on train

X_train = scaler.fit_transform(X_train)

# transforming the train using the already fit scaler

X_test = scaler.transform(X_test)
#Improting the PCA module

from sklearn.decomposition import PCA

pca = PCA(svd_solver='randomized', random_state=42)
#Doing the PCA on the train data

pca.fit(X_train)
pca.explained_variance_ratio_[:50]
#Making the screeplot - plotting the cumulative variance against the number of components

%matplotlib inline

fig = plt.figure(figsize = (12,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
#Using incremental PCA for efficiency - saves a lot of time on larger datasets

from sklearn.decomposition import IncrementalPCA

pca_final = IncrementalPCA(n_components=35)
X_train_pca = pca_final.fit_transform(X_train)

X_train_pca.shape
#creating correlation matrix for the principal components

corrmat = np.corrcoef(X_train_pca.transpose())

# 1s -> 0s in diagonals

corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())

print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)

# we see that correlations are indeed very close to 0
#Applying selected components to the test data - 50 components

X_test_pca = pca_final.transform(X_test)

X_test_pca.shape
#Training the model on the train data

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



lr0 = LogisticRegression(class_weight='balanced')
modelfit(lr0, X_train_pca, y_train)
# predictions on Test data

pred_probs_test = lr0.predict(X_test_pca)

getModelMetrics(y_test,pred_probs_test)
print("Accuracy : {}".format(metrics.accuracy_score(y_test,pred_probs_test)))

print("Recall : {}".format(metrics.recall_score(y_test,pred_probs_test)))

print("Precision : {}".format(metrics.precision_score(y_test,pred_probs_test)))
#Making prediction on the test data

pred_probs_train = lr0.predict_proba(X_train_pca)[:,1]

print("roc_auc_score(Train) {:2.2}".format(metrics.roc_auc_score(y_train, pred_probs_train)))
cut_off_prob=0.5

y_train_df = predictChurnWithProb(lr0,X_train_pca,y_train,cut_off_prob)

y_train_df.head()
draw_roc(y_train_df.churn, y_train_df.final_predicted)
#draw_roc(y_pred_final.Churn, y_pred_final.predicted)

print("roc_auc_score : {:2.2f}".format(metrics.roc_auc_score(y_train_df.churn, y_train_df.final_predicted)))
# finding cut-off with the right balance of the metrices

# sensitivity vs specificity trade-off

findOptimalCutoff(y_train_df)
# predicting with the choosen cut-off on train

cut_off_prob = 0.52

predictChurnWithProb(lr0,X_train_pca,y_train,cut_off_prob)
# predicting with the choosen cut-off on test

predictChurnWithProb(lr0,X_test_pca,y_test,cut_off_prob)
dt0 = DecisionTreeClassifier(class_weight='balanced',

                             max_features='auto',

                             min_samples_split=100,

                             min_samples_leaf=100,

                             max_depth=6,

                             random_state=10)

modelfit(dt0, X_train_pca, y_train)
# make predictions

pred_probs_test = dt0.predict(X_test_pca)

#Let's check the model metrices.

getModelMetrics(actual_churn=y_test,pred_churn=pred_probs_test)
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': range(5,15,3),

    'min_samples_leaf': range(100, 400, 50),

    'min_samples_split': range(100, 400, 100),

    'max_features': [8,10,15]

}

# Create a based model

dt = DecisionTreeClassifier(class_weight='balanced',random_state=10)

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = dt, param_grid = param_grid, 

                          cv = 3, n_jobs = 4,verbose = 1,scoring="f1_weighted")
# Fit the grid search to the data

grid_search.fit(X_train_pca, y_train)
# printing the optimal accuracy score and hyperparameters

print('We can get recall of',grid_search.best_score_,'using',grid_search.best_params_)
# model with the best hyperparameters

dt_final = DecisionTreeClassifier(class_weight='balanced',

                             max_depth=14,

                             min_samples_leaf=100, 

                             min_samples_split=100,

                             max_features=15,

                             random_state=10)
modelfit(dt_final,X_train_pca,y_train)
# make predictions

pred_probs_test = dt_final.predict(X_test_pca)

#Let's check the model metrices.

getModelMetrics(actual_churn=y_test,pred_churn=pred_probs_test)
# classification report

print(classification_report(y_test,pred_probs_test))
# predicting churn with default cut-off 0.5

cut_off_prob = 0.5

y_train_df = predictChurnWithProb(dt_final,X_train_pca,y_train,cut_off_prob)

y_train_df.head()
# finding cut-off with the right balance of the metrices

findOptimalCutoff(y_train_df)
# predicting churn with cut-off 0.56

cut_off_prob=0.55

y_train_df = predictChurnWithProb(dt_final,X_train_pca,y_train,cut_off_prob)

y_train_df.head()
#Lets see how it performs on test data.

y_test_df= predictChurnWithProb(dt_final,X_test_pca,y_test,cut_off_prob)

y_test_df.head()
def plot_traintestAcc(score,param):

    scores = score

    # plotting accuracies with max_depth

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
parameters = {'max_depth': range(10, 30, 5)}

rf0 = RandomForestClassifier()

rfgs = GridSearchCV(rf0, parameters, 

                    cv=5, 

                   scoring="f1")

rfgs.fit(X_train_pca,y_train)
scores = rfgs.cv_results_

scores
parameters = {'n_estimators': range(50, 150, 25)}

rf1 = RandomForestClassifier(max_depth=20,random_state=10)

rfgs = GridSearchCV(rf1, parameters, 

                    cv=3, 

                   scoring="recall")
rfgs.fit(X_train_pca,y_train)
parameters = {'max_features': [4, 8, 14, 20, 24]}

rf3 = RandomForestClassifier(max_depth=20,n_estimators=80,random_state=10)

rfgs = GridSearchCV(rf3, parameters, 

                    cv=5, 

                   scoring="f1")
rfgs.fit(X_train_pca,y_train)
parameters = {'min_samples_leaf': range(100, 400, 50)}

rf4 = RandomForestClassifier(max_depth=20,n_estimators=80,max_features=5,random_state=10)

rfgs = GridSearchCV(rf4, parameters, 

                    cv=3, 

                   scoring="f1")
rfgs.fit(X_train_pca,y_train)
parameters = {'min_samples_split': range(50, 300, 50)}

rf5 = RandomForestClassifier(max_depth=20,n_estimators=80,max_features=5,min_samples_leaf=100,random_state=10)

rfgs = GridSearchCV(rf5, parameters, 

                    cv=3, 

                   scoring="f1")
rfgs.fit(X_train_pca,y_train)
rf_final = RandomForestClassifier(max_depth=20,

                                  n_estimators=80,

                                  max_features=5,

                                  min_samples_leaf=100,

                                  min_samples_split=50,

                                  random_state=10)
print("Model performance on Train data:")

modelfit(rf_final,X_train_pca,y_train)
# predict on test data

predictions = rf_final.predict(X_test_pca)
print("Model performance on Test data:")

getModelMetrics(y_test,predictions)
# predicting churn with default cut-off 0.5

cut_off_prob=0.5

y_train_df = predictChurnWithProb(rf_final,X_train_pca,y_train,cut_off_prob)

y_train_df.head()
# finding cut-off with the right balance of the metrices

findOptimalCutoff(y_train_df)
cut_off_prob=0.45

predictChurnWithProb(rf_final,X_train_pca,y_train,cut_off_prob)
y_test_df= predictChurnWithProb(rf_final,X_test_pca,y_test,cut_off_prob)

y_test_df.head()
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm

# Fitting the default GradientBoostingClassifier

gbm0 = GradientBoostingClassifier(random_state=10)

modelfit(gbm0, X_train_pca, y_train)
# Hyperparameter tuning for n_estimators

param_test1 = {'n_estimators':range(20,150,10)}

gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 

param_grid = param_test1, scoring='f1',n_jobs=4,iid=False, cv=3)

gsearch1.fit(X_train_pca, y_train)
gsearch1.best_params_, gsearch1.best_score_
# Hyperparameter tuning for max_depth and min_sample_split

param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}

gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=140, max_features='sqrt', subsample=0.8, random_state=10), 

param_grid = param_test2, scoring='f1',n_jobs=4,iid=False, cv=3)

gsearch2.fit(X_train_pca, y_train)
gsearch2.best_params_, gsearch2.best_score_
# Hyperparameter tuning for min_sample_leaf

param_test3 = {'min_samples_leaf':range(30,71,10)}

gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=140,max_depth=15,min_samples_split=200, max_features='sqrt', subsample=0.8, random_state=10), 

param_grid = param_test3, scoring='f1',n_jobs=4,iid=False, cv=3)

gsearch3.fit(X_train_pca, y_train)
gsearch3.best_params_, gsearch3.best_score_
# Hyperparameter tuning for max_features

param_test4 = {'max_features':range(7,20,2)}

gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=140,max_depth=15, min_samples_split=200, min_samples_leaf=30, subsample=0.8, random_state=10),

param_grid = param_test4, scoring='f1',n_jobs=4,iid=False, cv=3)

gsearch4.fit(X_train_pca, y_train)
gsearch4.best_params_, gsearch4.best_score_
# Tunned GradientBoostingClassifier

gbm_final = GradientBoostingClassifier(learning_rate=0.1, n_estimators=140,max_features=15,max_depth=15, min_samples_split=200, min_samples_leaf=40, subsample=0.8, random_state=10)

modelfit(gbm_final, X_train_pca, y_train)
# predictions on Test data

dtest_predictions = gbm_final.predict(X_test_pca)
# model Performance on test data

getModelMetrics(y_test,dtest_predictions)
# predicting churn with default cut-off 0.5

cut_off_prob=0.5

y_train_df = predictChurnWithProb(gbm_final,X_train_pca,y_train,cut_off_prob)

y_train_df.head()
findOptimalCutoff(y_train_df)
cut_off_prob=0.25

predictChurnWithProb(gbm_final,X_train_pca,y_train,cut_off_prob)
y_test_df= predictChurnWithProb(gbm_final,X_test_pca,y_test,cut_off_prob)

y_test_df.head()
# instantiate an object of class SVC()

# note that we are using cost C=1

svm0 = SVC(C = 1)
# fit

svm0.fit(X_train_pca, y_train)



# predict on train

y_pred = svm0.predict(X_train_pca)

getModelMetrics(y_train,y_pred)
# Predict on test

y_pred = svm0.predict(X_test_pca)

getModelMetrics(y_test,y_pred)
# specify range of parameters (C) as a list

params = {"C": [0.1, 1, 10, 100, 1000]}



svm1 = SVC()



# set up grid search scheme

# note that we are still using the 5 fold CV scheme

model_cv = GridSearchCV(estimator = svm1, param_grid = params, 

                        scoring= 'f1', 

                        cv = 5, 

                        verbose = 1,

                        n_jobs=4,

                       return_train_score=True) 

model_cv.fit(X_train_pca, y_train)
plot_traintestAcc(model_cv.cv_results_,'C')
model_cv.best_params_
svm_final = SVC(C = 1000)

# fit

svm_final.fit(X_train_pca, y_train)
# predict

y_pred = svm_final.predict(X_test_pca)
getModelMetrics(y_test,y_pred)
svm_k = SVC(C = 1000, kernel='rbf')

svm_k.fit(X_train_pca, y_train)
y_pred = svm_k.predict(X_test_pca)
getModelMetrics(y_test,y_pred)


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

grid_search.fit(X_train, y_train)
rf = RandomForestClassifier(max_depth=12,

                            max_features=20,

                            min_samples_leaf=100,

                            min_samples_split=200,

                            n_estimators=300,

                            random_state=10)
rf.fit(X_train, y_train)
plt.figure(figsize=(15,40))

feat_importances = pd.Series(rf.feature_importances_, index=X.columns)

feat_importances.nlargest(len(X.columns)).sort_values().plot(kind='barh', align='center')