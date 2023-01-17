# Supress Warnings
import warnings
warnings.filterwarnings('ignore')
# Importing Standard required packages

# Import the numpy and pandas packages
import numpy as np
import pandas as pd

# Import the matplotlib and seaborn packages to plot different types of charts
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Set number of rows and columns to be displayed
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Importing dataset
telecom_churn = pd.read_csv('../input/telecom-churn-data/telecom_churn_data.csv')

# Display dataset surprise_housing
telecom_churn.head()
# Print the full summary of the dataframe
telecom_churn.info(verbose=True)
# Check descriptive statistics
telecom_churn.describe().T
# Check the shape of data set
telecom_churn.shape
# save original rows count
original_rows = telecom_churn.shape[0]
original_rows
telecom_churn[['total_rech_amt_6', 'total_rech_amt_7']].head()
telecom_churn['AveRechGoodPhase'] = (telecom_churn['total_rech_amt_6'] + telecom_churn['total_rech_amt_7'])//2

telecom_churn[['total_rech_amt_6', 'total_rech_amt_7', 'AveRechGoodPhase']].head()

percentiles = telecom_churn['AveRechGoodPhase'].quantile([0.50,0.70,0.90]).values
print('50 percentile of Good Phase Recharge : ', percentiles[0])
print('70 percentile of Good Phase Recharge : ', percentiles[1])
print('90 percentile of Good Phase Recharge : ', percentiles[2])

seventyPercentile = percentiles[1]

telecom_filtered = telecom_churn[telecom_churn['AveRechGoodPhase'] >= seventyPercentile] 
telecom_filtered.head()
telecom_filtered.shape
listOfColsOf9thMonth = list(telecom_filtered.filter(regex='_9'))
listOfColsOf9thMonth.append('sep_vbc_3g')
listOfColsOf9thMonth
def isChurned(x):
    if ((x.total_ic_mou_9 == 0) and (x.total_og_mou_9 == 0) and (x.vol_2g_mb_9 == 0) and (x.vol_3g_mb_9 == 0)):
        return 1
    else:
        return 0

telecom_filtered['Churn'] = telecom_filtered.apply(isChurned, axis=1)
telecom_filtered['Churn'].value_counts(normalize=True)
telecom_filtered.drop(listOfColsOf9thMonth,axis=1, inplace=True)
telecom_filtered.head()
telecom_filtered.shape
unwanted_col = list(telecom_filtered.filter(regex='date'))
unwanted_col.append('mobile_number')
unwanted_col.append ('circle_id')
unwanted_col
telecom_filtered.drop(unwanted_col,axis=1, inplace=True)
telecom_filtered.head()
telecom_filtered.shape
def checkNullValues(data):
    return round((data.isnull().sum()/len(data.index))*100,2).sort_values(ascending=False)
# Code for column-wise null percentages
null_counts  = checkNullValues(telecom_filtered)
null_counts[null_counts > 0]
rechCols = list(telecom_filtered.filter(regex='rech'))
rechCols
telecom_filtered[rechCols].describe().T
telecom_filtered[rechCols] = telecom_filtered[rechCols].apply(lambda x : x.replace(np.NaN,0))
# Code for column-wise null percentages
null_counts  = checkNullValues(telecom_filtered)
null_counts[null_counts > 0]
fbCols = list(telecom_filtered.filter(regex='fb'))
fbCols
telecom_filtered[fbCols] = telecom_filtered[fbCols].apply(lambda x : x.replace(np.NaN,-1))
telecom_filtered[fbCols].head()
nightpckCols = list(telecom_filtered.filter(regex='night'))
nightpckCols
telecom_filtered[nightpckCols] = telecom_filtered[nightpckCols].apply(lambda x : x.replace(np.NaN,-1))
telecom_filtered[nightpckCols].head()
## convert them to categorical Variables.
telecom_filtered[fbCols] = telecom_filtered[fbCols].astype('category')
telecom_filtered[nightpckCols] = telecom_filtered[nightpckCols].astype('category')
telecom_filtered.info()
# Code for column-wise null percentages
null_counts  = checkNullValues(telecom_filtered)
colsGreterThan60 = list(null_counts[null_counts > 60].index)
colsGreterThan60
telecom_filtered[colsGreterThan60].describe().T
telecom_filtered.drop(colsGreterThan60,axis=1, inplace=True)
telecom_filtered.head()
telecom_filtered.describe().T
# Based on the above statistical summary, we can remove following columns
remove_cols = ['loc_og_t2o_mou', 'std_og_t2o_mou', 'loc_ic_t2o_mou', 'std_og_t2c_mou_6','std_og_t2c_mou_7',
               'std_og_t2c_mou_8','std_ic_t2o_mou_6','std_ic_t2o_mou_7','std_ic_t2o_mou_8']

telecom_filtered.drop(remove_cols,axis=1,inplace=True)
telecom_filtered.head()
# Code for column-wise null percentages
null_counts  = checkNullValues(telecom_filtered)
nullCols = null_counts[null_counts > 0]
telecom_filtered.describe().T
# imputing NaN with median value
telecom_filtered[nullCols.index] = telecom_filtered[nullCols.index].apply(lambda x : x.replace(np.NaN,x.median()))
# Code for column-wise null percentages
null_counts  = checkNullValues(telecom_filtered)
null_counts[null_counts > 0]
print('Retaied rows : ', round((len(telecom_filtered.index)/original_rows)*100,2))
#### On Total columns
totalCols = list(telecom_filtered.filter(regex='total'))
totalCols
#### Create a new variable that highlight the churn phase for total recharge amount
telecom_filtered['total_rech_amt_churnPhs'] =telecom_filtered['total_rech_amt_8']- ((telecom_filtered['total_rech_amt_7'] 
                                                         + telecom_filtered['total_rech_amt_6'])//2)
telecom_filtered['total_rech_amt_churnPhs'].head()
#### Create a new variable that highlight the churn phase for outgoing MOU
telecom_filtered['total_og_mou_churnPhs'] =telecom_filtered['total_og_mou_8']- ((telecom_filtered['total_og_mou_6'] 
                                                         + telecom_filtered['total_og_mou_7'])//2)
telecom_filtered['total_og_mou_churnPhs'].head()
#### On volume columns - Internet usage
volCols = list(telecom_filtered.filter(regex='vol'))
volCols
#### Create a new variable that highlight the churn phase for 3g usage
telecom_filtered['vol_3g_mb_churnPhs'] =telecom_filtered['vol_3g_mb_8']- ((telecom_filtered['vol_3g_mb_6'] 
                                                         + telecom_filtered['vol_3g_mb_7'])//2)
telecom_filtered['vol_3g_mb_churnPhs'].head()
#### Create a new variable that highlight the churn phase for 2g usage
telecom_filtered['vol_2g_mb_churnPhs'] =telecom_filtered['vol_2g_mb_8']- ((telecom_filtered['vol_2g_mb_6'] 
                                                         + telecom_filtered['vol_2g_mb_7'])//2)
telecom_filtered['vol_2g_mb_churnPhs'].head()
#Initial setup for the plots
plt.style.use('ggplot')
plt.rcParams['font.size']=8
plt.rcParams['patch.edgecolor'] = 'k'
#Analyze target column for imbalance
data = [telecom_filtered.Churn.value_counts()[0], telecom_filtered.Churn.value_counts()[1]]

plt.pie(data, explode=(0.1,0), labels=['0','1'],autopct='%1.2f%%',radius=1.2)
plt.title("Churn Ratio")
plt.show()
#Analyze 'rech' columns
analyzeCols = list(telecom_filtered.filter(regex='rech'))
analyzeCols

plt.figure(figsize=(18,32))
for i in enumerate(analyzeCols):
    plt.subplot(9, 3, i[0]+1)
    sns.distplot(telecom_filtered[i[1]])
plt.tight_layout()
plt.show()
def plotBivariate(xdata, ydata):
    plt.figure(figsize=(8,4))
    sns.scatterplot(x=xdata, y=ydata,hue=telecom_filtered.Churn,data=telecom_filtered)
    plt.show()
plotBivariate(telecom_filtered.total_rech_num_6,telecom_filtered.total_rech_num_8)
plotBivariate((telecom_filtered.total_rech_num_7+telecom_filtered.total_rech_num_6)//2,telecom_filtered.total_rech_num_8)
def plotBoxPlot(cols):
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    sns.boxplot(x='Churn',y=cols+"_6",hue='Churn',data=telecom_filtered)
    plt.subplot(1,3,2)
    sns.boxplot(x='Churn',y=cols+"_7",hue='Churn',data=telecom_filtered)
    plt.subplot(1,3,3)
    sns.boxplot(x='Churn',y=cols+"_8",hue='Churn',data=telecom_filtered)
    plt.show()
plotBoxPlot('total_rech_amt')
plotBoxPlot('max_rech_data')
numeric_Cols = list(telecom_filtered.select_dtypes(exclude=['category']).columns)
numeric_Cols.remove('Churn')
numeric_Cols
telecom_filtered[numeric_Cols].describe().T
# Using capping method to handle outliers

for i in numeric_Cols:
    Q1 = telecom_filtered[i].quantile(0.01)
    Q4 = telecom_filtered[i].quantile(0.99)
    telecom_filtered[i][telecom_filtered[i] < Q1] = Q1
    telecom_filtered[i][telecom_filtered[i] > Q4] = Q4
   
plotBoxPlot('total_rech_amt')
plotBoxPlot('total_ic_mou')
plotBoxPlot('arpu')
plotBoxPlot('total_og_mou')
telecom_filtered.shape
# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(telecom_filtered[fbCols], drop_first=True)

# Adding the results to the master dataframe
telecom_filtered = pd.concat([telecom_filtered, dummy1], axis=1)
telecom_filtered.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(telecom_filtered[nightpckCols], drop_first=True)

# Adding the results to the master dataframe
telecom_filtered = pd.concat([telecom_filtered, dummy1], axis=1)
telecom_filtered.head()
## drop the additional columns
telecom_filtered.drop(nightpckCols , inplace=True, axis=1)
telecom_filtered.drop(fbCols, inplace=True , axis=1)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
X = telecom_filtered.drop("Churn", axis = 1)
y = telecom_filtered.Churn
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100, stratify = y)
print(X_train.shape, X_test.shape)
scaler = StandardScaler()

X_train[numeric_Cols] = scaler.fit_transform(X_train[numeric_Cols])

X_train.head()
X_test[numeric_Cols] = scaler.transform(X_test[numeric_Cols])

X_test.head()
pca = PCA(random_state=42)
pca.fit(X_train)
#Looking at the explained variance ratio for each component

print(pd.Series(np.round(pca.explained_variance_ratio_.cumsum(), 4)*100))
# Making a scree plot for the explained variance

var_cumu = np.cumsum(pca.explained_variance_ratio_)

fig = plt.figure(figsize=[12,8])
plt.vlines(x=60, ymax=1, ymin=0, colors="r", linestyles="--")
plt.hlines(y=0.93, xmax=80, xmin=0, colors="g", linestyles="--")
plt.plot(var_cumu)
plt.ylabel("Cumulative variance explained")
plt.show()

pca_final = IncrementalPCA(n_components=60)
X_train_pca = pca_final.fit_transform(X_train)
X_test_pca = pca_final.transform(X_test)
#GridSearchCv for hyperparameter tuning

lr_pca = LogisticRegression(class_weight='balanced')

params = {'penalty':['l1','l2'],
          'C':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,20,30]}

lr_grid_pca = GridSearchCV(estimator=lr_pca,
                           param_grid=params,
                           scoring='recall',
                           cv=4,
                           n_jobs=-1, verbose=1)




# Fit the model

lr_grid_pca.fit(X_train_pca,y_train)

#Best result parameters

lr_grid_pca.best_estimator_
#Best score
lr_grid_pca.best_score_
#Predict with test data
y_pred_pca = lr_grid_pca.predict(X_test_pca)
# create onfusion matrix
cm = confusion_matrix(y_test, y_pred_pca)
print(cm)

accuracy = round(accuracy_score(y_test, y_pred_pca),2)
print(classification_report(y_test, y_pred_pca))
column_names = ['Model', 'Accuracy', 'Sensitivity','Specificity','Precision']
pca_results = pd.DataFrame(columns = column_names)

def addResults(index,model,cm,accuracy):
    sensi = round((cm[1,1]/float(cm[1,1] + cm[1,0])),2)
    speci = round((cm[0,0]/float(cm[0,0] + cm[0,1])),2)
    preci = round((cm[1,1]/float(cm[1,1] + cm[0,1])),2)
    pca_results.loc[index] =[model,accuracy,sensi,speci,preci]
    return pca_results
    
addResults(0,'LogisticRegression',cm,accuracy)
#GridSearchCv for hyperparameter tuning

#scale_pos_weight = total_negative_examples (class 0) / total_positive_examples (class 1)

xg_pca = XGBClassifier(learning_rate=0.1,objective='binary:logistic',scale_pos_weight=9,seed=27)

params = {
    'max_depth': [2,3,5,10,20],
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],
    'n_estimators': [10, 25, 50, 100],
    'subsample': [0.3, 0.6, 0.9]
} 

xg_grid_pca = GridSearchCV(estimator = xg_pca, 
                        param_grid = params,
                        scoring='recall',
                        n_jobs=-1,
                        cv=3,
                        verbose=1)
# Fit the model

xg_grid_pca.fit(X_train_pca,y_train)

#best result parameters

xg_grid_pca.best_estimator_
#Best score
xg_grid_pca.best_score_
#Predict with test data
y_pred_pca = xg_grid_pca.predict(X_test_pca)
# create confusion matrix
cm = confusion_matrix(y_test, y_pred_pca)
print(cm)

accuracy = round(accuracy_score(y_test, y_pred_pca),2)
print(classification_report(y_test, y_pred_pca))
addResults(1,'XGBClassifier',cm,accuracy)
#GridSearchCv for hyperparameter tuning

rf_pca = RandomForestClassifier(class_weight="balanced",random_state=42)

params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10, 25, 50, 100]
}

rf_grid_pca = GridSearchCV(estimator = rf_pca, 
                        param_grid = params,
                        scoring='recall',
                        n_jobs=-1,
                        cv=3,
                        verbose=1)
# Fit the model

rf_grid_pca.fit(X_train_pca,y_train)

#best result parameters

rf_grid_pca.best_estimator_
#Best score
rf_grid_pca.best_score_
#Predict with test data
y_pred_pca = rf_grid_pca.predict(X_test_pca)
# create confusion matrix
cm = confusion_matrix(y_test, y_pred_pca)
print(cm)

accuracy = round(accuracy_score(y_test, y_pred_pca),2)
print(classification_report(y_test, y_pred_pca))
addResults(2,'RandomForestClassifier',cm,accuracy)
#GridSearchCv for hyperparameter tuning

rf = RandomForestClassifier(class_weight="balanced",random_state=42)

params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10, 25, 50, 100]
}

rf_grid = GridSearchCV(estimator = rf, 
                        param_grid = params,
                        scoring='recall',
                        n_jobs=-1,
                        cv=3,
                        verbose=1)
# Fit the model

rf_grid.fit(X_train,y_train)

#best result parameters

rf_best = rf_grid.best_estimator_
rf_best
#Best score
rf_grid.best_score_
#Predict with test data
y_pred = rf_grid.predict(X_test)
# create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = round(accuracy_score(y_test, y_pred),2)
print(classification_report(y_test, y_pred))
addResults(3,'RandomForestClassifierWithoutPCA',cm,accuracy)
imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": rf_best.feature_importances_
}).sort_values(by="Imp", ascending=False)
imp_df['Varname'].head(10)
import statsmodels.api as sm

#### Feature Selection Using RFE

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight='balanced')

from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)

X_train.columns[rfe.support_]
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
rfe_Cols = X_train.columns[rfe.support_]
rfe_Cols
logm1 = sm.GLM(y_train,(sm.add_constant(X_train[rfe_Cols])), family = sm.families.Binomial())
m1 = logm1.fit()
m1.summary()
##### Checking VIFs

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate the VIFs for the new model

def calculateVIF(df):
    vif = pd.DataFrame()
    X = df
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif

calculateVIF(X_train[rfe_Cols])
#Drop 'std_og_mou_8' as VIF is very high
rfe_Cols = rfe_Cols.drop('std_og_mou_8',1)
logm2 = sm.GLM(y_train,(sm.add_constant(X_train[rfe_Cols])), family = sm.families.Binomial())
m2 = logm2.fit()
m2.summary()
##### Checking VIFs

calculateVIF(X_train[rfe_Cols])
#Drop 'loc_og_mou_8' as VIF is very high
rfe_Cols = rfe_Cols.drop('loc_og_mou_8',1)
logm3 = sm.GLM(y_train,(sm.add_constant(X_train[rfe_Cols])), family = sm.families.Binomial())
m3 = logm3.fit()
m3.summary()
##### Checking VIFs

calculateVIF(X_train[rfe_Cols])
#Drop 'count_rech_2g_8' as VIF is very high
rfe_Cols = rfe_Cols.drop('count_rech_2g_8',1)
logm4 = sm.GLM(y_train,(sm.add_constant(X_train[rfe_Cols])), family = sm.families.Binomial())
m4 = logm4.fit()
m4.summary()
#Drop 'sachet_2g_8' as p-Value is very high
rfe_Cols = rfe_Cols.drop('sachet_2g_8',1)
logm5 = sm.GLM(y_train,(sm.add_constant(X_train[rfe_Cols])), family = sm.families.Binomial())
m5 = logm5.fit()
m5.summary()
##### Checking VIFs

calculateVIF(X_train[rfe_Cols])
#Drop 'total_og_mou_8' as VIF is very high
rfe_Cols = rfe_Cols.drop('total_og_mou_8',1)
logm6 = sm.GLM(y_train,(sm.add_constant(X_train[rfe_Cols])), family = sm.families.Binomial())
m6 = logm6.fit()
m6.summary()
##### Checking VIFs

calculateVIF(X_train[rfe_Cols])
#Drop 'offnet_mou_8' as VIF is very high
rfe_Cols = rfe_Cols.drop('offnet_mou_8',1)
logm7 = sm.GLM(y_train,(sm.add_constant(X_train[rfe_Cols])), family = sm.families.Binomial())
m7 = logm7.fit()
m7.summary()
##### Checking VIFs

calculateVIF(X_train[rfe_Cols])
#Drop 'std_og_t2t_mou_7' as VIF is very high
rfe_Cols = rfe_Cols.drop('std_og_t2t_mou_7',1)
logm8 = sm.GLM(y_train,(sm.add_constant(X_train[rfe_Cols])), family = sm.families.Binomial())
m8 = logm8.fit()
m8.summary()
#Drop 'onnet_mou_7' as p-Value is very high
rfe_Cols = rfe_Cols.drop('onnet_mou_7',1)
logm9 = sm.GLM(y_train,(sm.add_constant(X_train[rfe_Cols])), family = sm.families.Binomial())
m9 = logm9.fit()
m9.summary()
##### Checking VIFs

calculateVIF(X_train[rfe_Cols])
y_train_pred = m9.predict(sm.add_constant(X_train[rfe_Cols]))
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Churn':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final.head()
##### Creating new column 'predicted' with 1 if Converted_Prob	> 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()
from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
print(confusion)
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))
## Metrics beyond simply accuracy
def calculateMatrix (confusion):
    TP = confusion[1,1] # true positive 
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] # false positives
    FN = confusion[1,0] # false negatives

 

    # Let's see the sensitivity of our logistic regression model
    sentitivity = round((TP / float(TP+FN))*100,2)
    print('Sentitivity : ', sentitivity)

 

    # Let us calculate specificity
    specificity = round((TN / float(TN+FP))*100,2)
    print('Specificity : ', specificity)

 

    # Calculate false postive rate - predicting churn when customer does not have churned
    falsePositive = round((FP/ float(TN+FP))*100,2)
    print('False Positive : ', falsePositive)

 

    # positive predictive value 
    positivePredictiveValue = round((TP / float(TP+FP))*100,2)
    print('Positive Predictive Value : ', positivePredictiveValue)
    
    # Negative predictive value
    negativePredictiveValue = round((TN / float(TN+ FN))*100,2)
    print('Negative Predictive Value : ', negativePredictiveValue)

 

    #  Precision 
    precision = round((TP / float(TP + FP))*100,2)
    print ('Precision  : ', precision)
calculateMatrix(confusion)
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Churn, y_train_pred_final.Converted_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Churn, y_train_pred_final.Converted_Prob)
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_Prob.map( lambda x: 1 if x > 0.1 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.final_predicted )
confusion2
calculateMatrix (confusion2)
X_test = X_test[rfe_Cols]
X_test.head()
X_test_sm = sm.add_constant(X_test)
#### Making predictions on the test set

y_test_pred = m9.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head
y_pred_1.head()
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
y_test_df.head()
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})
y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.10 else 0)
y_pred_final.head()
# AUC score
auc_score = round(metrics.roc_auc_score( y_pred_final.Churn, y_pred_final.final_predicted ),2)
auc_score
# Let's check the overall accuracy.
accuracy2 = round(metrics.accuracy_score(y_pred_final.Churn, y_pred_final.final_predicted),2)
accuracy2
confusion2 = metrics.confusion_matrix(y_pred_final.Churn, y_pred_final.final_predicted )
confusion2
calculateMatrix (confusion2)
addResults(4,'LogisticRegressionWithoutPCA',confusion2,accuracy2)
# Top Features
topFeatures = pd.DataFrame(m9.params)
topFeatures = topFeatures.iloc[1:,:]
topFeatures = topFeatures.reset_index()
topFeatures.columns=['Top Features','Coeffient']
topFeatures
pca_results
topFeatures