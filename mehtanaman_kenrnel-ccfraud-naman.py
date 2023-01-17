import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

from scipy.stats import skew,norm
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler
from collections import Counter

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
# Result data frame
df_result = pd.DataFrame(columns=['Model','Train/Test','Accuracy', 'Sensitivity/Recall', 
                                  'F1 Score','Roc Auc Score'])

df_cost = pd.DataFrame(columns=['Model','Total Cost'])

def drawHistPlotForColumns(df, attribute):
    '''
    * drawHistPlotForColumns: drawHistPlotForColumns to  plot histogram  based on the parameters.
    * df    : data frame
    * attributes   :attributes list
    '''
    plt.figure(figsize=(20,10))
    if len(attribute)>0:
        ax = plt.subplot(2,3,1)
        sns.distplot(df[attribute[0]][df.Class == 1], bins=50)
        sns.distplot(df[attribute[0]][df.Class == 0], bins=50)
        ax.set_xlabel('')
        ax.set_title('histogram of feature: ' + str(attribute[0]))
    if len(attribute)>1:
        ax = plt.subplot(2,3,2)
        sns.distplot(df[attribute[1]][df.Class == 1], bins=50)
        sns.distplot(df[attribute[1]][df.Class == 0], bins=50)
        ax.set_xlabel('')
        ax.set_title('histogram of feature: ' + str(attribute[1]))
    if len(attribute)>2:
        ax = plt.subplot(2,3,3)
        sns.distplot(df[attribute[2]][df.Class == 1], bins=50)
        sns.distplot(df[attribute[2]][df.Class == 0], bins=50)
        ax.set_xlabel('')
        ax.set_title('histogram of feature: ' + str(attribute[2]))
    plt.show()
    
def getModelCoef(model, X,number=10):
    '''
    * getModelCoef: getModelCoef to  get model coef.  based on the parameters.
    * model    : model instacne
    * X  :column name
    *number : default number or number of coef.
    '''
    for l, r in sorted(zip(X.columns, model.coef_), key=lambda x: abs(x[1]), reverse=True)[:number]:
        print("(%s, %.5f)" % (l, r))
        
def modelEvaluationMetrics(n,k,y_test_pred,y_test):
    '''
    *modelEvaluationMetrics : modelEvaluationMetrics to compute the model metrics based on prameters
    * n : number 
    * k : cutt-off value
    * y_test : test class values
    * y_test_pred : predicted test values
    '''
    resid=np.subtract(y_test_pred,y_test)
    print("Residual sum of squares (RSS)")
    rss=np.sum(np.power(resid,2))
    print("RSS:{}".format(rss))
    print("----------------------------------------------------\n")
    
    aic= n*np.log(rss/n) + 2*k
    print("Akaike information criterion (AIC)")
    print("AIC:{}".format(aic))
    print("-----------------------------------------------------\n")
    
    bic= n*np.log(rss/n) + k*np.log(n)
    print("Bayesian information criterion (BIC)")
    print("BIC:{}".format(bic))
    print("------------------------------------------------------\n")
    
    print("Rˆ2 Score")
    r_square_score=r2_score(y_test,y_test_pred)
    print("Rˆ2 score:{}".format(r_square_score))
    # adjusted r2 using formula adj_r2 = 1 - (1- r2) * (n-1) / (n - k - 1)
    # k = number of predictors = data.shape[1] - 1
    adj_r2 = 1 - (1-r_square_score)*(n- 1) / (n - k - 1)
    print("Adjust Rˆ2  :{}".format(adj_r2))
    print("------------------------------------------------------ \n")
    
    print("MAE, MSE, RMSE")
    print("MAE:{}".format(metrics.mean_absolute_error(y_test, y_test_pred)))
    print("MSE:{}".format(metrics.mean_squared_error(y_test, y_test_pred)))
    print("RMSE:{}".format(np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))
    print("-------------------------------------------------------- \n")
    
def countOutlier(df_in, col_name):
    """
    *countOutlier : countOutlier to repot number of outliers in each column based on parameters.
    *df_in  : data frame
    *col_name : column name
    """
    if df_in[col_name].nunique() > 2:
        orglength = len(df_in[col_name])
        q1 = df_in[col_name].quantile(0.00)
        q3 = df_in[col_name].quantile(0.90)
        iqr = q3-q1 #Interquartile range 
        fence_low  = q1-1.5*iqr 
        fence_high = q3+1.5*iqr 
        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
        newlength = len(df_out[col_name])
        return round(100 - (newlength*100/orglength),2)  
    else:
        return 0
    
def column_univariate(df,col,type,hue =None):
    
    '''
    Credit Univariate function will plot the graphs based on the parameters.
    df      : dataframe name
    col     : Column name
    type : variable type : continuos or categorical
                Continuos(0)   : Distribution, Violin & Boxplot will be plotted.
                Categorical(1) : Countplot will be plotted.
    hue     : It's only applicable for categorical analysis.
    
    '''
    sns.set(style="darkgrid")
    
    if type == 0:
        fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(20,4))
        ax[0].set_title("Distribution Plot")
        sns.distplot(df[col],ax=ax[0])
        ax[1].set_title("Violin Plot")
        if hue is not None: 
            sns.violinplot(data =df, x=col,y=hue,ax=ax[1], hue=hue, inner="quartile")
        else:
            sns.violinplot(data =df, x=col,ax=ax[1],inner="quartile")
        ax[2].set_title("Box Plot")
        if hue is not None: 
            sns.boxplot(data =df, y=col,x=hue, ax=ax[2],hue=hue, orient='v')
        else:
            sns.boxplot(data =df, x=col,ax=ax[2],hue=hue, orient='v')
    
    elif  type == 1:
        total_len = len(df[col])
        percentage_labels = round((df[col].value_counts()/total_len)*100,4)
    
        temp = pd.Series(data = hue)
        
        fig, ax=plt.subplots(nrows =1,ncols=1,figsize=(6,4))
        ax.set_title("Count Plot")
        width = len(df[col].unique()) + 6 + 4*len(temp.unique())
        sns.countplot(data = df, x= col,
                           order=df[col].value_counts().index,hue = hue)
          
         
        
        if len(temp.unique()) > 0:
            for p in ax.patches:
                ax.annotate('{:1.1f}%'.format((p.get_height()*100)/float(len(df))),
                            (p.get_x()+0.05, p.get_height()+20))  
        else:
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2.,
                height + 2,'{:.2f}%'.format(100*(height/total_len)),
                        fontsize=14, ha='center', va='bottom')
        del temp
    
    elif type == 2:
        plt.title("Box Plot for "+col)
        plt.boxplot(df[col])
        
    else:
        exit
    
    plt.show()
    
def modelFit(model, X_train, y_train, performCV=True, cv_folds=5):
    '''
    * modelFit: modelFit to fit the model based on the parameters.
    * model    : model instance
    * X_train, y_train   : X/y train dataset
    * performCV      : perform cross-validation or not 
    * cv_folds : default value of cross validation
    '''
    
    #Fit the model on the data
    model.fit(X_train, y_train)
        
    #Predict based on train data:
    dtrain_predictions = model.predict(X_train)
    dtrain_predprob = model.predict_proba(X_train)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='roc_auc')
    
    #Print model metrics report:
    print ("Accuracy : %.4g" % metrics.roc_auc_score(y_train, dtrain_predictions))
    print ("Recall/Sensitivity : %.4g" % metrics.recall_score(y_train, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
    
    if performCV:
        print ("Cross Validation Score : Mean - %.7g" % (np.mean(cv_score)))
        print ("Cross Validation Score : Std - %.7g" % (np.std(cv_score)))
        print ("Cross Validation Score : Min - %.7g" % (np.min(cv_score)))
        print ("Cross Validation Score : Max - %.7g" % (np.max(cv_score)))
        
def modelMetricsSummary(actual_fraud=False,pred_fraud=False,message="Train"):
    '''
    * modelMetricsSummary: report model metrics summary values based on the parameters.
    * actual_fraud   : actual chrun value
    * pred_fraud      : predicted chrun value
    * message         : Train/Test 
    '''
    confusion = metrics.confusion_matrix(actual_fraud, pred_fraud)

    TP = confusion[1,1] # true positive 
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] # false positives
    FN = confusion[1,0] # false negatives

    print(message + " Score")
    accuracy = metrics.accuracy_score(actual_fraud,pred_fraud)
    sensitivity = TP / float(TP+FN)
    specificity = TN / float(TN+FP)
    precision = metrics.precision_score(actual_fraud, pred_fraud)
    f1score = metrics.f1_score(actual_fraud, pred_fraud)
    rocauc = metrics.roc_auc_score(actual_fraud,pred_fraud)
    print("------------------------------------------------------------------- ")
    print("Accuracy (All Correct Predictions) : {0:.4f}".format(accuracy))
    print("Roc Auc Score (Area under the ROC Curve) : {0:.4f}".format(rocauc))
    print('Sensitivity/Recall (True Positive Rate): {0:.4f}'.format(sensitivity))
    print('Specificity (Correct Negative Predictions): {0:.4f}'.format(specificity))
    print('Precision/Positive predictive value (Correct Positive Predictions): {0:.4f}'.format(precision))
    print('F1 Score: {0:.4f}'.format(f1score))
    print('False Positive Rate (Incorrect Positive Predictions): {0:.4f}'.format(FP/ float(TN+FP)))
    print('Negative Predictive value (Correct Negative Predictions): {0:.4f}'.format(TN / float(TN+ FN)))
    print("---------------------------------------------------------------------- ")
    return accuracy,sensitivity,specificity,precision,f1score,rocauc

def predictFraudWithProb(model,X,y,prob):
    '''
    * predictFraudWithProb: predictFraudWithProb to predict the fraud based on the parameters.
    * model   : model instance
    * pred_fraud      : predicted fraud value
    * X,y          : To predict using model and cut-off probability predict
    '''
    pred_probs = model.predict_proba(X)[:,1]
    y_df= pd.DataFrame({'Fraud':y, 'Fraud_Prob':pred_probs})
    
    # Creating new column 'predicted' with 1 if fraud_Prob>0.5 else 0
    y_df['final_predicted'] = y_df.Fraud_Prob.map( lambda x: 1 if x > prob else 0)
    return y_df

def drawRoc(actual, probs ):
    '''
    * drawRoc: draw ROC curve based on the parameters.
    * actual    : actual value
    * probs     : probablity value
    '''
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
    plt.title('Receiver operating characteristic(ROC)')
    plt.legend(loc="lower right")
    plt.show()
    
def findOptimalCutoff(df):
    
    '''
    * findOptimalCutoff: To find the optimal cutoff for classifing as fraud/non-fraud based on the parameters.
    * df   : input dataframe instance
    * pred_fraud      : predicted fraud value
    * X,y          : To predict using model and cut-off probability predict
    '''
    
    # Create columns with different probability cutoffs 
    numbers = [float(x)/10 for x in range(10)]
    for i in numbers:
        df[i] = df.Fraud_Prob.map( lambda x: 1 if x > i else 0)
    
    # Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
    cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
    from sklearn.metrics import confusion_matrix
    
    # TP = confusion[1,1] # true positive 
    # TN = confusion[0,0] # true negatives
    # FP = confusion[0,1] # false positives
    # FN = confusion[1,0] # false negatives
    
    num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for i in num:
        cm1 = metrics.confusion_matrix(df.Fraud, df[i] )
        total1=sum(sum(cm1))
        accuracy = (cm1[0,0]+cm1[1,1])/total1
        
        speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
        cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
    print(cutoff_df)
    # Plot accuracy sensitivity and specificity for various probabilities.
    cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
    plt.show()
    
def storeResult(df_result,result,model,scorefor):
    '''
    * storeResult: storeResult to hold and return the model result based on the parameters.
    * df_result    : input data
    * result : model results
    * model  : model instance
    '''
    df_result = df_result.append({'Model': model, 'Train/Test': scorefor, 'Accuracy': round(result[0]*100,2), 
                                  'Sensitivity/Recall': round(result[1]*100,2), 'F1 Score': round(result[4]*100,2),
                                  'Roc Auc Score': round(result[5]*100,2)}, ignore_index=True)
    return df_result

def plotTrainTestAccuracy(score,param):
    '''
    * plotTrainTestAccuracy: plotTrainTestAccuracy to plot train/test accuracy based on the parameters.
    * score    : score
    * param : parameter list
    '''
        
    scores = score
    plt.figure()
    plt.plot(scores["param_"+param], 
    scores["mean_train_score"], 
    label="training accuracy")
    plt.plot(scores["param_"+param], 
    scores["mean_test_score"], 
    label="test accuracy")
    plt.xlabel(param)
    plt.ylabel("recall")
    plt.legend()
    plt.show()
    
def create_cost_df(amount, y, y_pred):
    ''' 
    * create_cost_df: create_cost_df to compute the cost per transction
    * amount : amount value
    * y      : actual y class label 
    * y_pred : predicted y class label
    '''
    result_i = pd.DataFrame(
        {
            'Amt': amount.tolist(),
            't': y.tolist(),
            'p': y_pred.tolist()
        }
    )

    result_i["Ca"] = 10
    result_i["Cost"] = (result_i.t*(result_i.p*result_i.Ca + (1 - result_i.p)*result_i.Amt)) + ((1 - result_i.t)*(result_i.p*result_i.Ca))
    return result_i
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
df.isnull().sum().all()
df.info()
df.describe()
#list numerical vriables/features
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    print(col, end=', ')
print("\n\n")   
print("Number of numeric Columns:",len(numerical_cols))
#list categorical vriables/features
categorical_cols = df.select_dtypes(include=[np.object]).columns.tolist()
for col in categorical_cols:
    print(col, end=', ')
print("\n\n")   
print("Number of categorical Columns:",len(categorical_cols))
duplicateRowsDF = df[df.duplicated()]
duplicateRowsDF.shape
df = df.drop_duplicates()
df.shape
plt.bar(['Non-Fraud','Fraud'], df['Class'].value_counts(), color=['b','r'])
plt.xlabel('Class')
plt.ylabel('Number of transactions')
plt.annotate('{}\n({:.4}%)'.format(df['Class'].value_counts()[0], 
                                         df['Class'].value_counts()[0]/df['Class'].count()*100),
             (0.20, 0.45), xycoords='axes fraction')
plt.annotate('{}\n({:.4}%)'.format(df['Class'].value_counts()[1], 
                                         df['Class'].value_counts()[1]/df['Class'].count()*100),
             (0.70, 0.45), xycoords='axes fraction')
plt.tight_layout()
plt.show()
# Time vs Class

f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(12,4))
ax1.hist(df.Time[df.Class == 1], bins = 50, color = 'red')
ax1.set_title('Fraud')
 
ax2.hist(df.Time[df.Class == 0], bins = 50, color = 'green')
ax2.set_title('Non-Fraud')

ax1.set_xlabel('Time (in Seconds)')
ax1.set_ylabel('Number of Fraud Transactions')
ax2.set_xlabel('Time (in Seconds)')
ax2.set_ylabel('Number of Non-Fraud Transactions')

plt.show()
plt.scatter(df['Time']/(60*60), df['Class'])
plt.xlabel('Time of transaction (in hours)')
plt.ylabel('Class')

plt.tight_layout()
plt.show()
#Time vs Amount vs Class
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(12,4))

ax1.scatter(df.Time[df.Class == 1], df.Amount[df.Class == 1], color = 'red')
ax1.set_title('Fraud')

ax2.scatter(df.Time[df.Class == 0], df.Amount[df.Class == 0],  color = 'green')
ax2.set_title('Non-Fraud')

ax1.set_xlabel('Time (in Seconds)')
ax1.set_ylabel('Amount')
ax2.set_xlabel('Time (in Seconds)')
ax2.set_ylabel('Amount')
plt.show()
plt.boxplot(df['Amount'], labels = ['Boxplot'])
plt.ylabel('Transaction amount')
plt.plot()

amount = df[['Amount']].sort_values(by='Amount')
q1, q3 = np.percentile(amount,[25,75])
iqr = q3 - q1
lower_bound = q1 -(1.5 * iqr) 
upper_bound = q3 +(1.5 * iqr)

print('Number of outliers below the lower bound (25th Percentile): ', amount[amount['Amount'] < lower_bound].count()[0],
     ' ({:.4}%)'.format(amount[amount['Amount'] < lower_bound].count()[0] / amount['Amount'].count() * 100))
print('Number of outliers above the upper bound (75th Percentile): ', amount[amount['Amount'] > upper_bound].count()[0],
      ' ({:.4}%)'.format(amount[amount['Amount'] > upper_bound].count()[0] / amount['Amount'].count() * 100))
df[df['Class']==1].where(df['Amount']>upper_bound).count()['Amount']
target_0 = df.loc[df['Class'] == 0]
target_1 = df.loc[df['Class'] == 1]
ax1=sns.distplot(target_0[['Amount']], hist=False, color='b', label='Non-fraud')
ax2=sns.distplot(target_1[['Amount']], hist=False, color='r', label='Fraud')
ax1.set_xlim(0, max(df[df['Class']==1]['Amount']))
ax2.set_xlim(0, max(df[df['Class']==1]['Amount']))
plt.legend()
plt.xlabel('Amount')
plt.ylabel('Density of probability')
plt.show()
numerical = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical:
    print(col,"=",countOutlier(df,col))
column_univariate(df=df[df.Class==0],col='Amount',type=0)
column_univariate(df=df[df.Class==1],col='Amount',type=0)
amount_outlier = df[df.Amount>5000]
amount_outlier.shape
amount_zero = df[df.Amount==0]
amount_zero.shape
amount_zero_nonfraud = amount_zero[amount_zero.Class==0]
amount_zero_nonfraud.shape
amount_zero_fraud = amount_zero[amount_zero.Class==1]
amount_zero_fraud.shape
i = 1
while i < 9:
    drawHistPlotForColumns(df,['V'+str(i),'V'+str(i+1),'V'+str(i+2)])
    i = i+3
i = 10
while i < 19:
    drawHistPlotForColumns(df,['V'+str(i),'V'+str(i+1),'V'+str(i+2)])
    i = i+3
i = 19
while i < 29:
    if i==28:
        drawHistPlotForColumns(df,['V'+str(i),'Amount'])
    else:
        drawHistPlotForColumns(df,['V'+str(i),'V'+str(i+1),'V'+str(i+2)])
    i = i+3
print("Average 'Amount' in case of Fraud: " + str(np.mean(df[df['Class'] == 1]['Amount'])))
print("The max 'Amount' in case of Fraud: " + str(max(df[df['Class'] == 1]['Amount'])))
print(' ')
print("Average 'Amount' in case of Non-Fraud: " + str(np.mean(df[df['Class']== 0]['Amount'])))
print("The max 'Amount' in case of Non-Fraud: " + str(max(df[df['Class']== 0]['Amount'])))
newdf=df.loc[:, df.columns != 'Class']
fig,ax = plt.subplots(figsize=(20,20))
sns.heatmap(newdf.corr(),ax=ax,annot= True,linewidth= 0.02,linecolor='black',fmt='.2f',cmap = 'Blues_r')
plt.show()
fig, (ax1, ax2) = plt.subplots(1,2,figsize =( 15, 6))
sns.heatmap(df.query('Class==1').drop(['Class','Time'],1).corr(), vmax = .8, square=True, ax = ax1, cmap = 'YlGnBu' );
ax1.set_title('Fraud')

sns.heatmap(df.query('Class==0').drop(['Class','Time'],1).corr(), vmax = .8, square=True, ax = ax2, cmap = 'YlGnBu' ); 
ax2.set_title('Non Fraud')
plt.show()
#feature vs correlation with class
df.corrwith(df.Class, method='spearman').plot.bar(
        figsize = (18, 5), title = "Correlation with class", fontsize = 15,
        rot = 45, grid = True, color=['grey'])
plt.show()
df.drop(['Time'],axis=1, inplace=True)
# Splitting the data into train and test
X = df.loc[:, df.columns != 'Class']
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, stratify=y, random_state=100)
X_test_copy = X_test.copy()
print("Counts of label '1': {}".format(sum(y_train==1)))
print("Counts of label '0': {}".format(sum(y_train==0)))
print("Event rate : {}% \n".format(round(sum(y_train==1)/len(y_train)*100,2)))
scaler = StandardScaler()
X_train['Amount'] = scaler.fit_transform(X_train['Amount'].values.reshape(-1, 1))
X_test['Amount'] = scaler.transform(X_test['Amount'].values.reshape(-1, 1))
X_train.head()
X_test.head()
cols= X.columns
# Make the graph 20 inches by 40 inches
plt.figure(figsize=(20,40), facecolor='white')

# plot numbering starts at 1, not 0
plot_number = 1
for variable in cols:
    # Inside of an image that's a 15x13 grid, put this
    # graph in the in the plot_number slot.
    ax = plt.subplot(20, 5, plot_number)
    plt.hist(X_train[variable], bins=50)
    ax.set_title(variable)
    # 
    ax.text(x=0.97, y=0.91, transform=ax.transAxes, s="Skewness: %f" % X_train[variable].skew(),\
        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\
        backgroundcolor='white', color='xkcd:poo brown')
    ax.text(x=0.97, y=0.81, transform=ax.transAxes, s="Kurtosis: %f" % X_train[variable].kurt(),\
        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\
        backgroundcolor='white', color='xkcd:dried blood')
    # Go to the next plot for the next loop
    plot_number = plot_number + 1
    plt.tight_layout()
#Train data transformation

from sklearn.preprocessing import PowerTransformer
features=X_train[cols]

pt = PowerTransformer(method='yeo-johnson', standardize=True,) 

#Fit the data to the powertransformer
skl_yeojohnson = pt.fit(features )

#Lets get the Lambdas that were found
print (skl_yeojohnson.lambdas_)

calc_lambdas = skl_yeojohnson.lambdas_

#Transform the data 
skl_yeojohnson = pt.transform(features)

#Pass the transformed data into a new dataframe 
X_train_xt = pd.DataFrame(data=skl_yeojohnson, columns=[cols])

X_train_xt.head() 
fig=plt.figure(figsize=(25,18))
ax=fig.gca()
X_train.hist(ax=ax,bins=50)
plt.show()
#Test Data Transoformation
features=X_test[cols]

#Fit the data to the powertransformer
skl_yeojohnson = pt.fit(features )

#Lets get the Lambdas that were found
print (skl_yeojohnson.lambdas_)

calc_lambdas = skl_yeojohnson.lambdas_

#Transform the data 
skl_yeojohnson = pt.transform(features)

#Pass the transformed data into a new dataframe 
X_test_xt = pd.DataFrame(data=skl_yeojohnson, columns=[cols])

X_test_xt.head() 
fig=plt.figure(figsize=(25,18))
ax=fig.gca()
X_test.hist(ax=ax,bins=50)
plt.show()
random_under_sample = RandomUnderSampler(sampling_strategy='auto', random_state=42, replacement=False)
X_random_under_sample, y_random_under_sample = random_under_sample.fit_resample(X_train_xt, y_train)
plt.bar(['Non-Fraud','Fraud'], [Counter(y_random_under_sample)[0], Counter(y_random_under_sample)[1]],
        color=['green','red'])
plt.xlabel('Class')
plt.ylabel('Number of transactions')
plt.annotate('{}'.format(Counter(y_random_under_sample)[0]), (0.25, 0.45), xycoords='axes fraction')
plt.annotate('{}'.format(Counter(y_random_under_sample)[1]), (0.75, 0.45), xycoords='axes fraction')

plt.tight_layout()
plt.show()
#logistic regression instance.
lr = LogisticRegression(class_weight='balanced')
#fit train data.
modelFit(lr, X_random_under_sample, y_random_under_sample)
# predictions on Test data
pred_test = lr.predict(X_test_xt)
result = modelMetricsSummary(y_test,pred_test,"Test")
X_random_under_sample
#cutoff selection
cut_off_prob=0.5
y_train_df = predictFraudWithProb(lr,X_random_under_sample,y_random_under_sample,cut_off_prob)
y_train_df.head()
# check the Model Summary
result = modelMetricsSummary(y_train_df.Fraud,y_train_df.final_predicted,"Train")
# Plot ROC curve
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding cut-off with the right balance of the metrices
# sensitivity vs specificity trade-off vs accuracy
findOptimalCutoff(y_train_df)
cut_off_prob = 0.30
res_df = predictFraudWithProb(lr,X_random_under_sample,y_random_under_sample,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Logistic-Random Under Sampling","Train")
# predicting with the choosen cut-off on test
res_df = predictFraudWithProb(lr,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Logistic-Random Under Sampling","Test")
df_result
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Logistic-Random Under Sampling", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
#Decision Tree
dt = DecisionTreeClassifier(class_weight='balanced',
                             max_features='auto',
                             min_samples_split=100,
                             min_samples_leaf=100,
                             max_depth=6,
                             random_state=10)
# Fit the train data
modelFit(dt, X_random_under_sample, y_random_under_sample)
# make predictions
pred_test = dt.predict(X_test_xt)
#Let's check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': range(10,25,3),
    'min_samples_leaf': range(300, 500, 50),
    'min_samples_split': range(300, 500, 100),
    'max_features': [15,18,21,24]
}
# Create a based model
dt = DecisionTreeClassifier(class_weight='balanced',random_state=10)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = dt, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1,scoring="f1_weighted",refit=False)
grid_search.fit(X_random_under_sample, y_random_under_sample)
print('Optimal hyperparameters for Model ',grid_search.best_score_,'using',grid_search.best_params_)
# model with the best hyperparameters
dt_final = DecisionTreeClassifier(class_weight='balanced',
                             max_depth=10,
                             min_samples_leaf=300, 
                             min_samples_split=300,
                             max_features=15,
                             random_state=10)
modelFit(dt_final,X_random_under_sample,y_random_under_sample)
# make predictions
pred_test = dt_final.predict(X_test_xt)
#Let's check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
print(classification_report(y_test,pred_test))
# predicting churn with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictFraudWithProb(dt_final,X_random_under_sample,y_random_under_sample,cut_off_prob)
y_train_df.head()
#Plot ROC Curve
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)
# predicting fraud/non-fraud class with optimal cut-off 0.19
cut_off_prob=0.19
res_df = predictFraudWithProb(dt_final,X_random_under_sample,y_random_under_sample,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Decision Tree-Random Under Sampling","Train")
#Check the test data performance
res_df= predictFraudWithProb(dt_final,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Decision Tree-Random Under Sampling","Test")
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Decision Tree-Random Under Sampling", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
parameters = {'max_depth': range(10, 30, 5)}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
#fit the train data
rfgs.fit(X_random_under_sample,y_random_under_sample)
scores = rfgs.cv_results_
# plotting accuracies with max_depth
plotTrainTestAccuracy(scores,'max_depth')
parameters = {'max_features': [8, 12, 16, 20, 24, 28]}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
rfgs.fit(X_random_under_sample,y_random_under_sample)
scores = rfgs.cv_results_
# plotting accuracies with max_features
plotTrainTestAccuracy(scores,'max_features')
parameters = {'min_samples_leaf': range(2, 200, 25)}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
rfgs.fit(X_random_under_sample,y_random_under_sample)
scores = rfgs.cv_results_
# plotting accuracies with min_samples_leaf
plotTrainTestAccuracy(scores,'min_samples_leaf')
parameters = {'min_samples_split': range(2, 200, 25)}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
rfgs.fit(X_random_under_sample,y_random_under_sample)
scores = rfgs.cv_results_
# plotting accuracies with min_samples_split
plotTrainTestAccuracy(scores,'min_samples_split')
#Final model post hyper parameters optimization
rf_final = RandomForestClassifier(max_depth=20,
                                  max_features=12,
                                  min_samples_leaf=50,
                                  min_samples_split=100,
                                  random_state=10,n_jobs = -1)
#fit the train data
modelFit(rf_final,X_random_under_sample,y_random_under_sample)
# make predictions
pred_test = rf_final.predict(X_test_xt)
#Let's check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
# predicting churn with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictFraudWithProb(rf_final,X_random_under_sample,y_random_under_sample,cut_off_prob)
y_train_df.head()
#Plot ROC curve
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)
# predicting Fraud/Non-Fraud with optimal cut-off 0.30
cut_off_prob=0.30
res_df = predictFraudWithProb(rf_final,X_random_under_sample,y_random_under_sample,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Random Forest-Random Under Sampling","Train")
#Chekc the test data performance
res_df= predictFraudWithProb(rf_final,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Random Forest-Random Under Sampling","Test")
df_result.loc[df_result.Model=="Random Forest-Random Under Sampling"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Random Forest-Random Under Sampling", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
y_random_over_sample = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_random_over_sample, y_random_over_sample = y_random_over_sample.fit_resample(X_train_xt, y_train)
plt.bar(['Non-Fraud','Fraud'], [Counter(y_random_over_sample)[0], Counter(y_random_over_sample)[1]], color=['green','red'])
plt.xlabel('Class')
plt.ylabel('Number of transactions')
plt.annotate('{}'.format(Counter(y_random_over_sample)[0]), (0.20, 0.45), xycoords='axes fraction')
plt.annotate('{}'.format(Counter(y_random_over_sample)[1]), (0.70, 0.45), xycoords='axes fraction')

plt.tight_layout()
plt.show()
# logistic regression
lr = LogisticRegression(class_weight='balanced')
#Fit the train data
modelFit(lr, X_random_over_sample, y_random_over_sample)
# predictions on Test data
pred_test = lr.predict(X_test_xt)
result = modelMetricsSummary(y_test,pred_test,"Test")
#choose the cutoff probablity.
cut_off_prob=0.5
y_train_df = predictFraudWithProb(lr,X_random_over_sample,y_random_over_sample,cut_off_prob)
y_train_df.head()
# Check the Model Summary
result = modelMetricsSummary(y_train_df.Fraud,y_train_df.final_predicted,"Train")
#Draw ROC curve.
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding cut-off with the right balance of the metrices
# sensitivity vs specificity trade-off vs accuracy
findOptimalCutoff(y_train_df)
#Optimal cutoff value.
cut_off_prob = 0.31
res_df = predictFraudWithProb(lr,X_random_over_sample,y_random_over_sample,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Logistic-Random Over Sampling","Train")
# predicting with the choosen cut-off on test
res_df = predictFraudWithProb(lr,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Logistic-Random Over Sampling","Test")
df_result.loc[df_result.Model=="Logistic-Random Over Sampling"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Logistic-Random Over Sampling", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
#Decision Tree
dt = DecisionTreeClassifier(class_weight='balanced',
                             max_features='auto',
                             min_samples_split=100,
                             min_samples_leaf=100,
                             max_depth=6,
                             random_state=10)
# Fit the train data
modelFit(dt, X_random_over_sample, y_random_over_sample)
# make predictions
pred_test = dt.predict(X_test_xt)
#Let's check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': range(10,25,3),
    'min_samples_leaf': range(300, 500, 50),
    'min_samples_split': range(300, 500, 100),
    'max_features': [15,18,21,24]
}
# Create a based model
dt = DecisionTreeClassifier(class_weight='balanced',random_state=10)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = dt, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1,scoring="f1_weighted",refit=False)
grid_search.fit(X_random_over_sample, y_random_over_sample)
print('Optimal hyperparameters for Model ',grid_search.best_score_,'using',grid_search.best_params_)
# model with the best hyperparameters
dt_final = DecisionTreeClassifier(class_weight='balanced',
                             max_depth=16,
                             min_samples_leaf=300, 
                             min_samples_split=300,
                             max_features=21,
                             random_state=10)
modelFit(dt_final,X_random_over_sample,y_random_over_sample)
# make predictions
pred_test = dt_final.predict(X_test_xt)
#Let's check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
print(classification_report(y_test,pred_test))
# predicting churn with default cut-off 0.5
cut_off_prob = 0.5
y_train_df = predictFraudWithProb(dt_final,X_random_over_sample,y_random_over_sample,cut_off_prob)
y_train_df.head()
#Plot ROC Curve
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)
# predicting fraud/non-fraud class with optimal cut-off 0.70
cut_off_prob=0.70
res_df = predictFraudWithProb(dt_final,X_random_over_sample,y_random_over_sample,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Decision Tree-Random Over Sampling","Train")
#Check the test data performance
res_df= predictFraudWithProb(dt_final,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Decision Tree-Random Over Sampling","Test")
df_result.loc[df_result.Model=="Decision Tree-Random Over Sampling"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Decision Tree-Random Over Sampling", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
#Random forest
parameters = {'max_depth': range(10, 30, 5)}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
#fit the train data
rfgs.fit(X_random_over_sample,y_random_over_sample)
scores = rfgs.cv_results_
# plotting accuracies with max_depth
plotTrainTestAccuracy(scores,'max_depth')
parameters = {'max_features': [8, 12, 16, 20, 24, 28]}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
rfgs.fit(X_random_over_sample,y_random_over_sample)
scores = rfgs.cv_results_
# plotting accuracies with max_features
plotTrainTestAccuracy(scores,'max_features')
parameters = {'min_samples_leaf': range(20, 100, 25)}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
rfgs.fit(X_random_over_sample,y_random_over_sample)
scores = rfgs.cv_results_
# plotting accuracies with min_samples_leaf
plotTrainTestAccuracy(scores,'min_samples_leaf')
parameters = {'min_samples_split': range(20, 300, 25)}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
rfgs.fit(X_random_over_sample,y_random_over_sample)
scores = rfgs.cv_results_
# plotting accuracies with min_samples_split
plotTrainTestAccuracy(scores,'min_samples_split')
#Final model post hyper parameters optimization
rf_final = RandomForestClassifier(max_depth=15,
                                  max_features=20,
                                  min_samples_leaf=120,
                                  min_samples_split=120,
                                  random_state=10,n_jobs = -1)
#fit the train data
modelFit(rf_final,X_random_over_sample,y_random_over_sample)
# make predictions
pred_test = rf_final.predict(X_test_xt)
#Let's check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
# predicting churn with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictFraudWithProb(rf_final,X_random_over_sample,y_random_over_sample,cut_off_prob)
y_train_df.head()
#Plot ROC curve
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)
# predicting Fraud/Non-Fraud with optimal cut-off 0.37
cut_off_prob=0.37
res_df = predictFraudWithProb(rf_final,X_random_over_sample,y_random_over_sample,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Random Forest-Random Over Sampling","Train")
#Chekc the test data performance
res_df= predictFraudWithProb(rf_final,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Random Forest-Random Over Sampling","Test")
df_result.loc[df_result.Model=="Random Forest-Random Over Sampling"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Random Forest-Random Over Sampling", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
asasyn = ADASYN(random_state=42)
X_train_an, y_train_an = asasyn.fit_resample(X_train_xt, y_train)
plt.bar(['Non-Fraud','Fraud'], [Counter(y_train_an)[0], Counter(y_train_an)[1]], color=['green','red'])
plt.xlabel('Class')
plt.ylabel('Number of transactions')
plt.annotate('{}'.format(Counter(y_random_over_sample)[0]), (0.20, 0.45), xycoords='axes fraction')
plt.annotate('{}'.format(Counter(y_random_over_sample)[1]), (0.70, 0.45), xycoords='axes fraction')

plt.tight_layout()
plt.show()
lr = LogisticRegression(class_weight='balanced')
modelFit(lr, X_train_an, y_train_an)
# predictions on Test data
pred_test = lr.predict(X_test_xt)
result = modelMetricsSummary(y_test,pred_test,"Test")
cut_off_prob=0.5
y_train_df = predictFraudWithProb(lr,X_train_an,y_train_an,cut_off_prob)
y_train_df.head()
# Let's see the Model Summary
result = modelMetricsSummary(y_train_df.Fraud,y_train_df.final_predicted,"Train")
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding cut-off with the right balance of the metrices
# sensitivity vs specificity trade-off vs accuracy
findOptimalCutoff(y_train_df)
cut_off_prob = 0.45
res_df = predictFraudWithProb(lr,X_train_an,y_train_an,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Logistic-ADASYN","Train")
# predicting with the choosen cut-off on test
res_df = predictFraudWithProb(lr,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Logistic-ADASYN","Test")
df_result.loc[df_result.Model=="Logistic-ADASYN"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Logistic-ADASYN", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
#Decision Tree
dt = DecisionTreeClassifier(class_weight='balanced',
                             max_features='auto',
                             min_samples_split=100,
                             min_samples_leaf=100,
                             max_depth=6,
                             random_state=10)
# Fit the train data
modelFit(dt, X_train_an, y_train_an)
# make predictions
pred_test = dt.predict(X_test_xt)
#Let's check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': range(10,25,3),
    'min_samples_leaf': range(300, 500, 50),
    'min_samples_split': range(300, 500, 100),
    'max_features': [15,18,21,24]
}
# Create a based model
dt = DecisionTreeClassifier(class_weight='balanced',random_state=10)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = dt, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1,scoring="f1_weighted",refit=False)
grid_search.fit(X_train_an, y_train_an)
print('Optimal hyperparameters for Model ',grid_search.best_score_,'using',grid_search.best_params_)
# model with the best hyperparameters
dt_final = DecisionTreeClassifier(class_weight='balanced',
                             max_depth=16,
                             min_samples_leaf=400, 
                             min_samples_split=300,
                             max_features=15,
                             random_state=10)
modelFit(dt_final,X_train_an,y_train_an)
# make predictions
pred_test = dt_final.predict(X_test_xt)
#Let's check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
print(classification_report(y_test,pred_test))
# predicting churn with default cut-off 0.5
cut_off_prob = 0.5
y_train_df = predictFraudWithProb(dt_final,X_train_an,y_train_an,cut_off_prob)
y_train_df.head()
#Plot ROC Curve
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)
# predicting fraud/non-fraud class with optimal cut-off 0.45
cut_off_prob=0.45
res_df = predictFraudWithProb(dt_final,X_train_an,y_train_an,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Decision Tree-ADASYN","Train")
#Check the test data performance
res_df= predictFraudWithProb(dt_final,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Decision Tree-ADASYN","Test")
df_result.loc[df_result.Model=="Decision Tree-ADASYN"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Decision Tree-ADASYN", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
#Random forest
parameters = {'max_depth': range(5, 20, 5)}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
#fit the train data
rfgs.fit(X_train_an,y_train_an)
scores = rfgs.cv_results_
# plotting accuracies with max_depth
plotTrainTestAccuracy(scores,'max_depth')
parameters = {'max_features': [8, 12, 16, 20, 24, 28]}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
rfgs.fit(X_train_an,y_train_an)
scores = rfgs.cv_results_
# plotting accuracies with max_features
plotTrainTestAccuracy(scores,'max_features')
parameters = {'min_samples_leaf': range(200, 500, 50)}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
rfgs.fit(X_train_an,y_train_an)
scores = rfgs.cv_results_
# plotting accuracies with min_samples_leaf
plotTrainTestAccuracy(scores,'min_samples_leaf')
parameters = {'min_samples_split': range(100, 400, 50)}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
rfgs.fit(X_train_an,y_train_an)
scores = rfgs.cv_results_
# plotting accuracies with min_samples_split
plotTrainTestAccuracy(scores,'min_samples_split')
#Final model post hyper parameters optimization
rf_final = RandomForestClassifier(max_depth=10,
                                  max_features=16,
                                  min_samples_leaf=350,
                                  min_samples_split=200,
                                  random_state=10,n_jobs = -1)
#fit the train data
modelFit(rf_final,X_train_an,y_train_an)
# make predictions
pred_test = rf_final.predict(X_test_xt)
#Let's check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
# predicting churn with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictFraudWithProb(rf_final,X_train_an,y_train_an,cut_off_prob)
y_train_df.head()
#Plot ROC curve
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)
# predicting Fraud/Non-Fraud with optimal cut-off 0.6
cut_off_prob=0.6
res_df = predictFraudWithProb(rf_final,X_train_an,y_train_an,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Random Forest-ADASYN","Train")
#Chekc the test data performance
res_df= predictFraudWithProb(rf_final,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Random Forest-ADASYN","Test")
df_result.loc[df_result.Model=="Random Forest-ADASYN"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Random Forest-ADASYN", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
# grid
param_grid = {
                "learning_rate": [0.0001, 0.001, 0.01],
                "subsample": [0.05, 0.03, 0.01]
             }
# XGBoost Classifer
xgb = XGBClassifier(max_depth=9, n_estimators=500, n_jobs = -1)
# Hyperparameter tunining with GridSearch,cross-validation =5 folds.
folds = 5
grid_search_GBC = GridSearchCV(xgb, 
                               cv = folds,
                               param_grid=param_grid, 
                               scoring = 'roc_auc', 
                               return_train_score=True,                         
                               verbose = 1,n_jobs = -1 ,refit=False)
#Fit the train data
grid_search_GBC.fit(X_train_an, y_train_an)
cv_results = pd.DataFrame(grid_search_GBC.cv_results_)
cv_results.head()
#Plotting
plt.figure(figsize=(16,6))
for n, subsample in enumerate(param_grid['subsample']):
    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]
    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')

# Fitting the XGBClassifier
xgb = XGBClassifier(learning_rate =0.001,
                    n_estimators=500,
                    max_depth=9,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.01,
                    colsample_bytree=0.8,
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    n_jobs = -1,
                    seed=27)
# Model fit and performance on Train data
modelFit(xgb, X_train_an, y_train_an)
# Hyperparameter tunning for the XGBClassifer
param_test1 = {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.001, n_estimators=500, subsample=0.05, colsample_bytree=0.8,
objective= 'binary:logistic', nthread=-1, scale_pos_weight=1, seed=27,n_jobs = -1), 
param_grid = param_test1, scoring='f1',n_jobs=-1,iid=False, cv=3 ,refit=False)
gsearch1.fit(X_train_an, y_train_an)
gsearch1.best_params_, gsearch1.best_score_
# Some more hyperparameter tunning for the XGBClassifer
param_test2 = param_test3 = {'gamma':[i/10.0 for i in range(0,5)]}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.001, n_estimators=500, subsample=0.05, colsample_bytree=0.8,
objective= 'binary:logistic', nthread=-1, scale_pos_weight=1,seed=27,n_jobs = -1), 
param_grid = param_test2, scoring='f1',n_jobs=-1,iid=False, cv=3 ,refit=False)
gsearch2.fit(X_train_an, y_train_an)
gsearch2.best_params_, gsearch2.best_score_
# Final XGBClassifier 
xgb_final = XGBClassifier(learning_rate =0.001,
                    n_estimators=500,
                    max_depth=9,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.01,
                    colsample_bytree=0.9,
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    n_jobs = -1,      
                    seed=27)
# Fit Train data
modelFit(xgb_final, X_train_an, y_train_an)
# Prediction on Test data
predictions = xgb_final.predict(X_test_xt)
# Check the Test data performance
result = modelMetricsSummary(y_test,predictions,message="Test")
# predicting churn with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictFraudWithProb(xgb_final,X_train_an,y_train_an,cut_off_prob)
y_train_df.head()
reuslt = modelMetricsSummary(y_train_df.Fraud,y_train_df.final_predicted,"Train")
# Finding optimal cut-off probability
findOptimalCutoff(y_train_df)
# Choose optinal cutoff :0.52  
cut_off_prob=0.52
res_df = predictFraudWithProb(xgb_final,X_train_an,y_train_an,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"XGBoost-ADASYN","Train")
#check the test data
res_df= predictFraudWithProb(xgb_final,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"XGBoost-ADASYN","Test")
df_result.loc[df_result.Model=="XGBoost-ADASYN"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "XGBoost-ADASYN", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
# parameter grid
param_grid = {
                "learning_rate": [0.001, 0.01, 0.1],
                "subsample": [0.03, 0.02, 0.01]
             }
#Gradient Boosting Classifier
gbm = GradientBoostingClassifier(max_depth=9, n_estimators=500, random_state=10)
#Hyperparameters runing with cross-validation 
folds = 5
grid_search_GBC = GridSearchCV(gbm, 
                               cv = folds,
                               param_grid=param_grid, 
                               scoring = 'roc_auc', 
                               return_train_score=True, 
                               n_jobs = -1,
                               verbose = 1 ,refit=False)
#Fit the train data
grid_search_GBC.fit(X_train_an, y_train_an)
cv_results = pd.DataFrame(grid_search_GBC.cv_results_)
cv_results.head()
## plotting
plt.figure(figsize=(16,6))
for n, subsample in enumerate(param_grid['subsample']):
    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]
    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')
gbm = GradientBoostingClassifier(learning_rate =0.01,subsample=0.03, random_state=10)
parameters = {'n_estimators':range(300,700,50)}
gsearch = GridSearchCV(gbm, 
                       cv = 3,
                       param_grid=parameters, 
                       scoring = 'roc_auc', 
                       return_train_score=True, 
                       n_jobs = -1,
                       verbose = 1 ,refit=False)
gsearch.fit(X_train_an, y_train_an)
gsearch.best_params_, gsearch.best_score_
parameters = {'max_depth':range(7,21,3)}
gsearch = GridSearchCV(gbm, 
                       cv = 3,
                       param_grid=parameters, 
                       scoring = 'roc_auc', 
                       return_train_score=True,                         
                       verbose = 1 ,refit=False)
gsearch.fit(X_train_an, y_train_an)
gsearch.best_params_, gsearch.best_score_
parameters = {'min_samples_split':range(10,300,50)}
gsearch = GridSearchCV(gbm, 
                       cv = 3,
                       param_grid=parameters, 
                       scoring = 'roc_auc', 
                       return_train_score=True, 
                       n_jobs = -1,
                       verbose = 1 ,refit=False)
gsearch.fit(X_train_an, y_train_an)
gsearch.best_params_, gsearch.best_score_
parameters = {'min_samples_leaf':range(10,50,10)}
gsearch = GridSearchCV(gbm, 
                       cv = 3,
                       param_grid=parameters, 
                       scoring = 'roc_auc', 
                       return_train_score=True,                         
                       verbose = 1 ,refit=False)
gsearch.fit(X_train_an, y_train_an)
gsearch.best_params_, gsearch.best_score_
parameters = {'max_features':range(7,20,2)}
gsearch = GridSearchCV(gbm, 
                       cv = 3,
                       param_grid=parameters, 
                       scoring = 'roc_auc', 
                       return_train_score=True, 
                       n_jobs = -1,
                       verbose = 1 ,refit=False)
gsearch.fit(X_train_an, y_train_an)
gsearch.best_params_, gsearch.best_score_
#final Gradiant Boosting classifer
gbm_final = GradientBoostingClassifier(
    learning_rate =0.01,
    subsample=0.03, 
    max_features=19,
    min_samples_leaf=10,
    min_samples_split=260,
    max_depth=19,
    n_estimators=650,
    random_state=10)
#fit the train data
modelFit(gbm_final, X_train_an, y_train_an)
# predict the test data
pred_test = gbm_final.predict(X_test_xt)
#Check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
# predicting churn with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictFraudWithProb(gbm_final,X_train_an,y_train_an,cut_off_prob)
y_train_df.head()
#Plot ROC Curve
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding optimal cut-off  
findOptimalCutoff(y_train_df)
# Optimal cutoff :0.52
cut_off_prob=0.52
res_df = predictFraudWithProb(gbm_final,X_train_an,y_train_an,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Gradient Boosting-ADASYN","Train")
#Check the test data performance
res_df= predictFraudWithProb(gbm_final,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Gradient Boosting-ADASYN","Test")
df_result.loc[df_result.Model=="Gradient Boosting-ADASYN"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Gradient Boosting-ADASYN", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
sm = SMOTE(random_state=12)
X_train_sm, y_train_sm = sm.fit_sample(X_train_xt, y_train)
plt.bar(['Non-Fraud','Fraud'], [Counter(y_train_sm)[0], Counter(y_train_sm)[1]], color=['green','red'])
plt.xlabel('Class')
plt.ylabel('Number of transactions')
plt.annotate('{}'.format(Counter(y_train_sm)[0]), (0.20, 0.45), xycoords='axes fraction')
plt.annotate('{}'.format(Counter(y_train_sm)[1]), (0.70, 0.45), xycoords='axes fraction')

plt.tight_layout()
plt.show()
# Logistic Regression
lr = LogisticRegression(class_weight='balanced')
#Fit the train data
modelFit(lr, X_train_sm, y_train_sm)
# predictions on Test data
pred_test = lr.predict(X_test_xt)
result = modelMetricsSummary(y_test,pred_test,"Test")
cut_off_prob=0.5
y_train_df = predictFraudWithProb(lr,X_train_sm,y_train_sm,cut_off_prob)
y_train_df.head()
# Check the Model Summary
result = modelMetricsSummary(y_train_df.Fraud,y_train_df.final_predicted,"Train")
#Plot ROC curve
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding cut-off with the right balance of the metrices
# sensitivity vs specificity trade-off vs accuracy
findOptimalCutoff(y_train_df)
#Choose optimal proablity cutoff
cut_off_prob = 0.33
res_df = predictFraudWithProb(lr,X_train_sm,y_train_sm,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Logistic-SMOTE","Train")
# predicting with the choosen cut-off on test
res_df = predictFraudWithProb(lr,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Logistic-SMOTE","Test")
df_result.loc[df_result.Model=="Logistic-SMOTE"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Logistic-SMOTE", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
#Decision Tree
dt = DecisionTreeClassifier(class_weight='balanced',
                             max_features='auto',
                             min_samples_split=100,
                             min_samples_leaf=100,
                             max_depth=6,
                             random_state=10)
# Fit the train data
modelFit(dt, X_train_sm, y_train_sm)
# make predictions
pred_test = dt.predict(X_test_xt)
#Let's check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': range(10,25,3),
    'min_samples_leaf': range(300, 500, 50),
    'min_samples_split': range(300, 500, 100),
    'max_features': [15,18,21,24]
}
# Create a based model
dt = DecisionTreeClassifier(class_weight='balanced',random_state=10)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = dt, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1,scoring="f1_weighted",refit=False)
grid_search.fit(X_train_sm, y_train_sm)
print('Optimal hyperparameters for Model ',grid_search.best_score_,'using',grid_search.best_params_)
# model with the best hyperparameters
dt_final = DecisionTreeClassifier(class_weight='balanced',
                             max_depth=22,
                             min_samples_leaf=300, 
                             min_samples_split=300,
                             max_features=21,
                             random_state=10)
modelFit(dt_final,X_train_sm,y_train_sm)
# make predictions
pred_test = dt_final.predict(X_test_xt)
#Let's check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
print(classification_report(y_test,pred_test))
# predicting churn with default cut-off 0.5
cut_off_prob = 0.5
y_train_df = predictFraudWithProb(dt_final,X_train_sm,y_train_sm,cut_off_prob)
y_train_df.head()
# Result Metrics Summary
reuslt = modelMetricsSummary(y_train_df.Fraud,y_train_df.final_predicted,"Train")
#Plot ROC Curve
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)
# predicting fraud/non-fraud class with optimal cut-off 0.55
cut_off_prob=0.55
res_df = predictFraudWithProb(dt_final,X_train_sm,y_train_sm,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Decision Tree-SMOTE","Train")
#Check the test data performance
res_df= predictFraudWithProb(dt_final,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Decision Tree-SMOTE","Test")
df_result.loc[df_result.Model=="Decision Tree-SMOTE"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Decision Tree-SMOTE", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
#Random forest
parameters = {'max_depth': range(10, 30, 5)}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
#fit the train data
rfgs.fit(X_train_sm,y_train_sm)
scores = rfgs.cv_results_
# plotting accuracies with max_depth
plotTrainTestAccuracy(scores,'max_depth')
parameters = {'max_features': [8, 12, 16, 20, 24, 28]}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
rfgs.fit(X_train_sm,y_train_sm)
scores = rfgs.cv_results_

# plotting accuracies with max_features
plotTrainTestAccuracy(scores,'max_features')
parameters = {'min_samples_leaf': range(20, 100, 25)}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
rfgs.fit(X_train_sm,y_train_sm)
scores = rfgs.cv_results_

# plotting accuracies with min_samples_leaf
plotTrainTestAccuracy(scores,'min_samples_leaf')
parameters = {'min_samples_split': range(20, 300, 25)}
rf = RandomForestClassifier(n_jobs = -1)
rfgs = GridSearchCV(rf, parameters, cv=5, scoring="recall", n_jobs = -1, return_train_score = True ,refit=False)
rfgs.fit(X_train_sm,y_train_sm)
scores = rfgs.cv_results_

# plotting accuracies with min_samples_split
plotTrainTestAccuracy(scores,'min_samples_split')
#Final model post hyper parameters optimization
rf_final = RandomForestClassifier(max_depth=14,
                                  max_features=20,
                                  min_samples_leaf=200,
                                  min_samples_split=300,
                                  random_state=10,n_jobs = -1)
#fit the train data
modelFit(rf_final,X_train_sm,y_train_sm)
# make predictions
pred_test = rf_final.predict(X_test_xt)
#Let's check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
# predicting Fraud/Non-Fraud with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictFraudWithProb(rf_final,X_train_sm,y_train_sm,cut_off_prob)
y_train_df.head()
#Check the model metrics summary
reuslt = modelMetricsSummary(y_train_df.Fraud,y_train_df.final_predicted,"Train")
#Plot ROC curve
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)
# predicting Fraud/Non-Fraud with optimal cut-off 0.6
cut_off_prob=0.6
res_df = predictFraudWithProb(rf_final,X_train_sm,y_train_sm,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Random Forest-SMOTE","Train")
#Chekc the test data performance
res_df= predictFraudWithProb(rf_final,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Random Forest-SMOTE","Test")
## taking copy of rf_smote model as it's going to be the selected one
rf_smote = rf_final
df_result.loc[df_result.Model=="Random Forest-SMOTE"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Random Forest-SMOTE", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
# XGBoost Classifer

# grid
param_grid = {
                "learning_rate": [0.0001, 0.001, 0.01],
                "subsample": [0.05, 0.03, 0.01]
             }
# XGBoost Classifer
xgb = XGBClassifier(max_depth=9, n_estimators=500, n_jobs = -1)
# Hyperparameter tunining with GridSearch,cross-validation =5 folds.
folds = 5
grid_search_GBC = GridSearchCV(xgb, 
                               cv = folds,
                               param_grid=param_grid, 
                               scoring = 'roc_auc', 
                               return_train_score=True,                         
                               verbose = 1,n_jobs = -1 ,refit=False)

#Fit the train data
grid_search_GBC.fit(X_train_sm, y_train_sm)
cv_results = pd.DataFrame(grid_search_GBC.cv_results_)
cv_results.head()
#Plotting
plt.figure(figsize=(16,6))


for n, subsample in enumerate(param_grid['subsample']):
    

    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')
# Fitting the XGBClassifier
xgb = XGBClassifier(learning_rate =0.0001,
                    n_estimators=500,
                    max_depth=9,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.01,
                    colsample_bytree=0.8,
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    n_jobs = -1,
                    seed=27)
# Model fit and performance on Train data
modelFit(xgb, X_train_sm, y_train_sm)
# Hyperparameter tunning for the XGBClassifer
param_test1 = {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.001, n_estimators=500, subsample=0.05, colsample_bytree=0.8,
objective= 'binary:logistic', nthread=-1, scale_pos_weight=1, seed=27,n_jobs = -1), 
param_grid = param_test1, scoring='f1',n_jobs=-1,iid=False, cv=3 ,refit=False)
gsearch1.fit(X_train_sm, y_train_sm)
gsearch1.best_params_, gsearch1.best_score_
# Some more hyperparameter tunning for the XGBClassifer
param_test2 = param_test3 = {'gamma':[i/10.0 for i in range(0,5)]}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.001, n_estimators=500, subsample=0.05, colsample_bytree=0.8,
objective= 'binary:logistic', nthread=-1, scale_pos_weight=1,seed=27,n_jobs = -1), 
param_grid = param_test2, scoring='f1',n_jobs=-1,iid=False, cv=3 ,refit=False)
gsearch2.fit(X_train_sm, y_train_sm)
gsearch2.best_params_, gsearch2.best_score_
# Final XGBClassifier
 
xgb_final = XGBClassifier(learning_rate =0.0001,
                    n_estimators=500,
                    max_depth=9,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.01,
                    colsample_bytree=0.9,
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    n_jobs = -1,      
                    seed=27)
# Fit Train data
modelFit(xgb_final, X_train_sm, y_train_sm)
# Prediction on Test data
predictions = xgb_final.predict(X_test_xt)
# Check the Test data performance
result = modelMetricsSummary(y_test,predictions,message="Test")
# predicting Fraud/Non-Fraud with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictFraudWithProb(xgb_final,X_train_sm,y_train_sm,cut_off_prob)
y_train_df.head()
#Check the model metric summary
reuslt = modelMetricsSummary(y_train_df.Fraud,y_train_df.final_predicted,"Train")
# Finding optimal cut-off probability
findOptimalCutoff(y_train_df)
# Choose optinal cutoff :0.5
cut_off_prob=0.5
res_df = predictFraudWithProb(xgb_final,X_train_sm,y_train_sm,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"XGBoost-SMOTE","Train")
#check the test data
res_df= predictFraudWithProb(xgb_final,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"XGBoost-SMOTE","Test")
df_result.loc[df_result.Model=="XGBoost-SMOTE"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "XGBoost-SMOTE", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_cost
# parameter grid
param_grid = {
                "learning_rate": [0.001, 0.01, 0.1],
                "subsample": [0.03, 0.02, 0.01]
             }
#Gradient Boosting Classifier
gbm = GradientBoostingClassifier(max_depth=9, n_estimators=500, random_state=10)
#Hyperparameters runing with cross-validation 
folds = 5
grid_search_GBC = GridSearchCV(gbm, 
                               cv = folds,
                               param_grid=param_grid, 
                               scoring = 'roc_auc', 
                               return_train_score=True, 
                               n_jobs = -1,
                               verbose = 1 ,refit=False)
#Fit the train data
grid_search_GBC.fit(X_train_sm, y_train_sm)
cv_results = pd.DataFrame(grid_search_GBC.cv_results_)
cv_results.head()
# # plotting
plt.figure(figsize=(16,6))


for n, subsample in enumerate(param_grid['subsample']):
    

    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')
gbm = GradientBoostingClassifier(learning_rate =0.1,subsample=0.03, random_state=10)
parameters = {'n_estimators':range(300,700,50)}
gsearch = GridSearchCV(gbm, 
                       cv = 3,
                       param_grid=parameters, 
                       scoring = 'roc_auc', 
                       return_train_score=True, 
                       n_jobs = -1,
                       verbose = 1 ,refit=False)
gsearch.fit(X_train_sm, y_train_sm)
gsearch.best_params_, gsearch.best_score_
parameters = {'max_depth':range(7,21,3)}
gsearch = GridSearchCV(gbm, 
                       cv = 3,
                       param_grid=parameters, 
                       scoring = 'roc_auc', 
                       return_train_score=True,                         
                       verbose = 1 ,refit=False)
gsearch.fit(X_train_sm, y_train_sm)
gsearch.best_params_, gsearch.best_score_
parameters = {'min_samples_split':range(100,300,50)}
gsearch = GridSearchCV(gbm, 
                       cv = 3,
                       param_grid=parameters, 
                       scoring = 'roc_auc', 
                       return_train_score=True, 
                       n_jobs = -1,
                       verbose = 1 ,refit=False)
gsearch.fit(X_train_sm, y_train_sm)
gsearch.best_params_, gsearch.best_score_
parameters = {'min_samples_leaf':range(10,50,10)}
gsearch = GridSearchCV(gbm, 
                       cv = 3,
                       param_grid=parameters, 
                       scoring = 'roc_auc', 
                       return_train_score=True,                         
                       verbose = 1 ,refit=False)
gsearch.fit(X_train_sm, y_train_sm)
gsearch.best_params_, gsearch.best_score_
parameters = {'max_features':range(7,20,2)}
gsearch = GridSearchCV(gbm, 
                       cv = 3,
                       param_grid=parameters, 
                       scoring = 'roc_auc', 
                       return_train_score=True, 
                       n_jobs = -1,
                       verbose = 1 ,refit=False)
gsearch.fit(X_train_sm, y_train_sm)
gsearch.best_params_, gsearch.best_score_
#final Gradiant Boosting classifer
gbm_final = GradientBoostingClassifier(
    learning_rate =0.1,
    subsample=0.03, 
    max_features=13,
    min_samples_leaf=10,
    min_samples_split=100,
    max_depth=19,
    n_estimators=650,
    random_state=10)
#fit the train data
modelFit(gbm_final, X_train_sm, y_train_sm)
# predict the test data
pred_test = gbm_final.predict(X_test_xt)
#Check the model metrices.
result = modelMetricsSummary(actual_fraud=y_test,pred_fraud=pred_test,message="Test")
# predicting Fraud/Non-Fraud with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictFraudWithProb(gbm_final,X_train_sm,y_train_sm,cut_off_prob)
y_train_df.head()
#Check the metrics summary
reuslt = modelMetricsSummary(y_train_df.Fraud,y_train_df.final_predicted,"Train")
#Plot ROC Curve
drawRoc(y_train_df.Fraud, y_train_df.final_predicted)
# finding optimal cut-off  
findOptimalCutoff(y_train_df)
# Optimal cutoff :0.45
cut_off_prob=0.45
res_df = predictFraudWithProb(gbm_final,X_train_sm,y_train_sm,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Train")
df_result = storeResult(df_result,result,"Gradient Boosting-SMOTE","Train")
#Check the test data performance
res_df= predictFraudWithProb(gbm_final,X_test_xt,y_test,cut_off_prob)
result = modelMetricsSummary(res_df.Fraud,res_df.final_predicted,"Test")
df_result = storeResult(df_result,result,"Gradient Boosting-SMOTE","Test")
df_result.loc[df_result.Model=="Gradient Boosting-SMOTE"]
result = create_cost_df(X_test_copy.Amount, res_df.Fraud, res_df.final_predicted)
df_cost = df_cost.append({'Model': "Gradient Boosting-SMOTE", 'Total Cost': round(result.Cost.sum(),2)}, ignore_index=True)
df_result
df_plot = df_result.loc[df_result['Train/Test']=="Test"]
df_plot.sort_values('Sensitivity/Recall',inplace=True)
plt.figure(figsize=(8,8))
plt.title("Sensitivity/Recall for Different Models on Test Data")
plt.bar(df_plot.Model,df_plot['Sensitivity/Recall'])
plt.xlabel('Models')
plt.ylabel('Sensitivity/Recall For Models')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
df_plot = df_result.loc[df_result['Train/Test']=="Test"]
df_plot.sort_values('Roc Auc Score',inplace=True)
plt.figure(figsize=(8,8))
plt.title("Roc Auc Score for Different Models on Test Data")
plt.bar(df_plot.Model,df_plot['Roc Auc Score'])
plt.xlabel('Models')
plt.ylabel('Roc Auc Score For Models')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
df_cost
df_cost.sort_values('Total Cost',inplace=True)
plt.figure(figsize=(8,8))
plt.title("Distribution of Cost for Different Models")
plt.bar(df_cost.Model,df_cost['Total Cost'])
plt.xlabel('Models')
plt.ylabel('Total Cost For Models')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,8))
plt.title("Important Features")
plt.xlabel('Weightage of Features')
plt.ylabel('Features')
imp_features = pd.Series(rf_smote.feature_importances_, index=X_train.columns)
imp_features.nlargest(10).sort_values().plot(kind='barh', align='center')