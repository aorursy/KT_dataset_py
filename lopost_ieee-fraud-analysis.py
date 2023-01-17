import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

from statsmodels.distributions.empirical_distribution import ECDF



# time logic:

import time

import datetime

from datetime import timedelta

from dateutil.relativedelta import relativedelta



#sklearn

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import average_precision_score

from sklearn.metrics import auc

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from sklearn.metrics import precision_recall_curve



from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing



from xgboost import XGBClassifier

import xgboost as xgb

import pickle



# Input data files are available in the "../input/" directory.



import os

print(os.listdir("../input"))



# Standard plotly imports

import plotly.express as px
train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')



#merge the two separate CSV files:

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print(train.shape)

print(test.shape)

train.head(5)
#check how many transaction there are for fraud vs non-fraud

fraud_non_fraud_df = train.groupby(by='isFraud').TransactionDT.count()

fraud_non_fraud_df = pd.DataFrame(fraud_non_fraud_df)

num_transactions = fraud_non_fraud_df.sum()

fraud_non_fraud_df['perc'] = fraud_non_fraud_df/num_transactions*100



print(fraud_non_fraud_df)
plt.figure(figsize=(16,6))

plt.subplot(121)

plt.pie(fraud_non_fraud_df.perc, colors=['g','r'],autopct='%.2f')

plt.title('Number of fraud/ non-fraud transactions')

plt.xlabel('Fraud/ non-fraud transaction')

plt.ylabel('Number of transactions')
device_type_df = train.groupby(by='DeviceType').TransactionDT.count()

device_type_df.plot(kind='pie',colors=['r','g'],autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Number of transactions by device type')

print(device_type_df)
df_device_type_pivot = train.pivot_table(columns='DeviceType',index='isFraud',values='TransactionDT',aggfunc='count',margins=True)

pd.options.display.float_format = '{:.2f}'.format

df_device_type_pivot
#as percent of total number of transactions:

df_device_type_pivot_percent = df_device_type_pivot / 140810*100



cm_green = sns.light_palette("green", as_cmap=True)

df_device_type_pivot_percent.style.background_gradient(cmap=cm_green, subset=['desktop','mobile'])
perc_desktop_fraud = round(df_device_type_pivot['desktop'].loc[1] / df_device_type_pivot['desktop'].loc[0],5)

perc_mobile_fraud  = round(df_device_type_pivot['mobile'].loc[1] / df_device_type_pivot['mobile'].loc[0],5)



print('% of fraud for desktop: ', perc_desktop_fraud)

print('% of fraud for mobile: ', perc_mobile_fraud)
df_card_type = train[['isFraud','card4']].reset_index()

df_card_type.head(5)
df_card_type_pivot = pd.pivot_table(data=df_card_type,index='card4', columns='isFraud',aggfunc='count')

df_card_type_pivot['perc'] = df_card_type_pivot.TransactionID[1]/df_card_type_pivot.TransactionID[0]*100

df_card_type_pivot.sort_values(by='perc',inplace=True)

df_card_type_pivot
plt.figure()

df_card_type_pivot.perc.plot(kind='bar',title='Percentage of fraudulent transactions by card type')

plt.xticks(rotation=40)

plt.xlabel('Card type')
df_card_type_amount = train[['isFraud','card4','TransactionAmt']].reset_index(drop=True)

print(df_card_type_amount.head(5))

df_card_type_amount_pivot = pd.pivot_table(data=df_card_type_amount,index='card4', columns='isFraud',aggfunc='mean')

df_card_type_amount_pivot
df_card_type_amount_pivot.plot(kind='bar',color=['g','r'],title='Avg.transaction amount for fraud/non-fraud transactions')

plt.ylabel('Avg transaction amount')

plt.xticks(rotation=40)
transaction_amts = train['TransactionAmt']

transaction_amts.describe()
print(transaction_amts.quantile(q=0.90))

print(transaction_amts.quantile(q=0.95))
transaction_amts_short = transaction_amts[transaction_amts< 275]

transaction_amts_short.hist(bins=50)

plt.title('Histogram for transaction amount')
fig = plt.figure()

plt.hist(transaction_amts_short, normed=True, cumulative=True, label='CDF',

         histtype='bar', alpha=0.8, color='g',bins=50)

plt.title('ECDF for transaction amount')

plt.xlabel('Transaction amount')
#use plotly to make fancy graphs:

transaction_amts_short_df = pd.DataFrame(transaction_amts_short).reset_index(drop=True)

print(transaction_amts_short_df.head(5))

fig = px.histogram(transaction_amts_short_df,x='TransactionAmt',nbins=50)

fig.update_layout(

    autosize=False,

    width=500,

    height=500,

    paper_bgcolor="LightSteelBlue",

)

fig.show()

print('Done')
import plotly.graph_objects as go

print(transaction_amts_short_df.head(10))

fig = go.Figure(data=[go.Histogram(x=transaction_amts_short_df.TransactionAmt, 

                                   cumulative_enabled=True,

                                   histnorm='percent',

                                  nbinsx=200)])

fig.update_layout(

    autosize=False,

    width=500,

    height=500,

    title_text='ECDF for transaction amounts under 275 USD', # title of plot

    xaxis_title_text='Transaction amount', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2, # gap between bars of adjacent location coordinates

    bargroupgap=0.1 # gap between bars 

)



fig.show()
print(train.head(5))

train.describe()
data_raw = train.drop(columns=['isFraud']).reset_index(drop=True)

target_raw = train[['isFraud']].reset_index(drop=True)
#list all columns to get an idea what we could work with

all_cols = data_raw.columns.values

print(all_cols)
data_raw.dtypes
#check the column 

data_raw[[ 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226',

       'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234',

       'V235', 'V236', 'V237', 'V238', 'V239', 'V240', 'V241', 'V242',

       'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250',

       'V251', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258',

       'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266',

       'V267', 'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274',

       'V275', 'V276', 'V277', 'V278', 'V279', 'V280', 'V281', 'V282',

       'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289', 'V290',

       'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298',

       'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306',

       'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314',

       'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'V322',

       'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330',

       'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338',

       'V339', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06',

       'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13',

       'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20',

       'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27',

       'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34',]].describe().transpose()
data_v1 = data_raw

data_v1.shape
from sklearn.preprocessing import LabelEncoder



cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',

            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',

            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',

            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']

for col in cat_cols:

    if col in data_v1.columns:

        le = LabelEncoder()

        le.fit(list(data_v1[col].astype(str).values) + list(test[col].astype(str).values))

        data_v1[col] = le.transform(list(data_v1[col].astype(str).values))

        test[col] = le.transform(list(test[col].astype(str).values)) 
data_v1.drop(columns=['TransactionDT'],axis=1, inplace=True)
data_v1_desc = data_v1.describe()

data_v1_desc
data_v1_coeffVars = data_v1_desc.loc['std'] / data_v1_desc.loc['mean']

data_v1_coeffVars = data_v1_coeffVars.sort_values(ascending=False)

print(data_v1_coeffVars.head(5))



plt.figure(figsize=(40,15))

data_v1_coeffVars.plot(kind='barh')

plt.axvline(x=0.1, color='k', linestyle='--',label='Potential cutoff variance threshold')

plt.legend()

plt.title
coeff_var_threshold = 0.5

data_v1_coeffVars_more5 = data_v1_coeffVars[data_v1_coeffVars>coeff_var_threshold]

len(data_v1_coeffVars_more5)
data_v1a = data_v1[data_v1_coeffVars_more5.index]

data_v1a.shape
#check the variance of data for each column:

data_v1_vars = data_v1.describe().loc['std'].sort_values(ascending=False)

plt.figure(figsize=(20,15))

data_v1_vars.plot(kind='barh')

plt.axvline(x=0.1, color='k', linestyle='--',label='Potential cutoff variance threshold')

print(data_v1_vars.head(20))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_v1a, target_raw, test_size=0.15, random_state=42)

print('X_train shape: ', X_train.shape)

print('X_test shape: ', X_test.shape)
#new model:

#import sys

print(datetime.datetime.now())

xgboost_model_v1a = XGBClassifier()

xgboost_model_v1a.fit(X_test, y_test)

print(datetime.datetime.now())
#set up functions to plot confusion matrix and precision recall curve

def print_confusion_matrix(

    confusion_matrix, class_names, figsize=(3, 2.5), fontsize=14

):

    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    """

    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)

    fig = plt.figure(figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(

        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize

    )

    heatmap.xaxis.set_ticklabels(

        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize

    )

    plt.ylabel("True label")

    plt.xlabel("Predicted label")

    plt.title("Confusion matrix for classifier:")

    # fig.colorbar(shrink=0.8)

    #return fig

    

def plot_precision_recall_curve(

    y_test, y_pred_proba_df, title="Precision/Recall Curve"

):

    """

    feed two arrays and output precision recall Curve

    """

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_df)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

    #step_kwargs = ({"step": "post"} if "step" in signature(plt.fill_between).parameters else {})

    plt.step(recall, precision, color="b", alpha=0.5, where="post")

    #plt.fill_between(recall, precision, alpha=0.2, color="b", **step_kwargs)



    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title(title)

    plt.show()

    return plt
#test the xgboost model:

y_pred_xgboost_v1 = pd.DataFrame(xgboost_model_v1a.predict_proba(X_test))

y_pred_xgboost_v1 = y_pred_xgboost_v1[1]

plot_precision_recall_curve(y_test=y_test,y_pred_proba_df=y_pred_xgboost_v1)
from sklearn.feature_selection import VarianceThreshold

#Remove all the features with low variance:

def variance_threshold_selector(data, threshold=0.01):

    """

    applies a variance threshold to a dataframe and returns the dataframe instead of array

    """

    selector = VarianceThreshold(threshold)

    selector.fit(data)

    return data[data.columns[selector.get_support(indices=True)]]



def apply_std_threshold(data,std_threshold = 0.1):

    '''removes all the columns with standard deviation under specific value'''

    data_dropped_low_variance = variance_threshold_selector(data,threshold=std_threshold)

    print(data_dropped_low_variance.shape)

    #print(type(data_dropped_low_variance))

    return data_dropped_low_variance





def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
auc = roc_auc_score(y_test, y_pred_xgboost_v1)

print('AUC: %.5f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_xgboost_v1)

plot_roc_curve(fpr, tpr)
cols_to_include = ['TransactionAmt', 'ProductCD', 'card1', 'card2',

       'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',

       'dist2', 'P_emaildomain', 'R_emaildomain','DeviceType']
data_filtered_1 = data_raw[cols_to_include]

data_filtered_1.head(5)
#define which columns are to be treated as numberic and which ones as categorical

list_categorical_cols = ['ProductCD','card4','card6','P_emaildomain','R_emaildomain','DeviceType']

list_numerical_cols   = [column for column in cols_to_include if column not in list_categorical_cols]

print('Categorical cols: ', list_categorical_cols)

print('Numberic cols: ', list_numerical_cols)
# TODO get this working:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit_transform(data_filtered_1[list_numerical_cols])

data_filtered_1.head(5)
print(data_filtered_1[list_numerical_cols].head(5))

scaled_data = scaler.fit_transform(data_filtered_1[list_numerical_cols])

scaled_data_df = pd.DataFrame(scaled_data)

print(scaled_data_df.head(5))
data_filtered_2 = data_filtered_1.merge(scaled_data_df,left_on=data_filtered_1.index, right_on = scaled_data_df.index)

data_filtered_2.drop(columns=list_numerical_cols,inplace=True)

data_filtered_2.drop(columns='key_0',inplace=True)

data_filtered_2.head(5)
#make dummy columns with categorical cols:

data_filtered_catCols = pd.get_dummies(data_filtered_2, columns = list_categorical_cols)

data_filtered_catCols.head(5)
#get rid of the NaN in the data:

data_filtered_catCols_woNa = data_filtered_catCols.fillna(data_filtered_catCols.mean())

data_filtered_catCols_woNa.head(5)
#check if any value in the dataframe is NaN:

if (data_filtered_catCols_woNa.isnull().values.any()) == False:

    print('No NaN values found in dataframe, all good.')

else:

    print('You have NaN values in the dataframe that need to be filled first.')
#check the variance of data for each column:

data_vars = data_filtered_catCols_woNa.describe().loc['std'].sort_values(ascending=False)

plt.figure(figsize=(20,15))

data_vars.plot(kind='barh')

plt.axvline(x=0.1, color='k', linestyle='--',label='Potential cutoff variance threshold')

print(data_vars.head(20))
actual_var_threshold = 0.02

data_low_var = apply_std_threshold(data_filtered_catCols_woNa,std_threshold = actual_var_threshold)

print(data_low_var.columns.values)
#check the data to be used in training:

data_filtered_catCols_woNa.head(5)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_filtered_catCols_woNa, target_raw, test_size=0.15, random_state=42)

print('X_train shape: ', X_train.shape)

print('X_test shape: ', X_test.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification
print('Number of datapoints: ', data_filtered_catCols_woNa.shape[0])
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

clf.fit(data_filtered_catCols_woNa, target_raw)  

#print(clf.feature_importances_)
#test the model performance:

y_pred = clf.predict_proba(X_test)

print(y_pred[0:5])

print(y_pred.min())

print(y_pred.max())
#make a dataframe out of the results and select the first column

y_pred_df = pd.DataFrame(y_pred)

y_pred_series = y_pred_df[1]

print(y_pred_series[0:10])

print(y_pred_series.min())

print(y_pred_series.max())
def round_up_or_down(x, threshold=0.08):

    if x > threshold:

        x = 1

    else:

        x =0

    return x

y_pred_custom = y_pred_series.apply(round_up_or_down)

y_pred_custom[0:10]
from sklearn.metrics import confusion_matrix

conf_matrix_1 = confusion_matrix(y_test, y_pred_custom)

print_confusion_matrix(conf_matrix_1,class_names = ['Non-Fraud','Fraud'])




auc = roc_auc_score(y_test, y_pred_series)

print('AUC: %.5f' % auc)

plot_precision_recall_curve(y_test=y_test,y_pred_proba_df=y_pred_series)
param_grid_test = { 

    'n_estimators': [5],

    'max_features': ['auto'],

    'max_depth' : [2],

    'criterion' :['gini']

}

param_grid_1 = { 

    'n_estimators': [100,200],

    'max_features': ['auto'],

    'max_depth' : [4,5,6],

    'criterion' :['gini']

}



param_grid_2 = { 

    'n_estimators': [10,20,50,100,200,300,400,500],

    'max_features': ['auto'],

    'max_depth' : np.arange(2,20,2),

    'criterion' :['gini']

}




def run_model_RandomForest(data, target, param_grid):

    """set the structure of the model and transform the target column"""

    rfc = RandomForestClassifier(random_state=42)

    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    CV_rfc.fit(data, target.values.ravel())

    return CV_rfc



print(datetime.datetime.now())

CV_rfc = run_model_RandomForest(data_filtered_catCols_woNa, target_raw,param_grid_test)

print(datetime.datetime.now())
#test the model:

y_pred_randForest = pd.DataFrame(CV_rfc.predict_proba(X_test))

y_pred_randForest = y_pred_randForest[1]
plot_precision_recall_curve(y_test=y_test,y_pred_proba_df=y_pred_randForest)
#check if pickled model is available

os.listdir()

#load the model again:

try:

    model_filename = 'xgboost_v1.joblib.dat'

    loaded_model = pickle.load(open(model_filename, "rb"))

    print('Loaded model...')

except Exception as e:

    print(e)
#new model:

#import sys

print(datetime.datetime.now())

xgboost_model = XGBClassifier()

xgboost_model.fit(data_filtered_catCols_woNa, target_raw)

print(datetime.datetime.now())
#test the xgboost model:

y_pred_xgboost = pd.DataFrame(xgboost_model.predict_proba(X_test))

y_pred_xgboost = y_pred_xgboost[1]

plot_precision_recall_curve(y_test=y_test,y_pred_proba_df=y_pred_xgboost)
#save the model:



model_filename = 'xgboost_v1.joblib.dat'

pickle.dump(xgboost_model, open(model_filename, "wb"))

os.listdir()
auc = roc_auc_score(y_test, y_pred_xgboost)

print('AUC: %.5f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_xgboost)

plot_roc_curve(fpr, tpr)