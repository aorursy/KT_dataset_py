# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score,recall_score
pd.options.display.float_format = '{:,.2f}'.format
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def calculate_statistics_and_make_a_plot(input_data):
    '''this function calculates descriptive statistics of a numeric variable and makes a plot'''
    print(input_data.describe())                                                   #prints descriptive statistiscs
    draw_graph(input_data)                                                         #draw a graph
    
    
def draw_graph(dt):
    ''' this function makes a histogram plot'''
    row_size=22                                                                    #setting row size
    col_size=6                                                                     # setting column size
    fig=plt.figure(figsize=(row_size,col_size))                                    #creating a figure
    ax1=fig.add_subplot(1,2,1)                                                     #adding a subplot
    ax1=plt.hist(dt)                                                               #creating a histogram
    plt.title('{} Distribution'.format(dt.name))                                   #setting a title 
    plt.xlabel(dt.name)                                                            # setting x-axis label
    plt.ylabel('Count')                                                            # setting y-axis label
    
def calculate_statistics_and_make_a_plot_without_extremes(input_data,lower_threshold=0.01,upper_threshold=0.99):
    '''this function calculates descriptive statistics of a numeric variable and makes a plot without top and bottom %1 data'''
    mask=np.logical_and(input_data>input_data.quantile(lower_threshold),           #creating a list for spliting extreme values from the rest
                        input_data<input_data.quantile(upper_threshold))
    calculate_statistics_and_make_a_plot(input_data.loc[mask])                     #calculating descriptive statistics and making plot
            
def mark_extreme_values(input_data,col_list,lower_threshold=0.01,upper_threshold=0.99):
    '''this function marks extreme values.'''
    for col in col_list:                                                           #creating a for loop runs over the col list
        new_col_name=col+'_Is_Extreme'                                             #creating new name for new feature
        input_data[new_col_name]=np.logical_or(input_data[col]<input_data[col].quantile(lower_threshold),  #marking extreme values on a new column
                                                input_data[col]>input_data[col].quantile(upper_threshold))
        input_data[new_col_name]=input_data[new_col_name]*1                        #converting boolean values to numeric
        
def replace_extremes(input_data,col_list,lower_threshold=0.01,upper_threshold=0.99):
    ''' this function replaces extreme values'''
    for col in col_list:                                                           #creating a for loop runs over the col list
        new_value=input_data[col].median()                                         #calculating the median value
        is_extreme=np.logical_or(input_data[col]<input_data[col].quantile(lower_threshold),            #creating a list for spliting extreme values from the rest
                             input_data[col]>input_data[col].quantile(upper_threshold))
        input_data[col][is_extreme]=new_value                                      #replacing extreme values with the median value
    return input_data
        
def remove_extremes(input_data,col_list,lower_threshold=0.01,upper_threshold=0.99):
    '''this function removes extreme values'''
    for col in col_list:                                                           #creating a for loop runs over the col list
        is_extreme=np.logical_or(input_data[col]<input_data[col].quantile(lower_threshold),            #creating a list for spliting extreme values from the rest
                             input_data[col]>input_data[col].quantile(upper_threshold))
        input_data.drop(input_data[is_extreme].index, axis=0,inplace=True)         #removing rows with extreme values
    return input_data
        
def standardize_a_column(data,column):
    '''this function transforms a feature to a standardized feature'''
    data[column]=StandardScaler().fit_transform(data[column].values.reshape(-1,1))  #standardizing a column
    return None

def split_dataset(input_df,col_name):
    '''this function splits the dataset as X and Y'''
    X=input_df.drop(columns=[col_name]).to_numpy()                                  #extracting X
    y=input_df[col_name].to_numpy()                                                 #extracting X
    return X,y

def precision_recall_score(classifier,X,y):
    '''this function calculates precision and recall score'''
    from sklearn.metrics import precision_score,recall_score
    y_pred=classifier.predict(X)                                                    #predict y values
    return precision_score(y_true=y,y_pred=y_pred),recall_score(y_true=y,y_pred=y_pred)               #calculating precision and recall scores
#Reading data
dataset=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
#Time
calculate_statistics_and_make_a_plot(dataset['Time'])                           #calculating descriptive statistics and making graph
#Converting time feature to hours 
dataset['Time-Hour']=[i%24 for i in dataset['Time']/3600]                           #converting seconds to hours
dataset['Time-Hour']=dataset['Time-Hour'].astype(int)                               #converting it to integer
dataset.drop(columns=['Time'],inplace=True)                                         #dropping the old time feature. After having time-hour feature, no need to keep this feature any more.
#V1
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V1'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V1'])
#V2
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V2'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V2'])
#V3
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V3'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V3'])
#V4
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V4'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V4'])
#V5
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V5'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V5'])
#V6
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V6'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V6'])
#V7
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V7'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V7'])
#V8
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V8'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V8'])
#V9
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V9'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V9'])
#V10
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V10'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V10'])
#V11
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V11'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V11'])
#V12
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V12'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V12'])
#V13
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V13'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V13'])
#V14
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V14'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V14'])
#V15
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V15'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V15'])
#V16
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V16'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V16'])
#V17
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V17'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V18'])
#V18
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V18'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V18'])
#V19
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V19'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V19'])
#V20
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V20'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V20'])
#V21
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V21'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V21'])
#V22
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V22'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V22'])
#V23
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V23'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V23'])
#V24
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V24'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V24'])
#V25
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V25'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V25'])
#V26
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V26'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V26'])
#V27
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V27'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V27'])
#V28
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['V28'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['V28'])
#Amount
print('Descriptive Statistics\n')
print('Original Data\n')
calculate_statistics_and_make_a_plot(dataset['Amount'])
print('\nData Without Extremes\n')
calculate_statistics_and_make_a_plot_without_extremes(dataset['Amount'],lower_threshold=0,upper_threshold=0.9)
#Target Variable
print('Value Counts')
print(dataset['Class'].value_counts())
print('\nFraund Ratio:{:.3f}'.format(dataset['Class'].mean()))
features_with_extreme_values=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11','V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                              'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
#dataset_v1:Amount feature is standardized.
dataset_v1=dataset.copy()                                                       # creating a copy of dataset.
standardize_a_column(data=dataset_v1,column='Amount')                           # standardizing "Amount" feature.

#dataset_v2
#Amount feature is standardized+Extreme values are flagged
dataset_v2=dataset.copy()                                                       # creating a copy of dataset.
standardize_a_column(data=dataset_v2,column='Amount')                           # standardizing "Amount" feature.
mark_extreme_values(input_data=dataset_v2,                                      # marking extreme values
                    col_list=features_with_extreme_values,
                    lower_threshold=0,upper_threshold=0.99)
#dataset_v3
#Amount feature is standardized+Extreme values are removed
dataset_v3=dataset.copy()                                                       # creating a copy of dataset.
standardize_a_column(data=dataset_v3,column='Amount')                           # standardizing "Amount" feature.
dataset_v3_refined=remove_extremes(dataset_v3,features_with_extreme_values)     # removing extreme values

#dataset_v4
#Amount feature is standardized+Extreme values are replaced with median.
dataset_v4=dataset_v2.copy()                                                    # creating a copy of dataset.
standardize_a_column(data=dataset_v4,column='Amount')                           # standardizing "Amount" feature.
replace_extremes(dataset_v4,features_with_extreme_values)                       # replacing extreme values with median
#in this section, I work on 4 different datasets and find classifiers to predict fraud cases. 
#As I am doing that, I also compare effect of different extreme value replacement techniques on predictive modelling process.
#I have run and found parameters of logistic regression and decision tree on my own with grid search cv and use them directly below.

classifier_dict=pd.DataFrame(columns=['Classifier','Precision_score','Recall_score'])   #creating a dataframe to store models' results

#1st dataset: dataset_v1 --> dataset + Amount feature standardizated and no extreme values modification
X,y=split_dataset(dataset_v1,'Class')                                                     #splitting the features as dependent(y) and independent variables(X).
logistic_regression_1=LogisticRegression(C=0.1,class_weight=None, dual=False,             #creating a logistic regression classifier
                                         fit_intercept=True,intercept_scaling=1,l1_ratio=0.1, 
                                         max_iter=10,multi_class='warn', n_jobs=None, 
                                         penalty='elasticnet',random_state=None, solver='saga', 
                                         tol=0.0001,verbose=0,warm_start=False)
logistic_regression_1.fit(X,y)                                                             #fitting the logistic regression classifier
precision_score,recall_score=precision_recall_score(logistic_regression_1,X,y)             #calculating precision (hit ratio) and recall (get ratio) scores
classifier_dict.loc['logistic_regression_1']=[logistic_regression_1,precision_score,recall_score] #saving the results in the dataframe
decision_tree_1=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,   #creating a decision tree classifier
                                       max_features=0.8, max_leaf_nodes=None,
                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                       min_samples_leaf=1, min_samples_split=0.001,
                                       min_weight_fraction_leaf=0.0, presort=False,
                                       random_state=123, splitter='best')
decision_tree_1.fit(X,y)                                                                          #fitting the decision tree classifier
precision_score,recall_score=precision_recall_score(decision_tree_1,X,y)                          #calculating precision (hit ratio) and recall (get ratio) scores
classifier_dict.loc['decision_tree_1']=[decision_tree_1,precision_score,recall_score]             #saving results in the dataframe

#2nd dataset: dataset_v2  --> dataset + Amount feature standardized + Extreme values flagged
X,y=split_dataset(dataset_v2,'Class')                                                             #splitting the features as dependent(y) and independent variables(X).
logistic_regression_2=LogisticRegression(C=100, class_weight=None, dual=False,                    #creating a logistic regression classifier 
                                       fit_intercept=True, intercept_scaling=1, l1_ratio=0.7, 
                                       max_iter=10, multi_class='warn', n_jobs=None, 
                                       penalty='elasticnet', random_state=None, solver='saga', 
                                       tol=0.0001, verbose=0, warm_start=False)
logistic_regression_2.fit(X,y)                                                                    #fitting the logistic regression classifier
precision_score,recall_score=precision_recall_score(logistic_regression_2,X,y)                    #calculating precision (hit ratio) and recall (get ratio) scores
classifier_dict.loc['logistic_regression_2']=[logistic_regression_2,precision_score,recall_score] #saving the results in the dataframe
decision_tree_2=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,          #creating a decision tree classifier
                                       max_features=0.5, max_leaf_nodes=None,
                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                       min_samples_leaf=1, min_samples_split=0.001,
                                       min_weight_fraction_leaf=0.0, presort=False,
                                       random_state=123, splitter='best')
decision_tree_2.fit(X,y)                                                                          #fitting the decision tree classifier
precision_score,recall_score=precision_recall_score(decision_tree_2,X,y)                          #calculating precision (hit ratio) and recall (get ratio) scores
classifier_dict.loc['decision_tree_2']=[decision_tree_2,precision_score,recall_score]             #saving results in the dataframe

#3rd dataset: dataset_v3 --> dataset+Amount feature standardized + Extreme values are removed
X,y=split_dataset(dataset_v3,'Class')                                                             #splitting the features as dependent(y) and independent variables(X).
logistic_regression_3=LogisticRegression(C=0.01, class_weight=None, dual=False,                   #creating a logistic regression classifier  
                                       fit_intercept=True, intercept_scaling=1, l1_ratio=0.1, 
                                       max_iter=10, multi_class='warn', n_jobs=None, 
                                       penalty='elasticnet', random_state=None, solver='saga', 
                                       tol=0.0001, verbose=0, warm_start=False)

logistic_regression_3.fit(X,y)                                                                    #fitting the logistic regression classifier
precision_score,recall_score=precision_recall_score(logistic_regression_3,X,y)                    #calculating precision (hit ratio) and recall (get ratio) scores
classifier_dict.loc['logistic_regression_3']=[logistic_regression_3,precision_score,recall_score] #saving the results in the dataframe
decision_tree_3=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,          #creating a decision tree classifier
                                       max_features=0.1, max_leaf_nodes=None,
                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                       min_samples_leaf=1, min_samples_split=0.1,
                                       min_weight_fraction_leaf=0.0, presort=False,
                                       random_state=123, splitter='best')
decision_tree_3.fit(X,y)                                                                          #fitting the decision tree classifier
precision_score,recall_score=precision_recall_score(decision_tree_3,X,y)                          #calculating precision (hit ratio) and recall (get ratio) scores
classifier_dict.loc['decision_tree_3']=[decision_tree_3,precision_score,recall_score]             #saving results in the dataframe



#4rd dataset: dataset_v4--> dataset+Amount feature standardized + Extreme values are replaced with median.
X,y=split_dataset(dataset_v4,'Class')                                                             #splitting the features as dependent(y) and independent variables(X).
logistic_regression=LogisticRegression(C=10, class_weight=None, dual=False,                       #creating a logistic regression classifier  
                                       fit_intercept=True, intercept_scaling=1, l1_ratio=0.7, 
                                       max_iter=10, multi_class='warn', n_jobs=None, 
                                       penalty='elasticnet', random_state=None, solver='saga', 
                                       tol=0.0001, verbose=0, warm_start=False)
logistic_regression.fit(X,y)                                                                      #fitting the logistic regression classifier
precision_score,recall_score=precision_recall_score(logistic_regression,X,y)                      #calculating precision (hit ratio) and recall (get ratio) scores
classifier_dict.loc['logistic_regression_4']=[logistic_regression,precision_score,recall_score]   #saving the results in the dataframe
decision_tree_4=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=6,          #creating a decision tree classifier
                                       max_features=0.8, max_leaf_nodes=None,
                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                       min_samples_leaf=1, min_samples_split=0.001,
                                       min_weight_fraction_leaf=0.0, presort=False,
                                       random_state=123, splitter='best')
decision_tree_4.fit(X,y)                                                                          #fitting the decision tree classifier
precision_score,recall_score=precision_recall_score(decision_tree_4,X,y)                          #calculating precision (hit ratio) and recall (get ratio) scores
classifier_dict.loc['decision_tree_4']=[decision_tree_4,precision_score,recall_score]             #saving results in the dataframe
#creating a plot to compare results easier
#creating a new feature for dataset information.
classifier_dict.loc['logistic_regression_1','Data_Process_Summary']='Amount feature is standardized'
classifier_dict.loc['decision_tree_1','Data_Process_Summary']='Amount feature is standardized'
classifier_dict.loc['logistic_regression_2','Data_Process_Summary']='Amount feature is standardized+Extreme values are flagged'
classifier_dict.loc['decision_tree_2','Data_Process_Summary']='Amount feature is standardized+Extreme values are flagged'
classifier_dict.loc['logistic_regression_3','Data_Process_Summary']='Amount feature is standardized+Extreme values are removed'
classifier_dict.loc['decision_tree_3','Data_Process_Summary']='Amount feature is standardized are removed'
classifier_dict.loc['logistic_regression_4','Data_Process_Summary']='Amount feature is standardized+Extreme values are flagged and replaced'
classifier_dict.loc['decision_tree_4','Data_Process_Summary']='Amount feature is standardized+Extreme values are flagged and replaced'
#make a plot
plt.figure(figsize=(12,8))
sns.scatterplot(x='Precision_score',y='Recall_score',data=classifier_dict,style='Data_Process_Summary',s=100)
plt.title('Classifiers Precision-Recall Plot')
plt.xlabel('Precision')
plt.ylabel('Recall');