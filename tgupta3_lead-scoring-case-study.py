'''
    Here we have all the important libraries for reading a CSV file or plotting the graphs.
    We will be importing the model_selection for splitting the data to trained and test data.
    
    statsmodels will be imported to get the statistics of the data frame.
    StandardScaler library wil be used to do the scaling of the data.
    
    we will be importing the LogisticRegression libraries to build the models.
    RFE library will help us in important featuer selections
    
    Variance_inflation_factor library will help us in determining the multiclooinearity
    confusion metrics ill help us in determining lots of factors like accuracy/precision etc
    through the matrix
    
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics


import warnings
warnings.filterwarnings('ignore')
# reading the data
'''
    Here we are reading the data from the csv file present in the below path. From the evaluation perspective,
    we just have to change the below path to the required csv path
'''
path  = r'../input/leads-scoring/Leads.csv'

leads = pd.read_csv(path)
'''
    Here we are creating a copy of the dataset, which can be used at the end.
'''
leads_copy = leads.copy(deep = True)
# viewing the first few records
leads.head()
'''
    Understanding the data
'''

leads.info()
'''
    checking the shape of the data
'''

leads.shape
'''
    describe will help us in determining the max, min, count, quartiles etc for each column.
    This helps us in understanding the column range, type of column mean, standard deviation, outliers, etc
'''

leads.describe()
'''
    Checking for the percentages of null values in each column
'''
round(leads.isnull().sum()/len(leads) * 100,2)
'''
    It is mentioned that the value 'Select' needs to be handled because it is as good as Null. 
    So, converting the value Select as Null
'''

leads.replace('Select', np.nan, inplace= True)
leads.head()
'''
    Rechecking the Null percentages after Select has been made Null
'''

print(round(leads.isnull().sum()/len(leads) * 100,2))
'''
    Dropping the columns 'Lead Profile','How did you hear about X Education' as they have very high percentage
    of missing values 
'''

leads = leads.drop(['Lead Profile','How did you hear about X Education'], axis = 1)
'''
    Check for the counts of distinct values present in Lead Quality column
'''
leads['Lead Quality'].value_counts()
'''
    Converting the missing values in the column Lead Quality column to "Not Sure".
'''

leads['Lead Quality'] = leads['Lead Quality'].replace(np.nan, 'Not Sure')
leads['Lead Quality'].value_counts()
plt.figure(figsize=[10,8])

plt.subplot(2,2,1)
sns.countplot(leads['Asymmetrique Profile Index'])

plt.subplot(2,2,2)
sns.countplot(leads['Asymmetrique Profile Score'])

plt.subplot(2,2,3)
sns.countplot(leads['Asymmetrique Activity Index'])

plt.subplot(2,2,4)
sns.countplot(leads['Asymmetrique Activity Score'])

plt.show()
'''
    Doing a pairplot inorder to check the correlation of columns 'Asymmetrique Activity Score', 
    'Asymmetrique Profile Score', 'Converted'.
'''
sns.pairplot(leads[['Asymmetrique Activity Score', 'Asymmetrique Profile Score','Converted']])
plt.show()
'''
    Converting the data to a pivot table for better understanding of the data. Now, we can clealry see how the values in the column
    Asymmetrique Activity Index are present against the "Converted" values

'''
df_pivot = pd.pivot_table(data = leads, columns = 'Converted', index = 'Asymmetrique Activity Index',values = 'Prospect ID', aggfunc='count')
df_pivot
'''
    Plotting the pivot table in a graph to make it more readable. 
'''
df_pivot.unstack('Asymmetrique Activity Index').plot(kind='bar')
plt.show()
'''
    Converting the data to a pivot table for better understanding of the data. Now, we can clealry see how the values in the column
    Asymmetrique Profile Index are present against the "Converted" values

'''
df_pivot = pd.pivot_table(data = leads, columns = 'Converted', index = 'Asymmetrique Profile Index',values = 'Prospect ID', aggfunc='count')
df_pivot
'''
    Plotting the pivot table in a graph to make it more readable. 
'''
df_pivot.unstack('Asymmetrique Profile Index').plot(kind='bar')
plt.show()
'''
    Converting the data to a pivot table for better understanding of the data. Now, we can clealry see how the values in the column
    Asymmetrique Activity Score are present against the "Converted" values

'''
df_pivot = pd.pivot_table(data = leads, columns = 'Converted', index = 'Asymmetrique Activity Score',values = 'Prospect ID', aggfunc='count')
df_pivot
'''
    Plotting the pivot table in a graph to make it more readable. 
'''
df_pivot.unstack('Asymmetrique Activity Score').plot(kind='bar')
plt.show()
'''
    Converting the data to a pivot table for better understanding of the data. Now, we can clealry see how the values in the column
    Asymmetrique Profile Score are present against the "Converted" values

'''
df_pivot = pd.pivot_table(data = leads, columns = 'Converted', index = 'Asymmetrique Profile Score',values = 'Prospect ID', aggfunc='count')
df_pivot
'''
    Plotting the pivot table in a graph to make it more readable. 
'''
df_pivot.unstack('Asymmetrique Profile Score').plot(kind='bar')
plt.show()
'''
   Looking at the correlation of the Scores with the values in Converted column
'''
leads[['Asymmetrique Activity Score', 'Asymmetrique Profile Score','Converted']].corr()
# dropping the above mentioned columns
leads = leads.drop(['Asymmetrique Activity Score', 'Asymmetrique Profile Score','Asymmetrique Activity Index', 'Asymmetrique Profile Index'], axis = 1)
'''
    Finding the percentage of null columns post removing few columns and imputing of other columns
'''
round(leads.isnull().sum()/len(leads) * 100,2)
# checking the frequency of values in the City column, so it can be handled appropriately
print(leads['City'].value_counts())
# replacing the missing values with the mode i.e Mumbai
leads.City.replace(np.nan,'Mumbai',inplace=True)
# checking the frequency of the values
print(leads['Specialization'].value_counts())
# replcaing null values with Others
leads.Specialization.replace(np.nan,'Others',inplace=True)
# checking the frequency of the values
print(leads['Tags'].value_counts())
# replacing the missing values with 'Will revert after reading the email'
leads.Tags.replace(np.nan,'Will revert after reading the email',inplace=True)
# checking the frequency of the values
print(leads['What matters most to you in choosing a course'].value_counts())
# replacing the null values
leads['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects',inplace=True)
# checking the frequency of the values
print(leads['What is your current occupation'].value_counts())
# replacing the null values
leads['What is your current occupation'].replace(np.nan,'Unemployed',inplace=True)
# checking the frequency of the values
print(leads['Country'].value_counts())
# replacing the null values
leads.Country.replace(np.nan,'India',inplace=True)
'''
    Now, we have handled most of the data and have done all the imputations on the remaining columns as well. 
    Checking the null percentage again to see if the data has been cleaned
'''
round(leads.isnull().sum()/len(leads) * 100,2)
'''
    Dropping the remianing columns which have around 1% of null data, deleting these many records will not impact much.
'''
leads = leads.dropna()
leads.shape
'''
    Checking the null values again
'''
round(leads.isnull().sum()/len(leads) * 100,2)
# plotting the pairplots of the numeric columns to see how they affect each other
sns.pairplot(leads)
plt.show()
'''
    Printing the heatmap of the entire data to find out the relation/dependency of each column.
'''
sns.heatmap(leads.corr(),annot = True)
plt.show()
'''
 Plotting the countplot to see map the frequency of the values in the columns with respect to the values in the column Converted 
'''
plt.figure(figsize = (30,20))

plt.subplot(2,3,1)
sns.countplot(x = 'Do Not Email',hue = 'Converted', data = leads)
plt.xticks(rotation = 90)

plt.subplot(2,3,2)
sns.countplot(x = 'Lead Origin',hue = 'Converted', data = leads)
plt.xticks(rotation = 45)

plt.subplot(2,3,3)
sns.countplot(x = 'Lead Source',hue = 'Converted', data = leads)
plt.xticks(rotation = 45)

plt.subplot(2,3,4)
sns.countplot(x = 'Do Not Call',hue = 'Converted', data = leads)
plt.xticks(rotation = 90)

plt.subplot(2,3,5)
sns.countplot(x = 'TotalVisits',hue = 'Converted', data = leads)
plt.xticks(rotation = 90)

plt.subplot(2,3,6)
sns.countplot(x = 'Last Notable Activity',hue = 'Converted', data = leads)
plt.xticks(rotation = 90)

plt.show()
'''
 Plotting the countplot to see map the frequency of the values in the columns with respect to the values in the column Converted
 
'''

plt.figure(figsize = (20,16))

plt.subplot(2,3,1)
sns.countplot(x = 'Page Views Per Visit',hue = 'Converted', data = leads)
plt.xticks(rotation = 90)

plt.subplot(2,3,2)
sns.countplot(x = 'Country',hue = 'Converted', data = leads)
plt.xticks(rotation = 90)

plt.subplot(2,3,3)
sns.countplot(x = 'Specialization',hue = 'Converted', data = leads)
plt.xticks(rotation = 90)

plt.subplot(2,3,4)
sns.countplot(x = 'What is your current occupation',hue = 'Converted', data = leads)
plt.xticks(rotation = 90)

plt.subplot(2,3,5)
sns.countplot(x = 'What matters most to you in choosing a course',hue = 'Converted', data = leads)
plt.xticks(rotation = 90)

plt.show()
'''
Plotting the countplot to see map the frequency of the values in the columns with respect to the values in the column Converted
'''
plt.figure(figsize = (20,16))

plt.subplot(2,3,1)
sns.countplot(x = 'Search',hue = 'Converted', data = leads)

plt.subplot(2,3,2)
sns.countplot(x = 'Magazine',hue = 'Converted', data = leads)

plt.subplot(2,3,3)
sns.countplot(x = 'Newspaper Article',hue = 'Converted', data = leads)

plt.subplot(2,3,4)
sns.countplot(x = 'X Education Forums',hue = 'Converted', data = leads)

plt.subplot(2,3,5)
sns.countplot(x = 'Newspaper',hue = 'Converted', data = leads)

plt.subplot(2,3,6)
sns.countplot(x = 'Digital Advertisement',hue = 'Converted', data = leads)

plt.show()
'''
Plotting the countplot to see map the frequency of the values in the columns with respect to the values in the column Converted

'''
plt.figure(figsize = (30,40))

plt.subplot(2,3,1)
sns.countplot(x = 'Through Recommendations',hue = 'Converted', data = leads)

plt.subplot(2,3,2)
sns.countplot(x = 'Receive More Updates About Our Courses',hue = 'Converted', data = leads)

plt.subplot(2,3,3)
sns.countplot(x = 'Tags',hue = 'Converted', data = leads)
plt.xticks(rotation = 90)

plt.subplot(2,3,4)
sns.countplot(x = 'Lead Quality',hue = 'Converted', data = leads)
plt.xticks(rotation = 90)

plt.subplot(2,3,5)
sns.countplot(x = 'Update me on Supply Chain Content',hue = 'Converted', data = leads)

plt.subplot(2,3,6)
sns.countplot(x = 'Get updates on DM Content',hue = 'Converted', data = leads)

plt.show()
'''
Plotting the countplot to see map the frequency of the values in the columns with respect to the values in the column Converted

'''   
plt.figure(figsize = (20,16))

plt.subplot(2,3,1)
sns.countplot(x = 'City',hue = 'Converted', data = leads)
plt.xticks(rotation = 90)

plt.subplot(2,3,2)
sns.countplot(x = 'I agree to pay the amount through cheque',hue = 'Converted', data = leads)


plt.subplot(2,3,3)
sns.countplot(x = 'A free copy of Mastering The Interview',hue = 'Converted', data = leads)

plt.show()
plt.figure(figsize=(16,8))

plt.subplot(2,3,1)
sns.boxplot(x = 'TotalVisits', data = leads)

plt.subplot(2,3,2)
sns.boxplot(x = 'Total Time Spent on Website',hue = 'Converted', data = leads)

plt.subplot(2,3,3)
sns.boxplot(x = 'Page Views Per Visit',hue = 'Converted', data = leads)

plt.show()
'''
    Defining a function in which a dataframe and a column(for which we want to remove the outliers) are passed. 
    The dataframe returned is sans the outliers. The range of values being returned are between 0.05 and 0.97 percentiles
'''
def remove_outliers(df,col):
    
    Q1 = df[col].quantile(0.05)
    Q3 = df[col].quantile(0.97)
    df = df[(df[col] >=Q1) &(df[col] <=Q3)]
    
    return df
'''
    
    Taking a backup of the existing data frame so, that we can use the actual one, in case we want the DF with no 
    outlier treatment.
    
'''

leads_outlier = leads.copy(deep = True)
len(leads_outlier)
'''
    Here the outliers for three columns TotalVisits, Total Time Spent on Website and Page Views Per Visit are removed
'''
leads_outlier = remove_outliers(leads_outlier,'TotalVisits')
leads_outlier = remove_outliers(leads_outlier,'Total Time Spent on Website')
leads_outlier = remove_outliers(leads_outlier,'Page Views Per Visit')
'''

   Checking the number of records left post the outlier treatment

'''

len(leads_outlier)
'''
   
   Plotting the boxplots after removing the outliers from the columns TotalVisits, Total Time Spent on Website and Page Views Per Visit
   
'''
plt.figure(figsize=(16,8))

plt.subplot(2,3,1)
sns.boxplot(x = 'TotalVisits', data = leads_outlier)

plt.subplot(2,3,2)
sns.boxplot(x = 'Total Time Spent on Website',hue = 'Converted', data = leads_outlier)

plt.subplot(2,3,3)
sns.boxplot(x = 'Page Views Per Visit',hue = 'Converted', data = leads_outlier)

plt.show()
# Since the outliers were treated in another dataframe, updating the original dataframe with the cleansed one
leads = leads_outlier.copy(deep = True)
leads.head()
leads.shape
'''
    Converting the binary variables to 1 and 0 and using the map function to do the same.
'''
# the list of variables having binary value only
varlist =  ['Do Not Email', 'Do Not Call','Newspaper Article', 'X Education Forums', 'Newspaper','Digital Advertisement','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque','A free copy of Mastering The Interview','Search']

# Defining the map function to change Yes to 1 and No to 0
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

leads[varlist] = leads[varlist].apply(binary_map)
leads = leads.drop(varlist, axis =1 )
# checking how the dataframe looks like after the mapping is done
leads.head()
'''
    Converting all the categorical columns to the dummy variables.

'''
# creating a list of the columns
var_list = ['Lead Origin', 'Lead Source','Country','Specialization', 'What is your current occupation',
            'What matters most to you in choosing a course', 'Magazine', 'Last Activity',
            'Through Recommendations','Tags', 'Lead Quality','City','Last Notable Activity']

dummy1 = pd.get_dummies(leads[var_list], drop_first=True)

# Adding the results to the master dataframe
leads = pd.concat([leads, dummy1], axis=1)
leads = leads.drop(var_list, axis =1 )
leads.shape
# The features are assigned to X
X = leads.drop(['Prospect ID','Converted','Lead Number'], axis=1)
X.head()
# the target is assigned to y
y = leads['Converted']

y.head()
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
scaler = StandardScaler()

# the original numerical columns are stored in another variable. 
# All the other columns are either converted using binary mapping method or are dummy variables and 
# therefore do not require standardization since standardscaler is being used here
cont_varlist=['TotalVisits','Total Time Spent on Website','Page Views Per Visit']
X_train[cont_varlist] = scaler.fit_transform(X_train[cont_varlist])

X_train.head()
X_train[cont_varlist].describe()
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()
logreg = LogisticRegression()
rfe = RFE(logreg, 14)
rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
col
X_train_sm_2 = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm_2, family = sm.families.Binomial())
res2 = logm2.fit()
res2.summary()
col3 = col.drop('Tags_number not provided', 1)
col3
X_train_sm_3 = sm.add_constant(X_train[col3])
logm3 = sm.GLM(y_train,X_train_sm_3, family = sm.families.Binomial())
res3 = logm3.fit()
res3.summary()
col4 = col3.drop('Lead Source_Welingak Website', 1)
col4
X_train_sm_4 = sm.add_constant(X_train[col4])
logm4 = sm.GLM(y_train,X_train_sm_4, family = sm.families.Binomial())
res4 = logm4.fit()
res4.summary()
'''
    Created a function to calculate and return VIF. 
'''

def calculate_vif(df, col):
    vif = pd.DataFrame()
    vif['Features'] = df[col].columns
    vif['VIF'] = [variance_inflation_factor(df[col].values, i) for i in range(df[col].shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif
'''
 calculating VIF using the function created above
'''

calculate_vif(X_train,col4)
y_train_pred = res4.predict(X_train_sm_4)
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob_model':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()
# ro begin with, the threshold can be given as 0.5
y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob_model.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()
'''
    This method will help us in getting the confusion matrix, accuracy score, sensitivity, specificty values also.
    This method will also help us in finding the precision and recall.
'''
def calculate_all_metrics(df,col_conv,col_pred):
    confusion = metrics.confusion_matrix(df[col_conv], df[col_pred])
    print("Confusion matrix obtained is \n{val}".format(val=confusion))
    print("\nAccuracy score obtained is {val}".format(val = metrics.accuracy_score
                                                    (df[col_conv], df[col_pred])))
    TP = confusion[1,1] # true positive 
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] # false positives
    FN = confusion[1,0] # false negatives
    print("\nSensitivity for the above confusion matrix obtained is = ", TP / float(TP+FN))
    print("\nSpecificity for the above confusion matrix obtained is = ", TN / float(TN+FP))
    print("\nFalse Positive Rate for the above confusion matrix obtained is = ", FP/ float(TN+FP))
    print ("\nPrecision for the above confusion matrix obtained is = ", TP / float(TP+FP))
    print("\nRecall for the above confusion matrix obtained is = ", TP / float(TP+FN))
    print ("\nNegative predictive value for the above confusion matrix obtained is = ", TN / float(TN+ FN))
calculate_all_metrics(y_train_pred_final,'Converted','Predicted')
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob_model, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob_model)
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob_model.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
# Calculating the accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# plotting a graph for the various probability values

plt.figure(figsize=(15,10))
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.vlines(x=0.27, ymax=1, ymin=0, colors="black", linestyles="--")
plt.vlines(x=0.3, ymax=1, ymin=0, colors="y", linestyles="--")
plt.show()
y_train_pred_2 = pd.DataFrame({'Converted':y_train.values, 'Converted_prob_model':y_train_pred})
y_train_pred_2['Prospect ID'] = y_train.index
y_train_pred_2.head()
# using 0.27 as the index
y_train_pred_2['Predicted'] = y_train_pred_2.Converted_prob_model.map(lambda x: 1 if x > 0.27 else 0)
y_train_pred_2.head()
'''

    Calculating all the metrics after getting a final data frame.
    
'''
calculate_all_metrics(y_train_pred_2,'Converted','Predicted')
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_2.Converted, y_train_pred_2.Converted_prob_model, drop_intermediate = False )
draw_roc(y_train_pred_2.Converted, y_train_pred_2.Converted_prob_model)
X_test[cont_varlist] = scaler.transform(X_test[cont_varlist])
X_test = X_test[col4]
X_test_sm = sm.add_constant(X_test)
y_test_pred = res4.predict(X_test_sm)
y_test_pred_1 = pd.DataFrame(y_test_pred)
y_test_df = pd.DataFrame(y_test)
y_test_df['Prospect ID'] = y_test_df.index

y_test_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

y_test_pred_final = pd.concat([y_test_df, y_test_pred_1],axis=1)
y_test_pred_final.head()
y_test_pred_final= y_test_pred_final.rename(columns={ 0 : 'Converted_prob'})
y_test_pred_final.head()
y_test_pred_final['final_predicted'] = y_test_pred_final.Converted_prob.map(lambda x: 1 if x > 0.27 else 0)
y_test_pred_final.head()
'''
    find out all the metrics like precision,recall, accuracy,confusion matrix etc for the final data frame
'''
calculate_all_metrics(y_test_pred_final,'Converted','final_predicted')
'''
    Conversion rate
'''
# people getting converted actually
l1 = len(y_test_pred_final[y_test_pred_final['Converted']==1])

# people getting converted as per the model
l2 = len(y_test_pred_final[y_test_pred_final['final_predicted']==1])

# calculating and printing the conversion rate
print ("Conversion rate as per the model is {0}" .format(round(100*(l2/l1),2)))
'''
    this final_df will help us in determining the prospect id for the people who will be predicted (by the model) 
    to be converted as a customer
'''
final_df = y_test_pred_final[y_test_pred_final['final_predicted']==1]
final_df.head()
print("Total number of records in the final dataframe are ", len(final_df))
final_df["LeadScore"] = final_df["Converted_prob"] * 100
final_df.head()
final_df.describe()
'''
    Here we are finding the count of the records based on the some intervals.
    eg. count of people who will be converted(according to the model) and have probability of getting converted is
    greater than 0.8 and less than 0.9 etc
'''
df_greater_90 = list(final_df['Converted_prob'].between(0.9, 1, inclusive=False)).count(True)
df_in_80_90 =list( final_df['Converted_prob'].between(0.8, 0.9, inclusive=False)).count(True)
df_in_70_80 =list( final_df['Converted_prob'].between(0.7, 0.8, inclusive=False)).count(True)
df_in_60_70 =list( final_df['Converted_prob'].between(0.6, 0.7, inclusive=False)).count(True)
df_in_50_60 =list( final_df['Converted_prob'].between(0.5, 0.6, inclusive=False)).count(True)
df_in_40_50 =list( final_df['Converted_prob'].between(0.4, 0.5, inclusive=False)).count(True)
df_in_30_40 =list( final_df['Converted_prob'].between(0.3, 0.4, inclusive=False)).count(True)
df_in_20_30 =list( final_df['Converted_prob'].between(0.2, 0.3, inclusive=False)).count(True)
df_in_10_20 =list( final_df['Converted_prob'].between(0.1, 0.2, inclusive=False)).count(True)
df_in_0_10 = list(final_df['Converted_prob'].between(0, 0.1, inclusive=False)).count(True)


print('\nnumber of people getting converted with probability greater than 0.9 is {val}'.format(val = df_greater_90))
print('\nnumber of people getting converted with probability in range 0.8 - 0.9 is {val}'.format(val = df_in_80_90))
print('\nnumber of people getting converted with probability in range 0.7 - 0.8 is {val}'.format(val = df_in_70_80))
print('\nnumber of people getting converted with probability in range 0.6 - 0.7 is {val}'.format(val = df_in_60_70))
print('\nnumber of people getting converted with probability in range 0.5 - 0.6 is {val}'.format(val = df_in_50_60))
print('\nnumber of people getting converted with probability in range 0.4 - 0.5 is {val}'.format(val = df_in_40_50))
print('\nnumber of people getting converted with probability in range 0.3 - 0.4 is {val}'.format(val = df_in_30_40))
print('\nnumber of people getting converted with probability in range 0.2 - 0.3 is {val}'.format(val = df_in_20_30))
print('\nnumber of people getting converted with probability in range 0.1 - 0.2 is {val}'.format(val = df_in_10_20))
print('\nnumber of people getting converted with probability in range 0.0 - 0.1 is {val}'.format(val = df_in_0_10))
df_hot_leads = final_df[final_df['Converted_prob'] >= 0.8]
df_others = final_df[final_df['Converted_prob'] < 0.8]
df_hot_leads.head()
df_others.head()
print("Number of hot leads is ", df_hot_leads.shape[0])
