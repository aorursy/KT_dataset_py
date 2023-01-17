#Ignore warnings 

import warnings

warnings.filterwarnings('ignore')
#Import the necessary libraries

#Basic Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# setting restriction on the number of rows and coulmns displayed in output

pd.set_option('display.max_columns',999)

pd.set_option('display.max_rows',200)



# Libraries for data modelling

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import precision_recall_curve
# Creating a class color for setting print formatting

class color:

   BLUE = '\033[94m'

   BOLD = '\033[1m'

   END = '\033[0m'
#Read and Understand the data



df = pd.read_csv("../input/leadsscore/Leads.csv")

print(color.BOLD+color.BLUE+'Shape of the dataframe df : {}'.format(df.shape) +color.END)

df.head()
df.describe()
df.info()
#Replacing the SELECT value with null value

df = df.replace('Select',np.nan)
#Checking the number of missing values and its percentage

Total_missing = df.isnull().sum().sort_values(ascending = False)

Total_missing_Perc = (100*df.isnull().sum()/df.shape[0]).sort_values(ascending = False)

df_missing_values = pd.concat([Total_missing,Total_missing_Perc], axis=1, keys=['Total_missing_values', 'Percent_missing_values'])

df_missing_values.head(30)
# Checking how many columns have more than 45% of missing data



print(color.BOLD+ color.BLUE+'Total no of columns with missing values more than 45% : {}'.format(df_missing_values[df_missing_values['Percent_missing_values'] >= 45].shape[0])+ color.END)
#Create a new dataframe named df_cleaned with all columns with data misisng < 45% for our  further analysis



df_cleaned = df.loc[:,(100*df.isnull().sum()/df.shape[0]).sort_values(ascending = False) < 45]

df_cleaned.head()
#Checking the columns present in dataframe

df_cleaned.columns
#Dropping Prospect ID and Lead Number and Tags columns as these columns are scroe variables created by the sales team and may not be available.

df_cleaned = df_cleaned.drop(['Prospect ID','Lead Number','Tags'],axis=1)
#Checking the no of unique values in each column

df_cleaned.nunique()
#Dropping all columns with one unique value



df_cleaned.drop(['Magazine','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque'], axis=1,inplace = True)

#Verifying the unique values for each columns again to confirm

df_cleaned.nunique()
# Get the value counts of all the columns



for column in df_cleaned:

    

    print(df_cleaned[column].astype('category').value_counts())

    print('___________________________________________________')
# Defining the map function

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})

varlist =  ['Do Not Email', 'Do Not Call','Search','X Education Forums','Newspaper Article','Newspaper','Digital Advertisement','Through Recommendations','A free copy of Mastering The Interview']



# applying the map function

df_cleaned[varlist] = df_cleaned[varlist].apply(binary_map)
# Checking the missing value columns

Total_missing = df_cleaned.isnull().sum().sort_values(ascending = False)

Total_missing_Perc = (100*df.isnull().sum()/df.shape[0]).sort_values(ascending = False)

df_missing_values = pd.concat([Total_missing,Total_missing_Perc], axis=1, keys=['Total_missing_values', 'Percent_missing_values'])

df_missing_values.head(10)
# Data imputation for column 'City'



plt.figure(figsize=(8,12))

sns.countplot(df_cleaned.City)

plt.title("Checking the mode of column city- UniVariate Analysis")

plt.show()
df_cleaned.City.value_counts()
s = df_cleaned.City.value_counts()

ax=s.plot.bar(width=.8) 



for i, v in s.reset_index().iterrows():

    ax.text(i, v.City , v.City, color='green')
# Most of the leads are from Mumbai , so we can map the NAN values of 'City' to Mumbai

df_cleaned['City'] = df_cleaned['City'].replace(np.nan,'Mumbai')
# Data imputation for column 'Specialization'



plt.figure(figsize=(25,10))

sns.countplot(df_cleaned.Specialization)

plt.xticks(rotation=90)

plt.title("Checking the mode of column Specialization-UniVariate Analysis")

plt.show()

df_cleaned.Specialization.value_counts()
# As the specialization is distributed across multiple values ,we can impute it with mode and therefore we can put the Nan values as 'Unknown'



df_cleaned['Specialization']=df_cleaned['Specialization'].replace(np.nan,'Unknown')
# Data imputation for column 'What matters most to you in choosing a course'



plt.figure(figsize=(12,10))

sns.countplot(df_cleaned['What matters most to you in choosing a course'])

plt.xticks(rotation=90)

plt.title("Checking the mode of column 'What matters most to you in choosing a course'-UniVariate Analysis")

plt.show()

df_cleaned['What matters most to you in choosing a course'].value_counts()
# Most of the values here is related to a single value 'Better Career Prospects' so we can map the NAN values to Better Career Prospects



df_cleaned['What matters most to you in choosing a course']=df_cleaned['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# Data imputation for column 'What is your current occupation'



plt.figure(figsize=(12,10))

sns.countplot(df_cleaned['What is your current occupation'])

plt.xticks(rotation=90)

plt.title("Checking the mode of column 'What is your current occupation'-UniVariate Analysis")

plt.show()

df_cleaned['What is your current occupation'].value_counts()
# Replace all the Nan values with 'Unemployed' as it is the majority



df_cleaned['What is your current occupation']=df_cleaned['What is your current occupation'].replace(np.nan,'Unemployed')
# Data imputation for column 'Country'



plt.figure(figsize=(25,10))

sns.countplot(df_cleaned['Country'])

plt.xticks(rotation=90)

plt.title("Checking the mode of column 'Country'-UniVariate Analysis")

plt.show()

df_cleaned['Country'].value_counts()
#Replacing all the Nan values with India as it is the majority



df_cleaned['Country']=df_cleaned['Country'].replace(np.nan,'India')
# As the columns TotalVisits, Page Views Per Visit , Last Activity and Lead Source have less than 2% of Nan Values we chose to drop those rows .



df_cleaned.dropna(inplace=True)
#Final check of missing values 

Total_missing = df_cleaned.isnull().sum().sort_values(ascending = False)

Total_missing_Perc = (100*df_cleaned.isnull().sum()/df_cleaned.shape[0]).sort_values(ascending = False)

df_missing_values = pd.concat([Total_missing,Total_missing_Perc], axis=1, keys=['Total_missing_values', 'Percent_missing_values'])

df_missing_values
# Number of rows retained after data cleaning



print(round(100*(df_cleaned.shape[0] / df.shape[0]),2))

            
#Check the shape of cleaned dataframe

df_cleaned.shape
# Get the value counts of all the columns



for column in df_cleaned:

    

    print(df_cleaned[column].astype('category').value_counts())

    print('___________________________________________________')
df_cleaned=df_cleaned.replace(to_replace=['bing','google','Click2call','Press_Release','Social Media','Live Chat','Pay per Click Ads','welearnblog_Home','NC_EDM','WeLearn','blog','testone','youtubechannel'],

           value= 'Others')
df_cleaned['Lead Source'].value_counts()
df_cleaned=df_cleaned.replace(to_replace=['Email Marked Spam','View in browser link Clicked','Resubscribed to emails','Form Submitted on Website','Email Received','Approached upfront'],

           value= 'Others')
df_cleaned['Last Notable Activity'].value_counts()
#First we will check the numerical variables against converted.

sns.set(font_scale=1)

plt.figure(figsize=(20, 12))

plt.subplot(2,3,1)

sns.boxplot(x = 'Converted', y = 'TotalVisits', data = df_cleaned)

plt.subplot(2,3,2)

sns.boxplot(x = 'Converted', y = 'Total Time Spent on Website', data = df_cleaned)

plt.subplot(2,3,3)

sns.boxplot(x = 'Converted', y = 'Page Views Per Visit', data = df_cleaned)

plt.show()
# outlier treatment for TotalVisits

#PLots for outlier analysis

sns.boxplot(df_cleaned.TotalVisits)

plt.show()



# Defining outlier treatment function  

def outlier_treatment(datacolumn):

 sorted(datacolumn)

 Q1,Q3 = np.percentile(datacolumn , [25,75])

 IQR = Q3 - Q1

 lower_range = Q1 - (1.5 * IQR)

 upper_range = Q3 + (1.5 * IQR)

 return lower_range,upper_range



#Calculating IQR

lowerbound,upperbound = outlier_treatment(df_cleaned.TotalVisits)

print(lowerbound,upperbound)



#check outliers for the TotalVisits

clmn = 'TotalVisits'

Total_no_of_outliers = df_cleaned[(df_cleaned['TotalVisits'] > upperbound) | (df_cleaned['TotalVisits'] < lowerbound)] .shape[0]



print(color.BOLD + color.BLUE + 'Total no of outliers for column {0} : {1}'.format(clmn,Total_no_of_outliers ) + color.END)



#calculate % of outliers in the data

100*Total_no_of_outliers/df_cleaned.shape[0]
#Delete the TotalVisits outliers from the dataset as it is very less

df_cleaned = df_cleaned[(df_cleaned['TotalVisits'] <= upperbound) & (df_cleaned['TotalVisits'] >= lowerbound )]


#PLots for outlier analysis - Total Time Spent on Website

sns.boxplot(df_cleaned['Total Time Spent on Website'])

plt.show()





# Calculating IQR 

lowerbound,upperbound = outlier_treatment(df_cleaned['Total Time Spent on Website'])



print(lowerbound,upperbound)



#check outliers for the Total_Time_Spent_on_Website

clmn = 'Total Time Spent on Website'

Total_no_of_outliers = df_cleaned[(df_cleaned['Total Time Spent on Website'] > upperbound) | (df_cleaned['Total Time Spent on Website'] < lowerbound)] .shape[0]



print(color.BOLD + color.BLUE + 'Total no of outliers for column {0} : {1}'.format(clmn,Total_no_of_outliers ) + color.END)



#calculate % of outliers in the data

100*Total_no_of_outliers/df_cleaned.shape[0]
#Delete the 'Total Time Spent on Website' outliers from the dataset as it is very less 

df_cleaned = df_cleaned[(df_cleaned['Total Time Spent on Website']<= upperbound) & (df_cleaned['Total Time Spent on Website'] >= lowerbound )]
#PLots for outlier analysis - Page Views Per Visit

sns.boxplot(df_cleaned['Page Views Per Visit'])

plt.show()





# Calculating IQR 

lowerbound,upperbound = outlier_treatment(df_cleaned['Page Views Per Visit'])



print(lowerbound,upperbound)



#check outliers for the Page Views Per Visit

clmn = 'Page Views Per Visit'

Total_no_of_outliers = df_cleaned[(df_cleaned['Page Views Per Visit'] > upperbound) | (df_cleaned['Page Views Per Visit'] < lowerbound)] .shape[0]



print(color.BOLD + color.BLUE + 'Total no of outliers for column {0} : {1}'.format(clmn,Total_no_of_outliers ) + color.END)



#calculate % of outliers in the data

100*Total_no_of_outliers/df_cleaned.shape[0]
#Delete the 'Total Time Spent on Website' outliers from the dataset as it is very less

df_cleaned = df_cleaned[(df_cleaned['Page Views Per Visit']<= upperbound) & (df_cleaned['Page Views Per Visit'] >= lowerbound )]


plt.figure(figsize=(20, 7))

plt.subplot(1,3,1)

sns.boxplot(x = 'Converted', y = 'TotalVisits', data = df_cleaned)

plt.title("TotalVisits vs Converted")

plt.subplot(1,3,2)

sns.boxplot(x = 'Converted', y = 'Total Time Spent on Website', data = df_cleaned)

plt.title("Total Time spent on Website vs Converted")

plt.subplot(1,3,3)

sns.boxplot(x = 'Converted', y = 'Page Views Per Visit', data = df_cleaned)

plt.title("Page views per visit vs Converted")

plt.show()


sns.catplot(x = "TotalVisits", hue = "Converted", data = df_cleaned, kind = "count") #, aspect =3

plt.show()

g = sns.catplot(x = "Page Views Per Visit", hue = "Converted", data = df_cleaned, kind = "count", aspect =2) 

g.set_xticklabels(rotation=90)

plt.show()

plt.figure(figsize = (60,60)) 

sns.catplot(x = "Lead Origin", hue = "Converted", data = df_cleaned, kind = "count", aspect =1.5)

plt.show()

g = sns.catplot(x = "Lead Source", hue = "Converted", data = df_cleaned, kind = "count", aspect =3)

g.set_xticklabels(rotation=90)

plt.show()

sns.catplot(x = "Do Not Email", hue = "Converted", data = df_cleaned, kind = "count")#, aspect =3.5)

plt.show()

sns.catplot(x = "Do Not Call", hue = "Converted", data = df_cleaned, kind = "count")#, aspect =3.5)

plt.show()

g = sns.catplot(x = "Last Activity", hue = "Converted", data = df_cleaned, kind = "count", aspect =3)

g.set_xticklabels(rotation=90)

plt.show()

g = sns.catplot(x = "Country", hue = "Converted", data = df_cleaned, kind = "count", aspect =3)

g.set_xticklabels(rotation=90)

plt.show()

g = sns.catplot(x = "Specialization", hue = "Converted", data = df_cleaned, kind = "count", aspect =2)

g.set_xticklabels(rotation=90)

plt.show()

g = sns.catplot(x = "What is your current occupation", hue = "Converted", data = df_cleaned, kind = "count")

g.set_xticklabels(rotation=90)

plt.show()

g = sns.catplot(x = "What matters most to you in choosing a course", hue = "Converted", data = df_cleaned, kind = "count")

g.set_xticklabels(rotation=90)

plt.show()

g = sns.catplot(x = "Search", hue = "Converted", data = df_cleaned, kind = "count")#, aspect = 3)

plt.show()

g = sns.catplot(x = "Newspaper Article", hue = "Converted", data = df_cleaned, kind = "count")#, aspect = 3)

plt.show()

g = sns.catplot(x = "X Education Forums", hue = "Converted", data = df_cleaned, kind = "count")#, aspect = 3)

plt.show()

g = sns.catplot(x = "Newspaper", hue = "Converted", data = df_cleaned, kind = "count")#, aspect = 3)

plt.show()

g = sns.catplot(x = "Digital Advertisement", hue = "Converted", data = df_cleaned, kind = "count")#, aspect = 3)

plt.show()

g = sns.catplot(x = "Through Recommendations", hue = "Converted", data = df_cleaned, kind = "count")#, aspect = 3)

plt.show()

g = sns.catplot(x = "City", hue = "Converted", data = df_cleaned, kind = "count", aspect = 3)

plt.show()

g = sns.catplot(x = "A free copy of Mastering The Interview", hue = "Converted", data = df_cleaned, kind = "count")#, aspect = 3)

plt.show()

g = sns.catplot(x = "Last Notable Activity", hue = "Converted", data = df_cleaned, kind = "count", aspect = 3)

g.set_xticklabels(rotation=90)

plt.show()
#Dropping the columns which doesnt give much inferences

df_cleaned = df_cleaned.drop(['What matters most to you in choosing a course','Search','Newspaper Article','X Education Forums','Newspaper',

           'Digital Advertisement','Through Recommendations','Page Views Per Visit','TotalVisits','Last Activity','A free copy of Mastering The Interview','Country'],1)
# Final columns present in a cleaned data frame

df_cleaned.columns
# Creating dummies

dummy = pd.get_dummies(df_cleaned[['Lead Origin','Lead Source','Specialization','What is your current occupation','City',

                                   'Last Notable Activity']], drop_first=True)

dummy.head()
# Adding the results back to the cleaned dataframe

df_cleaned = pd.concat([df_cleaned, dummy], axis=1)

df_cleaned.head()
# We have created dummies for the below variables, so we can drop them

# Dropping the columns whose dummies have been created

df_cleaned=df_cleaned.drop(['Lead Origin','Lead Source','Specialization','What is your current occupation','City','Last Notable Activity'], axis = 1)
df_cleaned.head()
df_cleaned.shape
# Putting feature variable to X

X = df_cleaned.drop(['Converted'], axis=1)

X.head()
# Putting response variable to y

y = df_cleaned['Converted']



y.head()
# Splitting the data into train and test:

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
scaler = StandardScaler()



X_train[['Total Time Spent on Website']] = scaler.fit_transform(X_train[['Total Time Spent on Website']])



X_train.head()
# Plotting the the correlation matrix 

plt.figure(figsize = (60,60))        # Size of the figure

sns.heatmap(df_cleaned.corr(),annot = True)

plt.show()
#Checking the present lead conversion rate

convert = (sum(df_cleaned['Converted'])/len(df_cleaned['Converted'].index))*100

convert
# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
logreg = LogisticRegression()

from sklearn.feature_selection import RFE

rfe = RFE(logreg, 20)             # running RFE with 20 variables as output

rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
#Model 2

X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('What is your current occupation_Housewife', 1)
# Let's re-run the model using the selected variables

#Model 3

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Last Notable Activity_Had a Phone Conversation', 1)
# Let's re-run the model using the selected variables

#Model 4

X_train_sm = sm.add_constant(X_train[col])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Lead Origin_Lead Import', 1)
# Let's re-run the model using the selected variables

#Model 5

X_train_sm = sm.add_constant(X_train[col])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('Lead Origin_Lead Add Form', 1)
# Let's re-run the model using the selected variables

#Model 6

X_train_sm = sm.add_constant(X_train[col])

logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm6.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('What is your current occupation_Unemployed', 1)
# Let's re-run the model using the selected variables

#Model 7

X_train_sm = sm.add_constant(X_train[col])

logm7 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm7.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop('What is your current occupation_Student', 1)
# Let's re-run the model using the selected variables

#Model 8

X_train_sm = sm.add_constant(X_train[col])

logm8 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm8.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})

y_train_pred_final['Prospect ID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head

y_train_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )

print(confusion)
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting lead when it is not a lead

print(FP/ float(TN+FP))
# positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_Prob.map( lambda x: 1 if x > 0.35 else 0)



y_train_pred_final.head()
y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_Prob.map( lambda x: round(x*100))



y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# sensitivity of our logistic regression model

TP / float(TP+FN)
#specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting lead converted when lead does not have converted

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
#confusion matrix

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )

confusion
confusion[1,1]/(confusion[0,1]+confusion[1,1])
confusion[1,1]/(confusion[1,0]+confusion[1,1])
precision_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)

recall_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
X_test[['Total Time Spent on Website']] = scaler.transform(X_test[['Total Time Spent on Website']])
X_test = X_test[col]

X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head

y_pred_1.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)
# Putting Prospect ID to index

y_test_df['Prospect ID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})
# Rearranging the columns

y_pred_final = y_pred_final.reindex(['Prospect ID','Converted','Converted_Prob'], axis=1)
# Let's see the head of y_pred_final

y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.35 else 0)
y_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)
#Lead Score on Test data

y_pred_final['Lead_Score'] = y_pred_final.Converted_Prob.map( lambda x: round(x*100))



y_pred_final.head()
#confusion matrix

confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting lead converted when lead does not have converted

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
confusion[1,1]/(confusion[0,1]+confusion[1,1])
confusion[1,1]/(confusion[1,0]+confusion[1,1])
precision_score(y_pred_final.Converted, y_pred_final.final_predicted)
recall_score(y_pred_final.Converted, y_pred_final.final_predicted)
y_train_pred_final
y_pred_final