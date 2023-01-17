import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import seaborn as sns
%matplotlib inline


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import precision_score, recall_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

import statsmodels.api as sm
df = pd.read_csv('../input/Leads.csv')
df.head() 
df.shape
df.columns
df.info()
df.describe()
### We can see that there are some outliers present in a few columns like: Total Visits, Total time spent on website
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
conversion = (sum(df['Converted'])/len(df['Converted'].index))*100
conversion
df.describe(include='all')
print(df.Country.unique())
print("-"*100)
print(df['City'].unique())
print("-"*100)
sns.countplot(df.Country)
xticks(rotation = 90)
sns.countplot(df.Country)
xticks(rotation = 90)
## More than 90% of the value is India so we can safely remove this column

df = df.drop('Country',axis=1)
sns.countplot(df.City)
xticks(rotation = 90)
## Select values and null values for City can be imputed as Mumbai

df['City'] = df['City'].fillna(df['City'].mode()[0])
df['City'] = df['City'].replace("Select", "Mumbai")
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
sns.countplot(x = "City", hue = "Converted", data = df)
xticks(rotation = 90)

# Majority Conversion is also from Mumbai
sns.countplot(df['Specialization'])
xticks(rotation = 90)
## Specialization has almost 1750 select values and apart from this 15% null values
## What we can do about this create a category of others for Students or misc.

df['Specialization'] = df['Specialization'].replace(np.nan, 'Others')
df['Specialization'] = df['Specialization'].replace("Select", "Others")
sns.countplot(x = "Specialization", hue = "Converted", data = df)
xticks(rotation = 90)

# Majority Conversion Data is from Others
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
sns.countplot(df['How did you hear about X Education'])
xticks(rotation = 90)
## Column "How did you hear about X Education" column can be dropped as most of the values are select and 23% are null values apart from select

df = df.drop('How did you hear about X Education',axis=1)
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
sns.countplot(df['What is your current occupation'])
xticks(rotation = 90)
## Most of the values are unemployed so null values can be imputed as unemployed

df['What is your current occupation'] = df['What is your current occupation'].replace(np.nan, 'Unemployed')
sns.countplot(x = "What is your current occupation", hue = "Converted", data = df)
xticks(rotation = 90)

# Max Conversion is from Unemployed, but the ratio of being converted is better in Working Professionals
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
sns.countplot(df['What matters most to you in choosing a course'])
xticks(rotation = 90)
# Since majority data (more than 80% is for Better Career Prospect, if we impute this then it will be more than 90%) so it is safe to drop this column

df = df.drop('What matters most to you in choosing a course', axis=1)
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
sns.countplot(df['Tags'])
xticks(rotation = 90)
# Since most frequent option is Will Revert after reading the email, we can impute the mode in this case:

df['Tags'] = df['Tags'].fillna(df['Tags'].mode()[0])
plt.figure(figsize=(10,7))
sns.countplot(x = "Tags", hue = "Converted", data = df)
xticks(rotation = 90)
# We can see that conversion rate of the above imputed option is highest
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
sns.countplot(df['Lead Quality'])
xticks(rotation = 90)
# Lead Quality seems an important parameter as per the business so instead of dropping this we can impute the values to  not sure since whoever was filling the form did not mention explicitly

df['Lead Quality'] = df['Lead Quality'].replace(np.nan, 'Not Sure')
sns.countplot(x = "Lead Quality", hue = "Converted", data = df)
xticks(rotation = 90)

# Low, High in relevance have a better converion ratio but Might be have the most conversions 
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
sns.countplot(df['Lead Profile'])
xticks(rotation = 90)
# Lead Profile already has 4000 select values and then 29% null values which makes this column useless, so its safe to drop it

df = df.drop('Lead Profile',axis=1)
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
sns.countplot(df['Asymmetrique Activity Index'])
xticks(rotation = 90)
# For this column we can impute the null values to 02.Medium

df['Asymmetrique Activity Index'] = df['Asymmetrique Activity Index'].fillna(df['Asymmetrique Activity Index'].mode()[0])
sns.countplot(x = "Asymmetrique Activity Index", hue = "Converted", data = df)
xticks(rotation = 90)
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
sns.countplot(df['Asymmetrique Profile Index'])
xticks(rotation = 90)
sns.countplot(x = "Asymmetrique Profile Index", hue = "Converted", data = df)
xticks(rotation = 90)
# We can not deduce any analysis from this column and since the null values are high we can safely drop this column

df = df.drop('Asymmetrique Profile Index',axis=1)
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
sns.countplot(df['Asymmetrique Activity Score'])
xticks(rotation = 90)
# Values are too close to be imputed in this case and the number of null values is not small, we can drop this column

df = df.drop('Asymmetrique Activity Score',axis=1)
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
sns.countplot(df['Asymmetrique Profile Score'])
xticks(rotation = 90)
# Values are too close to be imputed in this case and the number of null values is not small, we can drop this column

df = df.drop('Asymmetrique Profile Score',axis=1)
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
## For Lead Source, Last Activity, Page Views per Visit, Total visits we can drop 
##those rows which contain null values since the number of missing values is less

df = df.dropna()
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
df.shape
sns.countplot(x = "Lead Source", hue = "Converted", data = df)
xticks(rotation = 90)
## We can club the values which do not have a considerable impact on Lead Source

df['Lead Source'] = df['Lead Source'].replace(['Pay per Click Ads','bing','blog','Social Media','WeLearn','Click2call','Live Chat','welearnblog_Home',
                                               'youtubechannel','testone','Press_Release','NC_EDM'], 'Others')

df.loc[(df['Lead Source'] == 'google'),'Lead Source'] = 'Google'
sns.countplot(x = "Lead Source", hue = "Converted", data = df)
xticks(rotation = 90)
sns.countplot(x = "Do Not Email", hue = "Converted", data = df)
xticks(rotation = 90)

# We can see that There is a very small amount of conversion that has happened when this value is Yes
sns.countplot(x = "Do Not Call", hue = "Converted", data = df)
xticks(rotation = 90)

# We can see that There is no value of conversion for Yes value of Do not call
sns.countplot(x = "Do Not Call", data = df)
xticks(rotation = 90)
# Also we can see that there is no value of Yes in this case so we can drop this column since it does not add any value to the data.

df = df.drop('Do Not Call',axis=1)
# Checking the percentage of missing values
round(100*(df.isnull().sum()/len(df.index)), 2)
sns.boxplot(x = "TotalVisits", hue = "Converted", data = df)
xticks(rotation = 90)

df['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])
# We can see that there are some outliers present so we can treat these outliers before proceeding further

percentiles = df['TotalVisits'].quantile([0.05,0.95]).values
df['TotalVisits'][df['TotalVisits'] <= percentiles[0]] = percentiles[0]
df['TotalVisits'][df['TotalVisits'] >= percentiles[1]] = percentiles[1]
sns.countplot(x = "TotalVisits", hue = "Converted", data = df)
xticks(rotation = 90)

# This data could mean that when a user visits the website often the ratio of conversion gets better
sns.boxplot(x = "Total Time Spent on Website", data = df)
xticks(rotation = 90)

sns.boxplot(x = "Page Views Per Visit",data = df)
xticks(rotation = 90)
## A number of outliers are also present in this case so we can remove these

percentiles = df['Page Views Per Visit'].quantile([0.05,0.95]).values
df['Page Views Per Visit'][df['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]
df['Page Views Per Visit'][df['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]
sns.boxplot(x = "Page Views Per Visit",data = df)
xticks(rotation = 90)
plt.figure(figsize=(10,7))
sns.countplot(x = "Last Activity", hue = "Converted", data = df)
xticks(rotation = 90)
# We can club calues which have no or very less data

df['Last Activity'] = df['Last Activity'].replace(['View in browser link Clicked','Visited Booth in Tradeshow',
                                                   'Approached upfront','Resubscribed to emails','Email Received','Email Marked Spam'], 'Others')
plt.figure(figsize=(10,7))
sns.countplot(x = "Last Activity", hue = "Converted", data = df)
xticks(rotation = 90)
sns.countplot(x = "Specialization", hue = "Converted", data = df)
xticks(rotation = 90)
# We can club calues which have no or very less data

df['Specialization'] = df['Specialization'].replace(['Services Excellence','Retail Management',
                                                   'Hospitality Management','Rural and Agrbusiness','E-Business'], 'Others')
sns.countplot(x = "Specialization", hue = "Converted", data = df)
xticks(rotation = 90)
sns.countplot(x = "What is your current occupation", hue = "Converted", data = df)
xticks(rotation = 90)

# From here we can see that most of the people are unemployed and have a good conversion rate, but in case of 
## working professionsals, we can see that the number of conversions is more than not converted
sns.countplot(x = "Search", hue = "Converted", data = df)
xticks(rotation = 90)
# Since there are no values for Yes we can delete this column as well. As it will not add up to the model

df = df.drop('Search',axis=1)
sns.countplot(x = "Magazine", hue = "Converted", data = df)
xticks(rotation = 90)
# We can drop this column as well as there are no values in yes counterpart so it wont add value to the data

df = df.drop('Magazine',axis=1)
sns.countplot(x = "Newspaper Article", hue = "Converted", data = df)
xticks(rotation = 90)
# We can drop this column as well as there are no values in yes counterpart so it wont add value to the data

df = df.drop('Newspaper Article',axis=1)
sns.countplot(x = "X Education Forums", hue = "Converted", data = df)
xticks(rotation = 90)
# We can drop this column as well as there are no values in yes counterpart so it wont add value to the data

df = df.drop('X Education Forums',axis=1)
sns.countplot(x = "Newspaper", hue = "Converted", data = df)
xticks(rotation = 90)
# We can drop this column as well as there are no values in yes counterpart so it wont add value to the data

df = df.drop('Newspaper',axis=1)
sns.countplot(x = "Digital Advertisement", hue = "Converted", data = df)
xticks(rotation = 90)
# We can drop this column as well as there are no values in yes counterpart so it wont add value to the data

df = df.drop('Digital Advertisement',axis=1)
sns.countplot(x = "Through Recommendations", hue = "Converted", data = df)
xticks(rotation = 90)
# We can drop this column as well as there are no values in yes counterpart so it wont add value to the data

df = df.drop('Through Recommendations',axis=1)
sns.countplot(x = "Receive More Updates About Our Courses", hue = "Converted", data = df)
xticks(rotation = 90)
# We can drop this column as well as there are no values in yes counterpart so it wont add value to the data

df = df.drop('Receive More Updates About Our Courses',axis=1)
plt.figure(figsize=(10,7))
sns.countplot(x = "Tags", hue = "Converted", data = df)
xticks(rotation = 90)
# We can club calues which have no or very less data

df['Tags'] = df['Tags'].replace(['Still Thinking','Lost to Others',
                                                   'Shall take in the next coming month','Lateral student','Interested in Next batch','in touch with EINS','In confusion whether part time or DLP', 'Recognition issue (DEC approval)','Want to take admission but has financial problems','University not recognized','opp hangup','number not provided'], 'Others')
plt.figure(figsize=(10,5))
sns.countplot(x = "Tags", hue = "Converted", data = df)
xticks(rotation = 90)
sns.countplot(x = "Lead Quality", hue = "Converted", data = df)
xticks(rotation = 90)
sns.countplot(x = "Update me on Supply Chain Content", hue = "Converted", data = df)
xticks(rotation = 90)
# We can drop this variable since there is no data for no

df = df.drop('Update me on Supply Chain Content',axis=1)
sns.countplot(x = "Get updates on DM Content", hue = "Converted", data = df)
xticks(rotation = 90)
# We can drop this variable since there is no data for Get updates on DM Content

df = df.drop('Get updates on DM Content',axis=1)
sns.countplot(x = "City", hue = "Converted", data = df)
xticks(rotation = 90)

# Most of the conversions happened from Mumbai
sns.countplot(x = "Asymmetrique Activity Index", hue = "Converted", data = df)
xticks(rotation = 90)

# Medium activity has most conversions
sns.countplot(x = "I agree to pay the amount through cheque", hue = "Converted", data = df)
xticks(rotation = 90)
# We can drop this variable since there is no data for I agree to pay through cheque

df = df.drop('I agree to pay the amount through cheque',axis=1)
sns.countplot(x = "A free copy of Mastering The Interview", hue = "Converted", data = df)
xticks(rotation = 90)

# Highest conversion with value no
df.shape
df.columns
df.head()
# List of variables to map which are in the form of Yes/No

varlist =  ['Do Not Email', 'A free copy of Mastering The Interview']

# Defining the map function
def mapping(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
df[varlist] = df[varlist].apply(mapping)
# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(df[['Lead Origin', 'Lead Source',
        'Last Activity', 'Specialization', 'What is your current occupation', 'Tags',
       'Lead Quality', 'City', 'Asymmetrique Activity Index',
       'A free copy of Mastering The Interview', 'Last Notable Activity']], drop_first=True)
dummy1.head()
# Adding the results to the original dataset
df = pd.concat([df, dummy1], axis=1)
df.head()
df = df.drop(['Lead Origin', 'Lead Source',
        'Last Activity', 'Specialization', 'What is your current occupation', 'Tags',
       'Lead Quality', 'City', 'Asymmetrique Activity Index',
       'A free copy of Mastering The Interview', 'Last Notable Activity'],axis=1)
df.head()
df = df.drop('Lead Number',axis=1)
# Dropping Lead number since both prospect id and Lead Number are unique we can keep just one column
df.head()
X = df.drop(['Prospect ID','Converted'], axis=1)
X.head()
y = df['Converted']
y.head()
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()
# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(X_train.corr(),annot = True)
plt.show()
# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()
logreg = LogisticRegression()
rfe = RFE(logreg, 15)             # running RFE with 13 variables as output
rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
col = col.drop('Tags_invalid number', 1)
# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()
# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()
# Creating a column "predicted" will be 1 if prob is > than 0.5
y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)
# Overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
TP = confusion[1,1] 
TN = confusion[0,0] 
FP = confusion[0,1] 
FN = confusion[1,0] 
# Sensitivity
TP / float(TP+FN)
# Specificity
TN / float(TN+FP)
# False postive rate
print(FP/ float(TN+FP))
# Positive Predictive Value 
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
# Different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Plot for 'accuracy','sensitivity','specificity'against Probability
cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])
plt.show()
### We can clearly see that 0.2 comes out to be an optimal cut off point in this case

y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.2 else 0)

y_train_pred_final.head()
y_train_pred_final['Lead Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final.head()
# Overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2

TP = confusion2[1,1]
TN = confusion2[0,0] 
FP = confusion2[0,1] 
FN = confusion2[1,0] 
# Sensitivity
TP / float(TP+FN)
# Specificity
TN / float(TN+FP)
# False Postive Rate
print(FP/ float(TN+FP))
# Positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
# Confusion matrix

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
confusion
# Precision
TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])
# Recall
TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])
precision_score(y_train_pred_final.Converted , y_train_pred_final.predicted)
recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
y_train_pred_final.Converted, y_train_pred_final.predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()
X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()
X_test = X_test[col]
X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head
y_pred_1.head()
y_test_df = pd.DataFrame(y_test)
# Prospect ID to index
y_test_df['Prospect ID'] = y_test_df.index
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})
y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.2 else 0)
y_pred_final.head()
# Overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)
confusion_matrix1 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion_matrix1
TP = confusion2[1,1] 
TN = confusion2[0,0] 
FP = confusion2[0,1] 
FN = confusion2[1,0]
# Sensitivity
TP / float(TP+FN)
# Specificity
TN / float(TN+FP)
