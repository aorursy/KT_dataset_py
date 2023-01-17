# creating a reuseable function that will help us in ploting our barplots for analysis

def BarPlot(df, Variable, plotSize):
    fig, axs = plt.subplots(figsize = plotSize)
    plt.xticks(rotation = 45)
    ax = sns.countplot(x=Variable, data=df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}'.format(height/len(df) * 100),
                ha="center")
# creating a reuseable function that will help us in Bivariate for analysis
def CountPlot(df,Variable,title,plotsize,hue=None):
    plt.figure(figsize=plotsize)
    plt.xticks(rotation=90)
    plt.title(title)
    sns.countplot(data = df, x=Variable, order=df[Variable].value_counts().index,hue = hue)
    plt.show()
    
    convertcount=df.pivot_table(values='Lead Number',index=Variable,columns='Converted', aggfunc='count').fillna(0)
    convertcount["Conversion(%)"] =round(convertcount[1]/(convertcount[0]+convertcount[1]),2)*100
    return print(convertcount.sort_values(ascending=False,by="Conversion(%)"))

#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import re
from scipy import stats 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing RFE and LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
# Reading File
df= pd.read_csv("../input/lead-scoring-x-online-education/Leads X Education.csv")
df.head(20)
# Getting Original onversion rate for the data set
orgConversionRate = round(100*(sum(df['Converted'])/len(df['Converted'].index)), 2)
print("The conversion rate of leads is: ",orgConversionRate)
# Checking shape of dataframe
df.shape
# Checking columns name
df.columns
# Checking columns type in dataframe
df.info()
# checking attributes for continuous variables
df.describe()
# AS Value  select represent that User has not selecte any values for that, Hence it can be converted to Null
# so that it can be treated as Null
df = df.replace('Select', np.nan)
df.head(20)
# Checking if any duplicate value in Lead Number and Prospect ID
print(sum(df.duplicated(subset= 'Lead Number'))!=0)
print(sum(df.duplicated(subset= 'Prospect ID'))!=0)
# Checking Null Values
print(df.isnull().sum(axis=0))
# Checking column-wise null percentages here
print(round(100*(df.isnull().sum()/len(df)).sort_values(ascending= False), 2))
# Droping columns having null percentage >50%
df = df.drop(df.columns[df.apply(lambda col: col.isnull().sum()/len(df) > 0.70)], axis=1)
df
## Checking number of unique values per column 
df.nunique()
df = df.drop(df.columns [df.apply(lambda col: col.nunique()==1)], axis=1)
df
### We can also drop the column 'Prospect ID' as we already have an identifying column with unique values: 'Lead Number'
df = df.drop('Prospect ID', axis=1)
df.head()
plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
sns.countplot(df['Asymmetrique Activity Index'])

plt.subplot(2,2,2)
sns.boxplot(df['Asymmetrique Activity Score'])

plt.subplot(2,2,3)
sns.countplot(df['Asymmetrique Profile Index'])

plt.subplot(2,2,4)
sns.boxplot(df['Asymmetrique Profile Score'])
colsToDrop = ['Asymmetrique Activity Index','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Profile Score']
df =df.drop(colsToDrop, axis=1)
# Checking Lead Quality
df['Lead Quality'].value_counts()
# Since 'Lead Quality' is based on an employees intuition, let us inpute any NAN values with 'Not Sure'
df['Lead Quality'] = df['Lead Quality'].fillna('Not Sure')
df['Lead Quality'].value_counts()
BarPlot(df, 'Lead Quality', (10,10))
df =df.drop('Lead Quality', axis=1)
df
df['City'].value_counts()
df.City.describe()
BarPlot(df, 'City', (10,10))
df.City= df.City.fillna('Mumbai')
df['City'].value_counts(normalize= True)
### Exploring 'Specialization' column which hs 36.58% null values
df.Specialization.describe()
BarPlot(df, 'Specialization', (10,10))
df.Specialization= df['Specialization'].fillna('Others')
df.Specialization.value_counts(normalize=True)
df.Tags.describe()
BarPlot(df,'Tags', (10,10))
df = df.drop(['Tags'], axis = 1)
# Checking again the Null values
print(round(100*(df.isnull().sum(axis=0)/len(df.index)).sort_values(ascending=False),2))
df['What matters most to you in choosing a course'].value_counts()
BarPlot(df, 'What matters most to you in choosing a course',(8,5))
# Dropping 'What matters most to you in choosing a course'
df= df.drop(['What matters most to you in choosing a course'], axis =1)
df['What is your current occupation'].value_counts(normalize= True)
BarPlot(df, 'What is your current occupation',(10,10))
df['What is your current occupation']=df['What is your current occupation'].fillna('Unemployed')
BarPlot(df, 'What is your current occupation',(10,10))
df['Country'].value_counts()
df['Country'].describe()
df['Country']= df['Country'].fillna('India')
df['Country'].value_counts()
df=df.drop(['Country'], axis=1)
df.head()
# Checking Null Values Again

print(round(100*(df.isnull().sum()/len(df)).sort_values(ascending= False), 2))
### Imputing missing values in Lead Source
df['Lead Source'].value_counts()
### Imputing missing values with 'Google'

df['Lead Source'] = df['Lead Source'].fillna('Google')
df['Lead Source'].value_counts()
df['Lead Source'] =  df['Lead Source'].apply(lambda x: x.capitalize())
df['Lead Source'].value_counts()
df['Page Views Per Visit'].value_counts()
df['Page Views Per Visit'].describe()
df['Page Views Per Visit'].median()
#### Imputing the missing values with '2.0' which is the median value
df['Page Views Per Visit']= df['Page Views Per Visit'].fillna(2.0)
df['Page Views Per Visit']
df.TotalVisits.describe()
### We will impute this value with the meadian value since the 
### mean and the median values are relatively close to each other
df.TotalVisits = df.TotalVisits.fillna(3.0)
df.TotalVisits.value_counts()
df['Last Activity'].value_counts()
#### Imputing the missing values with 'Email Opened'
df['Last Activity'] = df['Last Activity'].fillna('Email Opened')
df['Last Activity'].value_counts()
df.shape
print(round(100*(df.isnull().sum()/len(df)).sort_values(ascending= False), 2))
# Getting shape of dataframe after cleanup
df.shape
df.info()
features = ['TotalVisits','Total Time Spent on Website','Page Views Per Visit']
# Plotting Box plot for continuous columns
plt.figure(figsize = (20,12))
for i in enumerate(features):
    plt.subplot(2,2,i[0]+1)
    sns.boxplot(df[i[1]])
### Caping data at the 1% & 95% mark so as to not lose any values or drop rows
q1 = df['Page Views Per Visit'].quantile(0.01)
df['Page Views Per Visit'][df['Page Views Per Visit']<= q1] = q1

q3 = df['Page Views Per Visit'].quantile(0.95)
df['Page Views Per Visit'][df['Page Views Per Visit']>= q3] = q3
### Caping data at the 1% & 95% mark so as to not lose any values or drop rows
q1 = df['TotalVisits'].quantile(0.01)
df['TotalVisits'][df['TotalVisits']<= q1] = q1

q3 = df['TotalVisits'].quantile(0.95)
df['TotalVisits'][df['TotalVisits']>= q3] = q3
# Plotting Box plot for continuous columns to check after caping outliers
plt.figure(figsize = (20,12))
for i in enumerate(features):
    plt.subplot(2,2,i[0]+1)
    sns.boxplot(df[i[1]])
df['Lead Origin'].value_counts()
BarPlot(df,'Lead Origin', (15,10))
CountPlot(df,'Lead Origin','Conversion based on Lead Origin',(15,10),hue='Converted')
df['Lead Source'].value_counts()
BarPlot(df,'Lead Source', (15,4))
plt.figure(figsize=(20,30))
sns.countplot(data = df, x= 'Lead Source', order=df['Lead Source'].value_counts().index,hue = 'Converted')                      
plt.xticks(rotation=45)
plt.show()

## Printing % of converted Lead with respect to Lead Source
convertcount=df.pivot_table(values='Lead Number',index='Lead Source',columns='Converted', aggfunc='count').fillna(0)
convertcount["Conversion(%)"] =round(convertcount[1]/(convertcount[0]+convertcount[1]),2)*100
print(convertcount.sort_values(ascending=False,by="Conversion(%)"))
cols=['Click2call', 'Live chat', 'Nc_edm', 'Pay per click ads', 'Press_release',
  'Social media', 'Welearn', 'Bing', 'Blog', 'Testone', 'Welearnblog_home', 'Youtubechannel']
df['Lead Source'] = df['Lead Source'].replace(cols, 'Others')
BarPlot(df,'Lead Source', (15,10))
plt.figure(figsize=(20,30))
sns.countplot(data = df, x= 'Lead Source', order=df['Lead Source'].value_counts().index,hue = 'Converted')                      
plt.xticks(rotation=45)
plt.show()

convertcount=df.pivot_table(values='Lead Number',index='Lead Source',columns='Converted', aggfunc='count').fillna(0)
convertcount["Conversion(%)"] =round(convertcount[1]/(convertcount[0]+convertcount[1]),2)*100
print(convertcount.sort_values(ascending=False,by="Conversion(%)"))
df['Do Not Email'].value_counts()
BarPlot(df,'Do Not Email', (15,10))
CountPlot(df,'Do Not Email','Do Not Email',(15,10),hue='Converted')
df['Do Not Call'].value_counts()
BarPlot(df,'Do Not Call', (15,10))
CountPlot(df,'Do Not Call','Conversion based on Do Not Call',(15,10),hue='Converted')
df = df.drop(['Do Not Call', 'Do Not Email'], axis=1)
df.shape
df.info()
df['Last Activity'].value_counts()
BarPlot(df,'Last Activity', (15,5))
CountPlot(df,'Last Activity','Conversion based on Last Activity',(15,10),hue='Converted')
features = ['Search', 'Newspaper Article', 'X Education Forums', 'Newspaper' , 'Digital Advertisement','Through Recommendations']
for i in enumerate(features):
    print(df[i[1]].value_counts())
features = ['Search', 'Newspaper Article', 'X Education Forums', 'Newspaper' , 'Digital Advertisement','Through Recommendations']
plt.figure(figsize = (20,20))
for i in enumerate(features):
    plt.subplot(3,2,i[0]+1)
    sns.countplot(x = df[i[1]], data = df) 
plt.show()
Cols = ['Search', 'Newspaper', 'X Education Forums', 'Newspaper Article' , 'Digital Advertisement','Through Recommendations']
df = df.drop(Cols,axis=1)
df.head()
df['A free copy of Mastering The Interview'].value_counts()
BarPlot(df,'A free copy of Mastering The Interview', (15,10))
CountPlot(df,'A free copy of Mastering The Interview','Conversion based on A free copy of Mastering The Interview',(15,10),hue='Converted')
df['Last Notable Activity'].value_counts()
BarPlot(df,'Last Notable Activity', (15,10))
## Droping Last Notable Activity 
df= df.drop(['Last Notable Activity'], axis=1)
df.head()
df.shape
BarPlot(df,'Specialization', (15,10))
CountPlot(df,'Specialization','Conversion based on Specialization',(15,10),hue='Converted')
BarPlot(df,'City', (15,10))
CountPlot(df,'City','Conversion based on City',(15,10),hue='Converted')
df['What is your current occupation'].value_counts()
BarPlot(df,'What is your current occupation', (15,10))
CountPlot(df,'What is your current occupation','Conversion based on What is your current occupation',(15,10),hue='Converted')
BarPlot(df,'TotalVisits', (15,10))
CountPlot(df,'TotalVisits','Conversion based on Total Visit',(15,10),hue='Converted')
BarPlot(df,'Page Views Per Visit', (25,20))
CountPlot(df,'Page Views Per Visit','Page Views Per Visit vs Conversion',(20,20),hue='Converted')
df['Total Time Spent on Website'].value_counts()
df['Total Time Spent on Website'] = df['Total Time Spent on Website'].apply(lambda x: round((x/60), 2))
df.head()
sns.distplot(df['Total Time Spent on Website'])
plt.show()
# Let us split our dataframe to perform better analysis
df1=df[df['Total Time Spent on Website']>=1.0]
df1["Hours Spent"]= df1["Total Time Spent on Website"].astype(int)

df1.head()
CountPlot(df1,'Hours Spent','Conversion based on Last Activity',(15,10),hue='Converted')
plt.figure(figsize=(20,20))
plt.xticks(rotation=45)
plt.yscale('log')
sns.boxplot(data =df1, x='TotalVisits',y='Total Time Spent on Website', hue ='Converted',orient='v')
plt.title('Total Time Spent Vs Total Visits based on Conversion')
plt.show()
plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), cmap='YlGnBu',annot=True)
# Final Dataframe
df.head()
### First we will convert the Yes/No values in the 'A free copy of Mastering The Interview' column to 1/0

df['A free copy of Mastering The Interview'] = df['A free copy of Mastering The Interview'].map(dict(Yes=1, No=0))
df.head()
dummy_Cols = ['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','City']
dummy = pd.get_dummies(df[dummy_Cols],drop_first=True)
dummy.head()
combined = df.copy()
combined.shape
combined = pd.concat([combined, dummy], axis=1)
combined.head()
### We will now drop the original columns and the columns that have 'Others' as a sub heading since we had 
### combined various values to create those columns

cols = ['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','City',
                     'Lead Source_Others','Specialization_Others']
combined = combined.drop(cols, axis=1)
combined.head()
combined.shape
combined.info()
### First we will drop the Converted & Lead Number columns 
X = combined.drop(['Converted','Lead Number'], axis=1)
X.head()
X.shape
### Adding the target variable 'Converted' to y
y = combined['Converted']

y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
X_train.shape
X_test.shape
X_train.head()
X_train.shape
## Scaling numeric Variables
scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])


X_train.head()
X_train.shape
combined.corr()
plt.figure(figsize=(20,20))
sns.heatmap(combined.corr(),cmap='YlGnBu',annot=True)
### Dropping highly correlated variables
X_train = X_train.drop(['Lead Origin_Lead Add Form', 'Lead Source_Facebook'], axis=1)
X_test = X_test.drop(['Lead Origin_Lead Add Form', 'Lead Source_Facebook'], axis=1)
X_train.head()
X_train.shape
plt.figure(figsize=(30,20))
sns.heatmap(combined[X_train.columns].corr(),cmap='YlGnBu',annot=True)
X_train.info()
## Creating Logistic Regression Model
logisticRegressionModel = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logisticRegressionModel.fit().summary()
logreg = LogisticRegression()

rfe = RFE(logreg, 15)
rfe= rfe.fit(X_train,y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
X_train1= X_train[col]
X_train1
X_train_sm = sm.add_constant(X_train1)
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Lead_Score_Prob':y_train_pred})
y_train_pred_final['Lead'] = y_train.index
y_train_pred_final.head()
y_train_pred_final['Final_Predicted_Hot_Lead'] = y_train_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()
from sklearn import metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted_Hot_Lead)
print(confusion)
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted_Hot_Lead))
### Checking VIF values
vif = pd.DataFrame()
vif['Features'] = X_train1.columns
vif['VIF'] = [variance_inflation_factor(X_train1.values, i) for i in range(X_train1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train2 = X_train1.drop('Last Activity_Resubscribed to emails', axis=1)
X_train2
X_train_sm = sm.add_constant(X_train2)
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()
# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final['Lead_Score_Prob'] = y_train_pred
y_train_pred_final['Final_Predicted_Hot_Lead'] = y_train_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted_Hot_Lead)
print(confusion)
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted_Hot_Lead))
vif = pd.DataFrame()
vif['Features'] = X_train2.columns
vif['VIF'] = [variance_inflation_factor(X_train2.values, i) for i in range(X_train2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train3 = X_train2.drop('What is your current occupation_Housewife', axis=1)
X_train3.columns
X_train_sm = sm.add_constant(X_train3)
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final['Lead_Score_Prob'] = y_train_pred
y_train_pred_final['Final_Predicted_Hot_Lead'] = y_train_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted_Hot_Lead)
print(confusion)
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted_Hot_Lead))
vif = pd.DataFrame()
vif['Features'] = X_train3.columns
vif['VIF'] = [variance_inflation_factor(X_train3.values, i) for i in range(X_train3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train4 = X_train3.drop('What is your current occupation_Working Professional', axis=1)
X_train4.columns 
X_train_sm = sm.add_constant(X_train4)
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final['Lead_Score_Prob'] = y_train_pred
y_train_pred_final['Final_Predicted_Hot_Lead'] = y_train_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted_Hot_Lead)
print(confusion)
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted_Hot_Lead))
vif = pd.DataFrame()
vif['Features'] = X_train4.columns
vif['VIF'] = [variance_inflation_factor(X_train4.values, i) for i in range(X_train4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
plt.figure(figsize=(20,20))
sns.heatmap(X_train_sm.corr(),cmap='YlGnBu',annot=True)
y_train_pred = res.predict(X_train_sm)
y_train_pred.head()
y_train_pred_final['Final_Predicted_Hot_Lead'] = y_train_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted_Hot_Lead)
print(confusion)
# Let's check the overall accuracy.
print(round(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted_Hot_Lead),2))
y_train_pred_final['Lead_Score'] = round((y_train_pred_final['Lead_Score_Prob'] * 100),0)

y_train_pred_final['Lead_Score'] = y_train_pred_final['Lead_Score'].astype('int')

# Let's see the head
y_train_pred_final.head()
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
round((TP / float(TP+FN)),2)
# Let us calculate specificity
round((TN / float(TN+FP)),2)
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Lead_Score_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Lead_Score_Prob)
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['Probability','Accuracy','Sensitivity','Specificty'])


num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificty'])
plt.show()
y_train_pred_final['Final_Predicted_Hot_Lead'] = y_train_pred_final.Lead_Score_Prob.map( lambda x: 1 if x > 0.33 else 0)

y_train_pred_final.head()
# Accuracy
round(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted_Hot_Lead),2)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted_Hot_Lead )
confusion2
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
round(TP / float(TP+FN),2)
# Let us calculate specificity
round(TN / float(TN+FP),2)
### Calculating Precision
precision =round(TP/float(TP+FP),2)
precision
### Calculating Recall
recall = round(TP/float(TP+FN),2)
recall
### Let us generate the Precision vs Recall tradeoff curve 
p ,r, thresholds=precision_recall_curve(y_train_pred_final.Converted,y_train_pred_final['Lead_Score_Prob'])
plt.title('Precision vs Recall tradeoff')
plt.plot(thresholds, p[:-1], "g-")    # Plotting precision
plt.plot(thresholds, r[:-1], "r-")    # Plotting Recall
plt.show()
### The F statistic is given by 2 * (precision * recall) / (precision + recall)
## The F score is used to measure a test's accuracy, and it balances the use of precision and recall to do it.
### The F score can provide a more realistic measure of a test's performance by using both precision and recall
F1 =2 * (precision * recall) / (precision + recall)
round(F1,2)
X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits',
                                'Total Time Spent on Website','Page Views Per Visit']])
X_train4.shape
X_test = X_test[X_train4.columns]

X_test.shape
X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1.head()
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
y_test_df['Lead'] = y_test_df.index
# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.shape
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Lead_Score_Prob'})
# Rearranging the columns

y_pred_final = y_pred_final.reindex(['Lead','Converted','Lead_Score_Prob'], axis=1)
# Adding Lead_Score column

y_pred_final['Lead_Score'] = round((y_pred_final['Lead_Score_Prob'] * 100),0)

y_pred_final['Lead_Score'] = y_pred_final['Lead_Score'].astype(int)
# Let's see the head of y_pred_final
y_pred_final.head()
y_pred_final['Final_Predicted_Hot_Lead'] = y_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > 0.33 else 0)
y_pred_final.head()
# Let's check the overall accuracy.
round(metrics.accuracy_score(y_pred_final.Converted, y_pred_final.Final_Predicted_Hot_Lead),2)
confusion3 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.Final_Predicted_Hot_Lead )
confusion3
TP = confusion3[1,1] # true positive 
TN = confusion3[0,0] # true negatives
FP = confusion3[0,1] # false positives
FN = confusion3[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
round((TP / float(TP+FN)),2)
# Let us calculate specificity
round(TN / float(TN+FP),2)
y_train_pred_final = y_train_pred_final.reindex(['Lead','Converted','Lead_Score_Prob','Lead_Score','Final_Predicted_Hot_Lead'], axis=1)
y_train_pred_final
### Generating table
resultingTable1 = pd.merge(y_train_pred_final,df,how='inner',left_on='Lead',right_index=True)
resultingTable1[['Lead Number','Lead_Score']].head()
### Generating table
resultingTable2 = pd.merge(y_pred_final,df,how='inner',left_on='Lead',right_index=True)
resultingTable2[['Lead Number','Lead_Score']].head()
result_df= pd.concat([resultingTable1, resultingTable2])
result_df
### renaming Converted_x to Converted and droping Converted_y as both are same
result_df=result_df.rename(columns={'Converted_x' : 'Converted'})
result_df= result_df.drop(['Converted_y'], axis=1)
result_df
# coefficients of our final model 

pd.options.display.float_format = '{:.2f}'.format
new_params = res.params[1:]
new_params
# Getting a relative coeffient value for all the features wrt the feature with the highest coefficient

feature_importance = new_params
feature_importance = 100.0 * (feature_importance / feature_importance.max())
feature_importance