import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timedelta


# import all libraries and dependencies for data visualization
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [8,8]
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1) 
sns.set(style='darkgrid')
import matplotlib.ticker as plticker
%matplotlib inline

#Model Building libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
Lead_score=pd.read_csv('../input/lead-scoring-dataset/Lead Scoring.csv')
Lead_score.head()
# Reading the data dictionary file
Lead_score_data_dict=pd.read_excel('../input/lead-scoring-dataset/Leads Data Dictionary.xlsx')
Lead_score_data_dict
# Checking shape of Dataframe
Row_col=Lead_score.shape
Row_col
# Checking size
Lead_score.size
#checking the datatypes
Lead_score.info()
# Checking the statistical details of numerical columns
Lead_score.describe()
# Checking for 'select' Values
print(Lead_score['Specialization'].str.contains('Select').value_counts())
print(Lead_score['How did you hear about X Education'].str.contains('Select').value_counts())
print(Lead_score['Lead Profile'].str.contains('Select').value_counts())
print(Lead_score['Country'].str.contains('Select').value_counts())
# Converting 'Select' values to NaN.
Lead_score = Lead_score.replace('Select', np.nan)
# Checking for 'select' Values after replacing them with np.nan
print(Lead_score['Specialization'].str.contains('Select').value_counts())
print(Lead_score['How did you hear about X Education'].str.contains('Select').value_counts())
print(Lead_score['Lead Profile'].str.contains('Select').value_counts())
print(Lead_score['Country'].str.contains('Select').value_counts())
# Checking for total count and percentage of null values in all columns of the dataframe.

total = pd.DataFrame(Lead_score.isnull().sum().sort_values(ascending=False), columns=['Total'])
percentage = pd.DataFrame(round(100*(Lead_score.isnull().sum()/Lead_score.shape[0]),2).sort_values(ascending=False)\
                          ,columns=['Percentage'])
pd.concat([total, percentage], axis = 1)
# we will drop the columns having more than 40% NA values.
Lead_score = Lead_score.drop(Lead_score.loc[:,list(round(100*(Lead_score.isnull().sum()/len(Lead_score.index)), 2)>45)].columns, axis=1)
Lead_score.shape
#We can check the number of unique values is a column
# If the number of unique values <=40: Categorical column
# If the number of unique values in a columns> 50: Continuous

Lead_score.nunique().sort_values()
# Dropping the columns
drop_cols=['Tags']
Lead_score.drop(labels=drop_cols,axis=1,inplace=True)
round((Lead_score['What matters most to you in choosing a course'].value_counts(normalize=True)*100),2)
need = Lead_score['What matters most to you in choosing a course'].value_counts().index[:2]
Lead_score['What matters most to you in choosing a course'] = np.where(Lead_score['What matters most to you in choosing a course'].isin(need),Lead_score['What matters most to you in choosing a course'], 'OTHER')
round((Lead_score['What matters most to you in choosing a course'].value_counts(normalize=True)*100),2)
round((Lead_score['Lead Origin'].value_counts(normalize=True)*100),2)
need = Lead_score['Lead Origin'].value_counts().index[:4]
Lead_score['Lead Origin'] = np.where(Lead_score['Lead Origin'].isin(need),Lead_score['Lead Origin'], 'OTHER')
round((Lead_score['Lead Origin'].value_counts(normalize=True)*100),2)
round((Lead_score['What is your current occupation'].value_counts(normalize=True)*100),2)
need = Lead_score['What is your current occupation'].value_counts().index[:5]
Lead_score['What is your current occupation'] = np.where(Lead_score['What is your current occupation'].isin(need),Lead_score['What is your current occupation'], 'OTHER')
round((Lead_score['What is your current occupation'].value_counts(normalize=True)*100),2)
round((Lead_score['City'].value_counts(normalize=True)*100),2)
need = Lead_score['City'].value_counts().index[:4]
Lead_score['City'] = np.where(Lead_score['City'].isin(need),Lead_score['City'], 'OTHER')
round((Lead_score['City'].value_counts(normalize=True)*100),2)
round((Lead_score['Last Notable Activity'].value_counts(normalize=True)*100),2)
need = Lead_score['Last Notable Activity'].value_counts().index[:6]
Lead_score['Last Notable Activity'] = np.where(Lead_score['Last Notable Activity'].isin(need),Lead_score['Last Notable Activity'], 'OTHER')
round((Lead_score['Last Notable Activity'].value_counts(normalize=True)*100),2)
round((Lead_score['Last Activity'].value_counts(normalize=True)*100),2)
need = Lead_score['Last Activity'].value_counts().index[:10]
Lead_score['Last Activity'] = np.where(Lead_score['Last Activity'].isin(need),Lead_score['Last Activity'], 'OTHER')
round((Lead_score['Last Activity'].value_counts(normalize=True)*100),2)
round((Lead_score['Specialization'].value_counts(normalize=True)*100),2)
need = Lead_score['Specialization'].value_counts().index[:10]
Lead_score['Specialization'] = np.where(Lead_score['Specialization'].isin(need),Lead_score['Specialization'], 'OTHER')
round((Lead_score['Specialization'].value_counts(normalize=True)*100),2)
round((Lead_score['Lead Source'].value_counts(normalize=True)*100),2)
need = Lead_score['Lead Source'].value_counts().index[:8]
Lead_score['Lead Source'] = np.where(Lead_score['Lead Source'].isin(need),Lead_score['Lead Source'], 'OTHER')
round((Lead_score['Lead Source'].value_counts(normalize=True)*100),2)
round((Lead_score['Country'].value_counts(normalize=True)*100),2)
need = Lead_score['Country'].value_counts().index[:10]
Lead_score['Country'] = np.where(Lead_score['Country'].isin(need),Lead_score['Country'], 'OTHERS')
round((Lead_score['Country'].value_counts(normalize=True)*100),2)
round(100*(Lead_score.isnull().sum()/len(Lead_score.index)), 2)
# Rest missing values are under 2% so we can drop these rows.
Lead_score.dropna(inplace = True)
round(100*(Lead_score.isnull().sum()/len(Lead_score.index)), 2)
row_col=Lead_score.shape
row_col
# Checking row wise null values
(Lead_score.isnull().sum(axis=1)*100/len(Lead_score)).value_counts(ascending=True)
Lead_score.size
# Checking for duplicate values in dataset
Lead_score_dupl=Lead_score.copy()
Lead_score_dupl.drop_duplicates(subset=None,inplace=True)
Lead_score_dupl.size
round(100*(row_col[0])/(Row_col[0]),2)
plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
cols = ['r','c']

Lead_score['Converted'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True, colors=cols)
plt.title('Lead Conversion',fontweight="bold", size=20)
plt.subplot(1,2,2)
sns.countplot('Converted',data=Lead_score,palette='PuRd')
plt.title('Lead Conversion',fontweight="bold", size=20)
plt.subplots_adjust(right=1)
plt.show()

plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)

sns.countplot(x='Lead Source', data =Lead_score, palette='magma')
plt.xticks(rotation = 90,fontweight="bold")
plt.title('Lead Source', fontweight='bold',size=20)
plt.subplot(1,2,2)
sns.barplot(x="Lead Source", y="Converted", data=Lead_score,palette='magma')
plt.title('Lead Source vs Converted',fontweight="bold", size=20)
plt.ylabel("Conversion  Rate")
plt.xticks(rotation = 90,fontweight="bold")
plt.subplots_adjust(right=1)
plt.show()
plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
cols = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','r']
Lead_score['Lead Origin'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True, colors=cols)
plt.title('Lead Origin',fontweight="bold", size=20)
plt.subplot(1,2,2)
sns.barplot(x="Lead Origin", y="Converted", data=Lead_score,palette='magma')
plt.title('Lead Origin vs Converted',fontweight="bold", size=20)
plt.ylabel("Conversion  Rate",fontweight="bold", size=20)
plt.xticks(rotation = 45,fontweight="bold")
plt.subplots_adjust(right=1)
plt.show()
plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
sns.countplot(x = "Do Not Email", hue = "Converted", data = Lead_score, palette= 'gist_rainbow')

plt.title('Email vs Lead Conversion', fontweight="bold", size=20)
plt.subplot(1,2,2)
sns.countplot(x = "Do Not Call", hue = "Converted", data = Lead_score, palette='cool')
plt.title('Call vs Lead Conversion',fontweight="bold", size=20)
plt.subplots_adjust(right=1)
plt.show()
Lead_score['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])
percentiles = Lead_score['TotalVisits'].quantile([0.05,0.95]).values
Lead_score['TotalVisits'][Lead_score['TotalVisits'] <= percentiles[0]] = percentiles[0]
Lead_score['TotalVisits'][Lead_score['TotalVisits'] >= percentiles[1]] = percentiles[1]
plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
sns.violinplot(y= "TotalVisits", x = "Converted", data = Lead_score, palette= 'BuPu')
plt.xticks(rotation = 90,fontweight="bold")
plt.title('Total visits to website vs Lead Conversion', fontweight="bold", size=20)
plt.subplot(1,2,2)
sns.violinplot(y = "Total Time Spent on Website",x = "Converted", data = Lead_score, palette='husl')
plt.title('Total time spent on website vs Lead Conversion',fontweight="bold", size=20)
plt.subplots_adjust(right=1)
plt.show()
plt.figure(figsize=(14, 6))
sns.countplot(x='Last Activity',hue='Converted', data= Lead_score, palette='summer')
plt.title('Last Activity vs Lead Conversion',fontweight="bold", size=20)
plt.xticks(rotation = 45,fontweight="bold")
plt.show()
Lead_score.Country.describe()
plt.figure(figsize=(10, 6))
sns.barplot(x="Specialization", y="Converted", data=Lead_score,palette='ocean')
plt.xticks(rotation = 45,fontweight="bold")
plt.title('Specialization vs Lead Conversion', fontweight='bold', size=20)

plt.show()
plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)

sns.countplot(x='What is your current occupation', data=Lead_score,palette='husl')
plt.title('Current Occupation',fontweight="bold", size=20)
plt.xticks(rotation = 45,fontweight="bold")

plt.subplot(1,2,2)
sns.barplot(x='What is your current occupation', y="Converted", data=Lead_score,palette='winter')
plt.title('Current occupation vs Lead Conversion',fontweight="bold", size=20)
plt.ylabel("Conversion  Rate",fontweight="bold", size=20)
plt.xticks(rotation = 45,fontweight="bold")
plt.subplots_adjust(right=1)
plt.show()
Lead_score['What matters most to you in choosing a course'].describe()
Lead_score['Search'].value_counts()
plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
cols = ['lightskyblue','salmon','yellowgreen', 'gold', 'purple']
Lead_score['City'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True, colors=cols)
plt.title('City of the Customer',fontweight="bold", size=20)
plt.subplot(1,2,2)
sns.barplot(x="City", y="Converted", data=Lead_score,palette='twilight')
plt.title('City vs Lead Conversion',fontweight="bold", size=20)
plt.ylabel("Conversion  Rate",fontweight="bold", size=20)
plt.xticks(rotation = 45,fontweight="bold")
plt.subplots_adjust(right=1)

plt.show()
plt.figure(figsize=(14, 6))
sns.countplot(x='Last Notable Activity', hue="Converted", data=Lead_score,palette='nipy_spectral')
plt.xticks(rotation = 45,fontweight="bold")
plt.title('Last notable activity vs Lead Conversion', fontweight='bold', size=20)

plt.show()
Lead_score = Lead_score.drop(['Lead Number','Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
           'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content',
           'Get updates on DM Content','I agree to pay the amount through cheque','A free copy of Mastering The Interview','Country','Last Notable Activity','Do Not Email', 'Do Not Call'],1)
Lead_score.shape
#Categorical columns
Lead_score.loc[:,Lead_score.dtypes == 'object'].columns
# Create dummy variables using the 'get_dummies'
dummy = pd.get_dummies(Lead_score[['Lead Origin','Specialization' ,'Lead Source','What is your current occupation','City', 'What matters most to you in choosing a course','Last Activity']], drop_first=False)
# Add the results to the master dataframe
Lead_final = pd.concat([Lead_score, dummy], axis=1)
Lead_final.head()
#Drop columns after dummy variable creation
Lead_final = Lead_final.drop(['Lead Origin','Specialization' ,'Lead Source','What is your current occupation', 
'What matters most to you in choosing a course','Last Activity', 'City'],1)
Lead_final.info()
Lead_final.columns
#Lets drop the columns having Other in their Category
Lead_final = Lead_final.drop(['Specialization_OTHER','Lead Source_OTHER','City_OTHER', 'City_Other Cities','What is your current occupation_OTHER',
       'What is your current occupation_Other','What matters most to you in choosing a course_OTHER','Last Activity_OTHER'],1)
#Number of columns after dummy variable creation
Lead_final.shape
# Putting feature variable to X
X = Lead_final.drop(['Prospect ID','Converted'], axis=1)

X.head()
# Putting response variable to y
y = Lead_final['Converted']

y.head()
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
#Rows and columns after split
print(X_train.shape)
print(X_test.shape)

#scaling continuous variables in the dataset
scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()
### Checking the Lead Conversion Rate
converted = (sum(Lead_final['Converted'])/len(Lead_final['Converted'].index))*100
converted
# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 23)             # running RFE with 23 variables as output
rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
col
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
col1=col.drop('What is your current occupation_Housewife',1)

#Lets run the model with selected variables
X_train_sm = sm.add_constant(X_train[col1])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
col2=col1.drop('What matters most to you in choosing a course_Flexibility & Convenience',1)
X_train_sm = sm.add_constant(X_train[col2]) #Rerun the model
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
col3=col2.drop('Lead Source_Facebook',1)
X_train_sm = sm.add_constant(X_train[col3])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
col4=col3.drop('Lead Source_Referral Sites',1)
X_train_sm = sm.add_constant(X_train[col4])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
col5=col4.drop('Lead Origin_API',1)
X_train_sm = sm.add_constant(X_train[col5])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
col6=col5.drop('Lead Source_Organic Search',1)
X_train_sm = sm.add_constant(X_train[col6])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col6].columns
vif['VIF'] = [variance_inflation_factor(X_train[col6].values, i) for i in range(X_train[col6].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
col7=col6.drop('What matters most to you in choosing a course_Better Career Prospects',1)
# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col7])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
vif = pd.DataFrame()
vif['Features'] = X_train[col7].columns
vif['VIF'] = [variance_inflation_factor(X_train[col7].values, i) for i in range(X_train[col7].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
col8=col7.drop('Lead Origin_Landing Page Submission',1)
# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col8])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col8].columns
vif['VIF'] = [variance_inflation_factor(X_train[col8].values, i) for i in range(X_train[col8].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_sm = sm.add_constant(X_train[col8])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
plt.figure(figsize=(15,8), dpi=80, facecolor='w', edgecolor='k', frameon='True')

cor = X_train[col8].corr()
ax=sns.heatmap(cor, annot=True, cmap="YlGnBu")

plt.tight_layout()
plt.show()
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
print('Sensitivity:')
TP / float(TP+FN)
# Let us calculate specificity
print('Specificity:')
TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)
def auc_val(fpr,tpr):
    AreaUnderCurve = 0.
    for i in range(len(fpr)-1):
        AreaUnderCurve += (fpr[i+1]-fpr[i]) * (tpr[i+1]+tpr[i])
    AreaUnderCurve *= 0.5
    return AreaUnderCurve
auc = auc_val(fpr,tpr)
auc
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Slightly alter the figure size to make it more horizontal.

#plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k', frameon='True')
sns.set_style("whitegrid") # white/whitegrid/dark/ticks
sns.set_context("paper") # talk/poster
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'], figsize=(10,6))
# plot x axis limits
plt.xticks(np.arange(0, 1, step=0.05), size = 12)
plt.yticks(size = 12)
plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.34 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
print('Sensitivity:')
TP / float(TP+FN)
# Let us calculate specificity
print('Specificity:')
TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))
# Positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
confusion
precision=confusion[1,1]/(confusion[0,1]+confusion[1,1])
precision
recall=confusion[1,1]/(confusion[1,0]+confusion[1,1])
recall
precision_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
y_train_pred_final.Converted, y_train_pred_final.predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)
# Slightly alter the figure size to make it more horizontal.
plt.figure(figsize=(8, 4), dpi=100, facecolor='w', edgecolor='k', frameon='True')
plt.plot(thresholds, p[:-1], "b-")
plt.plot(thresholds, r[:-1], "r-")
plt.xticks(np.arange(0, 1, step=0.05))
plt.show()
F1 = 2*(precision*recall)/(precision+recall)
F1

X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])


X_test = X_test[col8]
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
# Putting CustID to index
y_test_df['Prospect ID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()

# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Conversion_Prob'})
# Rearranging the columns
y_pred_final = y_pred_final.reindex(['Converted','Prospect ID','Conversion_Prob'], axis=1)
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.34 else 0)
y_pred_final
# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion2
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
print('Sensitivity:')
TP / float(TP+FN)
# Let us calculate specificity
print('Specificity:')
TN / float(TN+FP)
print(FP/ float(TN+FP))
print (TP / float(TP+FP))
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

    return fpr,tpr, thresholds
fpr, tpr, thresholds = metrics.roc_curve( y_pred_final.Converted, y_pred_final.Conversion_Prob, drop_intermediate = False )
draw_roc(y_pred_final.Converted, y_pred_final.Conversion_Prob)
def auc_val(fpr,tpr):
    AreaUnderCurve = 0.
    for i in range(len(fpr)-1):
        AreaUnderCurve += (fpr[i+1]-fpr[i]) * (tpr[i+1]+tpr[i])
    AreaUnderCurve *= 0.5
    return AreaUnderCurve
# Selecting the test dataset along with the Conversion Probability and final predicted value for 'Converted'
leads_test_pred = y_pred_final.copy()
leads_test_pred.head()
#  columns from train dataset
leads_train_pred = y_train_pred_final[['Prospect ID','Converted','Conversion_Prob','final_predicted']]
leads_train_pred.head()
# Concatenating the 2 dataframes train and test along the rows with the append() function
lead_full_pred = leads_train_pred.append(leads_test_pred)
lead_full_pred.head()
# Inspecting the shape of the final dataframe and the test and train dataframes
print(leads_train_pred.shape)
print(leads_test_pred.shape)
print(lead_full_pred.shape)

# Ensuring the LeadIDs are unique for each lead in the finl dataframe
len(lead_full_pred['Prospect ID'].unique().tolist())
# Calculating the Lead Score value
# Lead Score = 100 * Conversion_Prob
lead_full_pred['Lead_Score'] = lead_full_pred['Conversion_Prob'].apply(lambda x : round(x*100))
lead_full_pred.head()
# Inspecing the max LeadID
lead_full_pred['Prospect ID'].max()
# Making the ProspectID column as index
# We willlater join it with the original_leads dataframe based on index
lead_full_pred = lead_full_pred.set_index('Prospect ID').sort_index(axis = 0, ascending = True)
lead_full_pred.head()
# Slicing the Lead Number column from original_leads dataframe
original_leads=pd.read_csv('../input/lead-scoring-dataset/Lead Scoring.csv')
original_leads= original_leads[['Lead Number']]
original_leads.head()
# Concatenating the 2 dataframes based on index and displaying the top 10 rows
# This is done son that Lead Score is associated to the Lead Number of each Lead. This will help in quick identification of the lead.
leads_with_score = pd.concat([original_leads, lead_full_pred], axis=1)
leads_with_score.head(10)
pd.options.display.float_format = '{:.2f}'.format
new_params = res.params[1:]
new_params
#feature_importance = abs(new_params)
feature_importance = new_params
feature_importance = 100.0 * (feature_importance / feature_importance.max())
feature_importance
sorted_idx = np.argsort(feature_importance,kind='quicksort',order='list of str')
sorted_idx
pos = np.arange(sorted_idx.shape[0]) + .5

featfig = plt.figure(figsize=(10,6))
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center', color = 'tab:blue',alpha=0.8)
featax.set_yticks(pos)
featax.set_yticklabels(np.array(X_train[col8].columns)[sorted_idx], fontsize=12)
featax.set_xlabel('Relative Feature Importance', fontsize=14)

plt.tight_layout()   
plt.show()
pd.DataFrame(feature_importance).reset_index().sort_values(by=0,ascending=False).head(3)
