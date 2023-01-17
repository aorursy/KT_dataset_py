# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import xticks

import seaborn as sns





# Data display coustomization

pd.set_option('display.max_colwidth', None)

pd.set_option('display.max_columns', 100)
leads_data_dict = pd.read_excel('/kaggle/input/leads-dataset/Leads Data Dictionary.xlsx', skiprows=2)

leads_data_dict.drop(leads_data_dict.columns[0], axis=1,inplace=True)

leads_data_dict
leadscore = pd.read_csv('/kaggle/input/leads-dataset/Leads.csv')
leadscore.head()
leadscore = leadscore.replace('Select', np.nan)
### This function will generate a table of features, total NULL values, and %age of NULL values in it.

def findNullValuesPercentage(dataframe):

    totalNullValues = dataframe.isnull().sum().sort_values(ascending=False)

    percentageOfNullValues = round((dataframe.isnull().mean()).sort_values(ascending=False),2)

    featuresWithPrcntgOfNullValues = pd.concat([totalNullValues, percentageOfNullValues], axis=1, keys=['Total Null Values', 'Percentage of Null Values'])

    return featuresWithPrcntgOfNullValues
### this function will create BarPlot for our visualization.



def createCountPlot(keyVariable, plotSize):

    fig, axs = plt.subplots(figsize = plotSize)

    plt.xticks(rotation = 90)

    dataframe = leadscore.copy()

    dataframe[keyVariable] = dataframe[keyVariable].fillna('Missing Values')

    ax = sns.countplot(x=keyVariable, data=dataframe)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}'.format(height/len(dataframe) * 100),

                ha="center") 


#This function will just drop the list of features(inplace) provided to it and print them.

def dropTheseFeatures(features):

    print('Dataset shape before dropping the features {}'.format(leadscore.shape))

    print('*****------------------------------------------*****')

    for col in features:

        print('Removing the column {}'.format(col))

        leadscore.drop(col, axis=1, inplace=True)

    print('*****------------------------------------------*****')

    print('Dataset shape after dropping the features {}'.format(leadscore.shape))
#This function will genrate a table which is populated with feature name and the %age of count of unique values in it.

def genarateUniqueValuePercentagePlot(features):

    cols=4

    rows = len(features)//cols +1

    fig = plt.figure(figsize=(16, rows*5))

    for plot, feature in enumerate(features):

        fig.add_subplot(rows,cols,plot+1)

        ax = sns.countplot(x=leadscore[feature], data=leadscore) 

        for p in ax.patches:

            height = p.get_height()

            ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/len(leadscore) * 100),

                ha="center") 
### Function to generate heatmaps

def generateHeatmaps(df, figsize):

    plt.figure(figsize = figsize)        # Size of the figure

    sns.heatmap(df.corr(),annot = True, annot_kws={"fontsize":7})

#As the name suggests this function will plot AUC-ROC curve.

def plot_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs)

                                            #, drop_intermediate = False )

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
def getRegressionMetrics(actual,predicted):

    from sklearn.metrics import precision_score, recall_score

    m={}

    confusion = metrics.confusion_matrix(actual, predicted )

    TP = confusion[1,1] # true positive 

    TN = confusion[0,0] # true negatives

    FP = confusion[0,1] # false positives

    FN = confusion[1,0] # false negatives

    m['sensitivity']=TP / float(TP+FN)

    m['specificity']=TN / float(TN+FP)

    m['recall']=recall_score(actual, predicted)

    m['precision']=precision_score(actual, predicted)

    m['accuracy']=metrics.accuracy_score(actual, predicted)

    m['F1-score']=metrics.f1_score(actual, predicted, average='weighted')

    

    print(confusion)

    for metric in m:

        print(metric + ': ' + str(round(m[metric],2)))
leadscore.shape
leadscore.info()
#Check if any duplicated value is present in the ID and Lead Number columns

print(sum(leadscore.duplicated('Prospect ID')) == 0)

print(sum(leadscore.duplicated('Lead Number')) == 0)
dropTheseFeatures(['Prospect ID'])
numUniquesInFeatures = leadscore.nunique().sort_values()

numUniquesInFeatures
dropFeaturesWithSingleVal=[]

for feature in numUniquesInFeatures.index:

#     print(feature, numUniquesInFeatures[feature])

    if numUniquesInFeatures[feature] == 1:

        dropFeaturesWithSingleVal.append(feature)

dropFeaturesWithSingleVal
dropTheseFeatures(dropFeaturesWithSingleVal)
genarateUniqueValuePercentagePlot(['A free copy of Mastering The Interview', 'Newspaper Article', 'Search','Through Recommendations',

             'X Education Forums', 'Converted', 'Do Not Call', 'Do Not Email', 'Newspaper', 'Digital Advertisement'])
dropHighySkewedFeatures = ['Newspaper Article', 'Search','Through Recommendations',

             'X Education Forums', 'Do Not Call', 'Do Not Email', 'Newspaper', 'Digital Advertisement']

dropTheseFeatures(dropHighySkewedFeatures)
findNullValuesPercentage(leadscore)
dropHighMissingValuesFeatues = ['How did you hear about X Education', 'Lead Profile', 'Lead Quality', 

                                'Asymmetrique Profile Score','Asymmetrique Activity Score','Asymmetrique Profile Index',

                                'Asymmetrique Activity Index']

dropTheseFeatures(dropHighMissingValuesFeatues)
findNullValuesPercentage(leadscore)
dropTheseFeatures(['Tags'])
findNullValuesPercentage(leadscore)
createCountPlot('City', (10,5))
leadscore['City'].fillna('Mumbai', inplace=True)



createCountPlot('City', (10,5))
createCountPlot('Specialization', (15,5))
leadscore['Specialization'].value_counts(normalize=True, dropna=False).head()


# impute the Missing Values with Finance, HR and  Marketing Management, each of them equally.

leadscore['Specialization'].iloc[:1000].fillna('Human Resource Management', inplace=True)

leadscore['Specialization'].iloc[1001:2000].fillna('Marketing Management', inplace=True)

leadscore['Specialization'].iloc[2000:].fillna('Finance Management', inplace=True)
leadscore['Specialization'].unique()
leadscore.Specialization.replace(to_replace=['Supply Chain Management',

       'IT Projects Management', 

       'Marketing Management',

       'Retail Management',

       'Hospitality Management',

       'Healthcare Management'], value='Other Management', inplace=True)
leadscore.Specialization.replace(to_replace=[

       'Media and Advertising',

       'Travel and Tourism', 

       'Banking, Investment And Insurance', 'International Business',

       'E-COMMERCE',

       'Services Excellence',

       'Rural and Agribusiness',

       'E-Business'], value='Others', inplace=True)
leadscore['Specialization'].value_counts(normalize=True, dropna=False)
createCountPlot('Specialization', (10,5))
createCountPlot('What matters most to you in choosing a course', (15,4))
createCountPlot('What is your current occupation', (10,7.5))
leadscore['What is your current occupation'].fillna('Not Specified', inplace=True)



leadscore['What is your current occupation'] = leadscore['What is your current occupation'].replace(['Student', 'Housewife','Businessman'], 'Other')



leadscore['What is your current occupation'].value_counts()
createCountPlot('What is your current occupation', (10,7.5))
fig, axs = plt.subplots(figsize = (20,4))

plt.xticks(rotation = 90)

sns.countplot('Country', data=leadscore)
dropTheseFeatures(['What matters most to you in choosing a course','Country'])
findNullValuesPercentage(leadscore)
leadscore.dropna(inplace=True)
findNullValuesPercentage(leadscore)
leadscore.shape
leads_data_dict[(leads_data_dict['Variables']=='Last Activity') | (leads_data_dict['Variables']=='Last Notable Activity')]
fig, axs = plt.subplots(figsize = (12,4))

plt.xticks(rotation = 90)

sns.countplot('Last Notable Activity', data=leadscore)
dropTheseFeatures(['Last Notable Activity'])
round(leadscore['Last Activity'].value_counts(normalize=True, ascending=False), 2)
leadscore['Last Activity'] = leadscore['Last Activity'].replace([           

                                                                'Form Submitted on Website',       

                                                                'Unreachable',                     

                                                                'Unsubscribed',                    

                                                                'Had a Phone Conversation',        

                                                                'View in browser link Clicked',    

                                                                'Approached upfront',              

                                                                'Email Received',                  

                                                                'Email Marked Spam',               

                                                                'Resubscribed to emails',          

                                                                'Visited Booth in Tradeshow'], 'Miscellaneous')


createCountPlot('Last Activity', (7, 5))

leadscore['A free copy of Mastering The Interview'].value_counts()
leadscore['A free copy of Mastering The Interview'].replace({'Yes':1, 'No':0}, inplace=True)

leadscore['A free copy of Mastering The Interview'].value_counts()
createCountPlot('City', (7, 5))
leadscore['City'] = leadscore['City'].replace(['Thane & Outskirts', 'Other Metro Cities', 'Other Cities',

       'Other Cities of Maharashtra', 'Tier II Cities'], 'Not Mumbai Cities')
leadscore['City'].value_counts()
createCountPlot('City', (7, 5))
leadscore['What is your current occupation'].value_counts()
createCountPlot('What is your current occupation', (10, 5))
sns.countplot(leadscore['What is your current occupation'], hue=leadscore.Converted)



plt.show()
fig, axs = plt.subplots(figsize = (15, 7.5))

sns.countplot(leadscore['Specialization'], hue=leadscore.Converted)

axs.set_xticklabels(axs.get_xticklabels(),rotation=90)

plt.show()
findNullValuesPercentage(leadscore)
sns.boxplot('Page Views Per Visit', data=leadscore)
q1 = leadscore['Page Views Per Visit'].quantile(0.05) #---- lower range taken

q4 = leadscore['Page Views Per Visit'].quantile(0.95) #----- higher range taken



leadscore['Page Views Per Visit'][leadscore['Page Views Per Visit']<=q1] = q1 #----- capping of lower range 

leadscore['Page Views Per Visit'][leadscore['Page Views Per Visit']>=q4] = q4 #----- capping of higher range
sns.boxplot('Page Views Per Visit', data=leadscore)
sns.boxplot(x=leadscore.Converted,y=leadscore['Page Views Per Visit'])

plt.show()
leadscore['Total Time Spent on Website'].describe()
sns.distplot(leadscore['Total Time Spent on Website'])
leadscore['Total Time Spent on Website'] = leadscore['Total Time Spent on Website'].apply(lambda x: round((x/60), 2))

sns.distplot(leadscore['Total Time Spent on Website'], )
sns.boxplot(x=leadscore.Converted, y=leadscore['Total Time Spent on Website'])

plt.show()
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = leadscore)

plt.show()
Q3 = leadscore.TotalVisits.quantile(0.95)

leadscore = leadscore[(leadscore.TotalVisits <= Q3)]

Q1 = leadscore.TotalVisits.quantile(0.05)

leadscore = leadscore[(leadscore.TotalVisits >= Q1)]

sns.boxplot(y=leadscore['TotalVisits'])

plt.show()
sns.boxplot(y='TotalVisits', x='Converted', data=leadscore)
createCountPlot('Lead Source', (10,5))
leadscore['Lead Source'].unique()
leadscore['Lead Source'] = leadscore['Lead Source'].replace(['blog', 'Pay per Click Ads', 

                                                'bing', 'Social Media','WeLearn', 'Click2call', 'Live Chat', 

                                                'welearnblog_Home', 'youtubechannel', 'testone', 'Press_Release', 'NC_EDM'], 'Other')



leadscore['Lead Source'] = leadscore['Lead Source'].replace('google', 'Google')
fig, axs = plt.subplots(figsize = (10, 5))

sns.countplot('Lead Source', hue='Converted', data=leadscore)
leadscore['Lead Origin'].describe()
fig, axs = plt.subplots(figsize = (10, 5))

sns.countplot('Lead Origin', hue='Converted', data=leadscore)
#Drop all the rows with `Lead Import` as Lead Origin

leadscore.drop(leadscore[leadscore['Lead Origin'] == 'Lead Import'].index, inplace=True)
fig, axs = plt.subplots(figsize = (10, 5))

sns.countplot('Lead Origin', hue='Converted', data=leadscore)
fig, axs = plt.subplots(figsize = (15, 5))

sns.countplot('Last Activity', hue='Converted', data=leadscore)




leadscore.columns
leadscore_corr = leadscore[['Lead Origin', 'Lead Source', 'Converted', 'TotalVisits',

       'Total Time Spent on Website', 'Page Views Per Visit', 'Last Activity',

       'Specialization', 'What is your current occupation', 'City',

       'A free copy of Mastering The Interview']]
generateHeatmaps(leadscore_corr, (12,8))
leadscore.shape
leadscore.head()
leadscore.info()
categorical_feature =  leadscore.select_dtypes(include=['object']).columns

categorical_feature
dummy = pd.get_dummies(leadscore['Specialization'], prefix  = 'Specialization')

dummy = dummy.drop(['Specialization_Business Administration'], 1)

leads_dummified = pd.concat([leadscore, dummy], axis = 1)
dummy = pd.get_dummies(leadscore['Lead Source'], prefix  = 'Lead_Source')

dummy = dummy.drop(['Lead_Source_Facebook'], 1)

leads_dummified = pd.concat([leads_dummified, dummy], axis = 1)
dummy = pd.get_dummies(leadscore['Last Activity'], prefix  = 'Last_Activity')

dummy = dummy.drop(['Last_Activity_Email Link Clicked'], 1)

leads_dummified = pd.concat([leads_dummified, dummy], axis = 1)
leadscore['Lead Origin'].value_counts()
dummy = pd.get_dummies(leadscore['Lead Origin'], prefix = 'Lead_Origin', drop_first=True)

# dummy = dummy.drop(['Lead_Origin_Lead Add Form'], 1)

leads_dummified = pd.concat([leads_dummified, dummy], axis = 1)
dummy = pd.get_dummies(leadscore['What is your current occupation'], prefix = 'Occupation')

dummy = dummy.drop(['Occupation_Other'], 1)

leads_dummified = pd.concat([leads_dummified, dummy], axis = 1)
dummy = pd.get_dummies(leadscore['City'], prefix = 'City')

dummy = dummy.drop(['City_Not Mumbai Cities'], 1)

leads_dummified = pd.concat([leads_dummified, dummy], axis = 1)
leads_dummified.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization',

       'What is your current occupation', 'City'], axis=1, inplace=True)
leads_dummified.shape
leads_dummified.head(5)
from sklearn.model_selection import train_test_split

np.random.seed(0)

lead_df_train,lead_df_test=train_test_split(leads_dummified,train_size=0.7,random_state=100)
X_train = lead_df_train.drop(['Converted','Lead Number'], axis=1)

y_train = lead_df_train['Converted']
X_train.info()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_train.head(5)
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE



leads_reg = LogisticRegression()



rfe = RFE(leads_reg, 15)

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
rfe_selected_features = X_train.columns[rfe.support_]

rfe_selected_features
X_train_sm = sm.add_constant(X_train[rfe_selected_features])

model1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

result = model1.fit()

result.summary()
rfe_selected_features = rfe_selected_features.drop('Lead_Source_Organic Search', 1)

rfe_selected_features
X_train_sm = sm.add_constant(X_train[rfe_selected_features])

model2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

result = model2.fit()

result.summary()
rfe_selected_features = rfe_selected_features.drop('Lead_Source_Reference', 1)

rfe_selected_features
X_train_sm = sm.add_constant(X_train[rfe_selected_features])

model3 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

result = model3.fit()

result.summary()


rfe_selected_features = rfe_selected_features.drop('Last_Activity_Miscellaneous', 1)

rfe_selected_features
X_train_sm = sm.add_constant(X_train[rfe_selected_features])

model4 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

result = model4.fit()

result.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train[rfe_selected_features].columns

vif['VIF'] = [variance_inflation_factor(X_train[rfe_selected_features].values, i) for i in range(X_train[rfe_selected_features].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
final_selected_features = rfe_selected_features
y_train_pred = result.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Lead_Score_Prob':y_train_pred})

y_train_pred_final['Lead Number'] = leadscore['Lead Number']

y_train_pred_final.head()
y_train_pred_final['Lead_Score'] = round((y_train_pred_final['Lead_Score_Prob'] * 100),0)



y_train_pred_final.head()
y_train_pred_final['Predicted_Hot_Lead'] = y_train_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
y_train_pred_final['Lead_Score'] = round((y_train_pred_final['Lead_Score_Prob'] * 100),0)

y_train_pred_final['Lead_Score'] = y_train_pred_final['Lead_Score'].astype(int)

y_train_pred_final.head()
from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted_Hot_Lead )

print(confusion_matrix)
print('Accuracy for the Model 4 is {}%'.format(round(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted_Hot_Lead),2)*100 ))
TP = confusion_matrix[1,1] # true positive 

TN = confusion_matrix[0,0] # true negatives

FP = confusion_matrix[0,1] # false positives

FN = confusion_matrix[1,0] # false negatives
sensitivity = round((TP / float(TP+FN)),2)

specificity = round((TN / float(TN+FP)),2)



print('Sensitivity is {}% and Specificity is {}%'.format(sensitivity*100, specificity*100))
plot_roc(y_train_pred_final.Converted, y_train_pred_final.Predicted_Hot_Lead)
y_train_pred_final
numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['Probability','Accuracy','Sensitivity','Specificty', 'Precision'])



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    preci = cm1[1,1]/(cm1[1,1]+cm1[0,1])   #TP/TP+FP

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci, preci]   

print(cutoff_df)
sns.set_style('whitegrid')

sns.set_context('paper')



cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificty'], figsize=(10,6))

plt.xticks(np.arange(0,1,step=.05), size=8)

plt.yticks(size=12)

plt.show()
cut_off = 0.36
y_train_pred_final['Predicted_Hot_Lead'] = y_train_pred_final.Lead_Score_Prob.map( lambda x: 1 if x > cut_off else 0)

y_train_pred_final.head()
getRegressionMetrics(y_train_pred_final.Converted,y_train_pred_final.Predicted_Hot_Lead)
from sklearn.metrics import precision_recall_curve

p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Lead_Score_Prob)


plt.figure(figsize=(8, 4), dpi=100, facecolor='w', edgecolor='k', frameon='True')

plt.title('Precision vs Recall')

plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.xticks(np.arange(0, 1, step=0.05))

plt.show()
lead_df_test.head()
lead_df_test[num_cols] = scaler.transform(lead_df_test[num_cols])
rfe_selected_features
X_test = lead_df_test[rfe_selected_features]

y_test = lead_df_test[['Lead Number', 'Converted']]
print(X_test.shape)

print(y_test.shape)
X_test_sm = sm.add_constant(X_test)

y_test_pred = result.predict(X_test_sm)

y_test_pred[:10]
# Coverting it to df

y_pred_df = pd.DataFrame(y_test_pred)

# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)

# Remove index for both dataframes to append them side by side 

y_pred_df.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)

# Append y_test_df and y_pred_df

y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)

# Renaming column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Lead_Score_Prob'})

y_pred_final.head()
y_pred_final['Predicted_Hot_Lead'] = y_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > cut_off else 0)

y_pred_final.head()
getRegressionMetrics(y_pred_final.Converted,y_pred_final.Predicted_Hot_Lead)
y_pred_final['Lead_Score'] = round((y_pred_final['Lead_Score_Prob'] * 100),0)

y_pred_final['Lead_Score'] = y_pred_final['Lead_Score'].astype(int)
y_pred_final.head()
leads_dummified.shape
leads_dummified.head()
leads_dummified[num_cols] = scaler.transform(leads_dummified[num_cols])

leads_dummified.head()
cleaned_lead_sm = sm.add_constant(leads_dummified[rfe_selected_features])

cleaned_predicted = result.predict(cleaned_lead_sm)

cleaned_predicted
final_lead_score_df = leadscore.copy()

final_lead_score_df.head()
final_lead_score_df['Lead Score']=round(cleaned_predicted*100,2)

final_lead_score_df.head()
hot_leads = final_lead_score_df.sort_values(by='Lead Score',ascending=False)[['Lead Number','Lead Score']]

hot_leads[hot_leads['Lead Score']>36] 
final_lead_score_df['Is_Hot_Lead'] = final_lead_score_df['Lead Score'].map(lambda x: 1 if x > 36 else 0)

final_lead_score_df.sort_values(by='Lead Score',ascending=False).head(10)
coeff = result.params[1:]

coeff
feature_relevance = 100.0 * (coeff / coeff.max())

feature_relevance
sorted_idx = np.argsort(feature_relevance,kind='quicksort',order='list of str')

sorted_idx


pos = np.arange(sorted_idx.shape[0]) + .5



fig = plt.figure(figsize=(10,6))

ax =  fig.add_subplot(1, 1, 1)

ax.barh(pos, feature_relevance[sorted_idx], align='center', color = 'tab:blue',alpha=0.8)

ax.set_yticks(pos)

ax.set_yticklabels(np.array(rfe_selected_features)[sorted_idx], fontsize=12)

ax.set_xlabel('Relative Feature Importance', fontsize=14)



plt.tight_layout()   

plt.show()


pd.DataFrame(feature_relevance).reset_index().sort_values(by=0,ascending=False).head(3)