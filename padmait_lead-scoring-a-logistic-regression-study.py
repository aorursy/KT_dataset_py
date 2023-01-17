#Data Analysis & Data wrangling

import numpy as np

import pandas as pd

from collections import Counter



#Visualization

import matplotlib.pyplot as plt

import matplotlib.style as style

import seaborn as sns

%matplotlib inline



# Plot Style

sns.set_context("paper")

style.use('fivethirtyeight')



# Machine Learning Libraries



#Sci-kit learn libraries

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score



#statmodel libraries

from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
lead = pd.read_csv("/kaggle/input/lead-scoring-dataset/Lead Scoring.csv")

lead.head()
lead.tail()
#database dimension

print("Database dimension     :",lead.shape)

print("Database size          :",lead.size)

print("Number of Row          :",len(lead.index))

print("Number of Columns      :",len(lead.columns))
#checking numerical columns statistics

lead.describe()
#info about the column types etc. 

lead.info()
lead = lead.replace('Select', np.nan)
plt.figure(figsize = (18,8))

sns.heatmap(lead.isnull(),cbar = False)

plt.show()
#Column wise null values in train data set 

null_perc = pd.DataFrame(round((lead.isnull().sum())*100/lead.shape[0],2)).reset_index()

null_perc.columns = ['Column Name', 'Null Values Percentage']

null_value = pd.DataFrame(lead.isnull().sum()).reset_index()

null_value.columns = ['Column Name', 'Null Values']

null_lead = pd.merge(null_value, null_perc, on='Column Name')

null_lead.sort_values("Null Values", ascending = False)
#plotting the null value percentage

sns.set_style("white")

fig = plt.figure(figsize=(12,5))

null_lead = pd.DataFrame((lead.isnull().sum())*100/lead.shape[0]).reset_index()

ax = sns.pointplot("index",0,data=null_lead)

plt.xticks(rotation =90,fontsize =9)

ax.axhline(45, ls='--',color='red')

plt.title("Percentage of Missing values")

plt.ylabel("PERCENTAGE")

plt.xlabel("COLUMNS")

plt.show()
Row_Null50_Count = len(lead[lead.isnull().sum(axis=1)/lead.shape[1]>0.5])

print( 'Total number of rows with more than 50% null values are : ', Row_Null50_Count)
print("Total number of duplicate values in Prospect ID column :" , lead.duplicated(subset = 'Prospect ID').sum())

print("Total number of duplicate values in Lead Number column :" , lead.duplicated(subset = 'Lead Number').sum())
cols_to_drop = ['Prospect ID','Lead Number','How did you hear about X Education','Lead Profile',

                'Lead Quality','Asymmetrique Profile Score','Asymmetrique Activity Score',

               'Asymmetrique Activity Index','Asymmetrique Profile Index','Tags','Last Notable Activity']
#dropping unnecessary columns



lead.drop(cols_to_drop, 1, inplace = True)

len(lead.columns)
categorical_col = lead.select_dtypes(exclude =["number"]).columns.values

numerical_col = lead.select_dtypes(include =["number"]).columns.values

print("CATEGORICAL FEATURES : \n {} \n\n".format(categorical_col))

print("NUMERICAL FEATURES : \n {} ".format(numerical_col))
# Checking unique values and null values for the categorical columns

def Cat_info(df, categorical_column):

    df_result = pd.DataFrame(columns=["columns","values","unique_values","null_values","null_percent"])

    

    df_temp=pd.DataFrame()

    for value in categorical_column:

        df_temp["columns"] = [value]

        df_temp["values"] = [df[value].unique()]

        df_temp["unique_values"] = df[value].nunique()

        df_temp["null_values"] = df[value].isna().sum()

        df_temp["null_percent"] = (df[value].isna().sum()/len(df)*100).round(1)

        df_result = df_result.append(df_temp)

    

    df_result.sort_values("null_values", ascending =False, inplace=True)

    df_result.set_index("columns", inplace=True)

    return df_result
df_cat = Cat_info(lead, categorical_col)

df_cat
# Appending the columns to col_to_drop where only 1 category value is present



cols_to_drop = df_cat[df_cat['unique_values']==1].index.values.tolist() 

cols_to_drop
#dropping unnecessary columns



lead.drop(cols_to_drop, 1, inplace = True)

len(lead.columns)
categorical_col = lead.select_dtypes(exclude =["number"]).columns.values

new_cat = Cat_info(lead, categorical_col)

new_cat
lead['City'].value_counts(normalize=True)*100
# Let's check how City and Country are connected with each other

lead.groupby(['Country','City'])['Country'].count()
style.use('fivethirtyeight')

ax = sns.countplot(lead['City'],palette = 'Set2')

plt.xticks(rotation = 90)

plt.show()
lead.drop("City",axis=1, inplace = True)

len(lead.columns)
lead['Specialization'].value_counts(normalize = True)*100
plt.figure(figsize=(12,6))

ax = sns.countplot(lead['Specialization'],palette = 'Set2')

plt.xticks(rotation = 90)

plt.show()
lead['Specialization'] = lead['Specialization'].replace(np.nan, 'Others')

plt.figure(figsize=(12,6))

ax = sns.countplot(lead['Specialization'],palette = 'Set2')

plt.xticks(rotation = 90)

plt.show()
lead['What matters most to you in choosing a course'].value_counts(normalize = True)*100
lead.drop('What matters most to you in choosing a course', axis = 1, inplace=True)

len(lead.columns)
lead['What is your current occupation'].value_counts(normalize=True)*100
#lead['What is your current occupation'] = lead['What is your current occupation'].replace(np.nan, 'Unemployed')

lead['What is your current occupation'] = lead['What is your current occupation'].replace(np.nan, 'Unknown')

lead['What is your current occupation'].value_counts(normalize = True)*100
#Let's check how is the Country data distributed

lead['Country'].value_counts(normalize=True)
lead.drop('Country', axis = 1, inplace = True)

len(lead.columns)
print("Number of null values in Last Activity column is : ", lead['Last Activity'].isnull().sum())

print("Percentage of null values in Last Activity column is : ", round(lead['Last Activity'].isnull().sum()/lead.shape[0]*100,2))
lead['Last Activity'].value_counts(normalize = True)*100
lead['Last Activity'] = lead['Last Activity'].replace(np.nan, 'Email Opened')

print("Number of null values in Last Activity column is : ", lead['Last Activity'].isnull().sum())
print("Number of null values in Lead Source column is : ", lead['Lead Source'].isnull().sum())

print("Percentage of null values in Lead Source column is : ", round(lead['Lead Source'].isnull().sum()/lead.shape[0]*100,2))
lead['Lead Source'].value_counts(normalize = True)*100
lead['Lead Source'] = lead['Lead Source'].replace(np.nan, 'Google')

lead['Lead Source'] = lead['Lead Source'].replace(['google'], 'Google')

print("Number of null values in Lead Source column is : ", lead['Lead Source'].isnull().sum())
# Checking unique values and null values for the categorical columns

def Num_info(df, numeric_column):

    df_result = pd.DataFrame(columns=["columns","null_values","null_percent"])

    

    df_temp=pd.DataFrame()

    for value in numeric_column:

        df_temp["columns"] = [value]

        df_temp["null_values"] = df[value].isna().sum()

        df_temp["null_percent"] = (df[value].isna().sum()/len(df)*100).round(1)

        df_result = df_result.append(df_temp)

    

    df_result.sort_values("null_values", ascending =False, inplace=True)

    df_result.set_index("columns", inplace=True)

    return df_result
df_num = Num_info(lead,numerical_col)

df_num
plt.figure(figsize = (12,6))

plt.subplot(1,2,1)

sns.distplot(lead['TotalVisits'])

plt.subplot(1,2,2)

sns.boxplot(lead['TotalVisits'])

plt.show()
lead['TotalVisits'].fillna(lead['TotalVisits'].median(), inplace=True)

lead['TotalVisits'].isnull().sum()
plt.figure(figsize = (12,6))

plt.subplot(1,2,1)

sns.distplot(lead['Page Views Per Visit'])

plt.subplot(1,2,2)

sns.boxplot(lead['Page Views Per Visit'])

plt.show()
lead['Page Views Per Visit'].fillna(lead['Page Views Per Visit'].median(), inplace=True)

lead['Page Views Per Visit'].isnull().sum()
converted = lead['Converted'].value_counts().rename_axis('unique_values').to_frame('counts')

converted


my_circle=plt.Circle( (0,0), 0.7, color='white')

plt.pie(converted.counts, labels = ['No','Yes'],colors = ['red','green'],autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()

# function for plotting repetitive countplots in univariate categorical analysis on the lead dataset

# This function will create two subplots: 

# 1. Count plot of categorical column w.r.t Converted; 

# 2. Percentage of converted leads within column



def univariate_categorical(feature,label_rotation=False,horizontal_layout=True):

    temp_count = lead[feature].value_counts()

    temp_perc = lead[feature].value_counts(normalize = True)

    df1 = pd.DataFrame({feature: temp_count.index,'Total Leads': temp_count.values,'% Values': temp_perc.values * 100})

    print(df1)

    

    # Calculate the percentage of Converted=1 per category value

    cat_perc = lead[[feature, 'Converted']].groupby([feature],as_index=False).mean()

    cat_perc["Converted"] = cat_perc["Converted"]*100

    cat_perc.sort_values(by='Converted', ascending=False, inplace=True)

    

    if(horizontal_layout):

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

    else:

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))   

    # 1. Subplot 1: Count plot of categorical column

    sns.set_palette("Set2")

    s = sns.countplot(ax=ax1, 

                    x = feature, 

                    data=lead,

                    hue ="Converted",

                    order=cat_perc[feature],

                    palette=['r','g'])



    # Define common styling

    ax1.set_title(feature, fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 

    ax1.legend(['Not Converted','Converted'])

    

    

    if(label_rotation):

        s.set_xticklabels(s.get_xticklabels(),rotation=90)

    

    # 2. Subplot 2: Percentage of defaulters within the categorical column

    s = sns.barplot(ax=ax2, 

                    x = feature, 

                    y='Converted', 

                    order=cat_perc[feature], 

                    data=cat_perc,

                    palette='Set2')

    

    if(label_rotation):

        s.set_xticklabels(s.get_xticklabels(),rotation=90)

    plt.ylabel('Percent of Converted leads [%]', fontsize=15)

    plt.xlabel(feature,fontsize=15) 

    plt.tick_params(axis='both', which='major', labelsize=10)

    ax2.set_title(feature + "( Converted % )", fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 



    plt.show();
lead.columns
# Renaming some of the column headers which has long header



lead.rename(columns={'What is your current occupation': 'Occupation', 

                     'Through Recommendations': 'Recommendation',

                     'A free copy of Mastering The Interview': 'Free Copy'                   

                    },inplace = True)

lead.columns
#Run the function to get plot categorical plots 

univariate_categorical("Lead Origin",label_rotation=True)
#Run the function to get plot categorical plots

univariate_categorical("Lead Source",label_rotation=True)
lead['Lead Source'] = lead['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',

  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Other Sources')
#Running the function again to check the updated statistics

univariate_categorical("Lead Source",label_rotation=True)
#Run the function to get plot categorical plots

univariate_categorical("Do Not Email")
#Run the function to get plot categorical plots

univariate_categorical("Last Activity",label_rotation=True)
# Let's keep considerable last activities as such and club all others to "Other_Activity"

lead['Last Activity'] = lead['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 

                                                       'Visited Booth in Tradeshow', 'Approached upfront',

                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other Activity')
#Run the function to get plot categorical plots

univariate_categorical("Last Activity",label_rotation=True)
#Run the function to get plot categorical plots

univariate_categorical("Specialization",label_rotation=True)
#Run the function to get plot categorical plots

univariate_categorical("Occupation",label_rotation=True)
def pieplot(col):

    my_circle=plt.Circle( (0,0), 0.7, color='white')

    converted = lead[col].value_counts().rename_axis('unique_values').to_frame('counts')

    plt.pie(converted.counts, labels = ["No","Yes"],colors = ['red','green'],autopct='%1.1f%%')

    p=plt.gcf()

    p.gca().add_artist(my_circle)

    plt.title(col)
# Lets lookinto the data distribution of the following columns

col = ['Do Not Call','Search', 'Newspaper', 'Newspaper Article', 'Digital Advertisement', 'X Education Forums', 'Free Copy','Recommendation']

plt.figure(figsize = (12,8))

i=1

for each_col in col:

    plt.subplot(2,4,i)

    pieplot(each_col)

    i+=1
#Run the function to get plot categorical plots

univariate_categorical("Free Copy",label_rotation=True)
lead.drop(col,axis = 1, inplace = True)

len(lead.columns)
lead.columns
numerical_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']

plt.figure(figsize=(12,12))



i=1

for col in numerical_cols:

    plt.subplot(2,3,i)

    sns.distplot(lead[col])

    plt.subplot(2,3,3+i)

    sns.boxplot(y=lead[col], x = lead['Converted'])

    i+=1
plt.figure(figsize =(20,20))

sns.pairplot(lead[numerical_col],hue="Converted",kind='scatter', plot_kws={'alpha':0.4},palette = 'Dark2')                                  

plt.show()
#Checking the detailed percentile values

lead.describe(percentiles=[.1,.5,.25,.75,.90,.95,.99])
numerical_col
#Plotting the numerical columns for outlier values

i=1

plt.figure(figsize=[16,8])

for col in numerical_col:

    plt.subplot(2,2,i)

    sns.boxplot(y=lead[col])

    plt.title(col)

    plt.ylabel('')

    i+=1
#Capping the data at 95% percetile value

Q4 = lead['TotalVisits'].quantile(0.95) # Get 95th quantile

print("Total number of rows getting capped for TotalVisits column : ",len(lead[lead['TotalVisits'] >= Q4]))

lead.loc[lead['TotalVisits'] >= Q4, 'TotalVisits'] = Q4 # outlier capping



Q4 = lead['Page Views Per Visit'].quantile(0.95) # Get 95th quantile

print("Total number of rows getting capped for Page Views Per Visit column : ",len(lead[lead['Page Views Per Visit'] >= Q4]))

lead.loc[lead['Page Views Per Visit'] >= Q4, 'Page Views Per Visit'] = Q4 # outlier capping
#replotting the graphs to check for outlier treatment

i=1

plt.figure(figsize=[16,8])

for col in numerical_col:

    plt.subplot(2,2,i)

    sns.boxplot(y=lead[col])

    plt.title(col)

    plt.ylabel('')

    i+=1
# Checking the percentile values again 

lead.describe(percentiles=[.1,.5,.25,.75,.90,.95,.99])
# Checking the unique value counts for categorcial columns

lead.nunique().sort_values()
# Checking the categorical values for 'Do Not Email' feature

lead['Do Not Email'].value_counts()
# List of variables to map



# Defining the map function

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})



# Applying the function to the lead YES/NO variable list

lead['Do Not Email'] = lead[['Do Not Email']].apply(binary_map)
# rechecking the categorical values for 'Do Not Email' feature

lead['Do Not Email'].value_counts()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(lead[['Lead Origin', 'Lead Source', 'Occupation', 'Last Activity', 'Specialization']], drop_first=True)



# Adding the results to the master dataframe

lead = pd.concat([lead, dummy1], axis=1)



lead.head()
# We have created dummies for the below variables, so we can drop them

lead = lead.drop(['Lead Origin', 'Lead Source', 'Occupation', 'Last Activity', 'Specialization'], axis=1)

lead.info()
# Visualizing the data using heatmap

plt.figure(figsize=[15,15])

sns.heatmap(lead.corr(), cmap="RdYlGn",linewidth =1)

plt.show()
print('Total number of columns after One-Hot Encoding : ',len(lead.columns))
corr_lead = lead.corr()

corr_lead = corr_lead.where(np.triu(np.ones(corr_lead.shape),k=1).astype(np.bool))

corr_df = corr_lead.unstack().reset_index()

corr_df.columns =['VAR1','VAR2','Correlation']

corr_df.dropna(subset = ["Correlation"], inplace = True) 

corr_df.sort_values(by='Correlation', ascending=False, inplace=True)



# Top 5 Positive correlated variables

corr_df.head(5)
corr_df.sort_values(by='Correlation', ascending=True, inplace=True)



# Top 5 Negatively correlated variables

corr_df.head(5)
# target variable

Y = lead['Converted']

X = lead.drop(['Converted'], axis=1)



# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)
#Checking the shape of the created Train & Test DFs

print(" Shape of X_train is : ",X_train.shape)

print(" Shape of y_train is : ",y_train.shape)

print(" Shape of X_test is  : ",X_test.shape)

print(" Shape of y_test is  : ",y_test.shape)
scaler = StandardScaler()



X_train[['Total Time Spent on Website','Page Views Per Visit','TotalVisits']] = scaler.fit_transform(X_train[['Total Time Spent on Website','Page Views Per Visit','TotalVisits']])

X_train.head()
# Using RFE to reduce the feature count from 54 to 20

logreg = LogisticRegression()

rfe = RFE(logreg, 20)           

rfe = rfe.fit(X_train, y_train)

#checking the output of RFE

list(zip(X_train.columns, rfe.support_, rfe.ranking_))
#checking which columns remained after RFE

rfe_col = X_train.columns[rfe.support_]

rfe_col
#Columns which have been removed after RFE

X_train.columns[~rfe.support_]
# Functions to repeat Logictis regression model and VIF calculation repeatedly



# function to build logistic regression model

def build_logistic_model(feature_list):

    X_train_local = X_train[feature_list] # get feature list for VIF

    X_train_sm = sm.add_constant(X_train_local) # required by statsmodels   

    log_model = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial()).fit() # build model and learn coefficients  

    return(log_model, X_train_sm) # return the model and the X_train fitted with constant 



#function to calculate VIF

def calculate_VIF(X_train):  # Calculate VIF for features

    vif = pd.DataFrame()

    vif['Features'] = X_train.columns # Read the feature names

    vif['VIF'] = [variance_inflation_factor(X_train.values,i) for i in range(X_train.shape[1])] # calculate VIF

    vif['VIF'] = round(vif['VIF'],2)

    vif.sort_values(by='VIF', ascending = False, inplace=True)  

    return(vif) # returns the calculated VIFs for all the features
features = list(rfe_col) #  Use RFE selected variables

log_model1, X_train_sm1 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction

log_model1.summary()
#Checking VIF values

calculate_VIF(X_train)
features.remove('Occupation_Housewife') # Remove 'Occupation_Housewife number' from RFE features list

log_model2, X_train_sm2 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction

log_model2.summary()
#Checking VIF Values

calculate_VIF(X_train[features])
features.remove('Specialization_Retail Management')

log_model3, X_train_sm3 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction

log_model3.summary()
#Checking VIF Values

calculate_VIF(X_train[features])
features.remove('Lead Source_Facebook')

log_model4, X_train_sm4 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction

log_model4.summary()
#Checking VIF Values

calculate_VIF(X_train[features])
features.remove('Specialization_Rural and Agribusiness')

log_model5, X_train_sm5 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction

log_model5.summary()
#Checking VIF Values

calculate_VIF(X_train[features])
def calculate_woe_iv(dataset, feature, target):

    lst = []

    for i in range(dataset[feature].nunique()):

        val = list(dataset[feature].unique())[i]

        lst.append({

            'Value': val,

            'All': dataset[dataset[feature] == val].count()[feature],

            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],

            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]

        })

        

    dset = pd.DataFrame(lst)

    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()

    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()

    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])

    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']

    iv = dset['IV'].sum()

    

    dset = dset.sort_values(by='WoE')

    

    return dset, iv
for col in lead.columns:

    if col in features:

        df, iv = calculate_woe_iv(lead, col, 'Converted')

        print('IV score of column : ',col, " is ", round(iv,4))
features.remove('Occupation_Unknown')

log_model6, X_train_sm6 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction

log_model6.summary()
#Checking VIF Values

calculate_VIF(X_train[features])
for col in lead.columns:

    if col in features:

        df, iv = calculate_woe_iv(lead, col, 'Converted')

        print('IV score of column : ',col, " is ", round(iv,4))
features.remove('Specialization_Others')

log_model7, X_train_sm7 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction

log_model7.summary()
#Checking VIF Values

calculate_VIF(X_train[features])
for col in lead.columns:

    if col in features:

        df, iv = calculate_woe_iv(lead, col, 'Converted')

        print('IV score of column : ',col, " is ", round(iv,4))
features.remove('Specialization_Hospitality Management')

log_model8, X_train_sm8 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction

log_model8.summary()
#Checking VIF Values

calculate_VIF(X_train[features])
features.remove('Last Activity_Other Activity')

log_model9, X_train_sm9 = build_logistic_model(features) # Call the function and get the model and the X_train_sm for prediction

log_model9.summary()
# How many features in the model ?

len(features)
# Create a matrix to Print the Accuracy, Sensitivity and Specificity

def lg_metrics(confusion_matrix):

    TN =confusion_matrix[0,0]

    TP =confusion_matrix[1,1]

    FP =confusion_matrix[0,1]

    FN =confusion_matrix[1,0]

    accuracy = (TP+TN)/(TP+TN+FP+FN)

    speci = TN/(TN+FP)

    sensi = TP/(TP+FN)

    precision = TP/(TP+FP)

    recall = TP/(TP+FN)

    TPR = TP/(TP + FN)

    TNR = TN/(TN + FP)

    FPR = FP/(TN + FP)

    FNR = FN/(TP + FN)

    pos_pred_val = TP /(TP+FP)

    neg_pred_val = TN /(TN+FN)

    

    print ("Model Accuracy value is              : ", round(accuracy*100,2),"%")

    print ("Model Sensitivity value is           : ", round(sensi*100,2),"%")

    print ("Model Specificity value is           : ", round(speci*100,2),"%")

    print ("Model Precision value is             : ", round(precision*100,2),"%")

    print ("Model Recall value is                : ", round(recall*100,2),"%")

    print ("Model True Positive Rate (TPR)       : ", round(TPR*100,2),"%")

    print ("Model False Positive Rate (FPR)      : ", round(FPR*100,2),"%")

    print ("Model Poitive Prediction Value is    : ", round(pos_pred_val*100,2),"%")

    print ("Model Negative Prediction value is   : ", round(neg_pred_val*100,2),"%")
# Getting the predicted values on the train set

y_train_pred = log_model9.predict(X_train_sm9)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
#Creating a dataframe with the actual Converted flag and the Predicted probabilities

y_train_pred_final = pd.DataFrame({'Converted_IND':y_train.values, 'Converted_Prob':y_train_pred})

y_train_pred_final['Prospect_IND'] = y_train.index

y_train_pred_final.head()
#Finding Optimal Cutoff Point

# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci','Precision','Recall'])

from sklearn.metrics import confusion_matrix



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final['Converted_IND'], y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    prec, rec, thresholds = precision_recall_curve(y_train_pred_final['Converted_IND'], y_train_pred_final[i])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci, prec[1], rec[1]]

cutoff_df
# Let's plot accuracy sensitivity and specificity for various probabilities.

plt.figure(figsize=(18,8))

sns.set_style("whitegrid")

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.xticks(np.arange(0,1,step=0.05),size=8)

plt.axvline(x=0.335, color='r', linestyle='--') # additing axline

plt.yticks(size=12)

plt.show()
y_train_pred_final['final_predicted_1'] = y_train_pred_final['Converted_Prob'].map( lambda x: 1 if x > 0.335 else 0)

y_train_pred_final.drop([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],axis = 1, inplace = True) # deleting the unnecessary columns

y_train_pred_final.head()
# Let's assign Lead_score for the leads in Train Data Set

y_train_pred_final['lead_score_1']=(y_train_pred_final['Converted_Prob']*100).astype("int64")

y_train_pred_final.sort_values(by='Converted_Prob',ascending=False)
# Function for Confusion Matrix :

def draw_cm( actual, predicted, cmap ): 

    cm = metrics.confusion_matrix( actual, predicted, [0,1] ) 

    sns.heatmap(cm, annot=True, fmt='.0f', cmap=cmap,

    xticklabels = ["Not Converted", "Converted"] ,

    yticklabels = ["Not Converted", "Converted"] ) 

    plt.ylabel('True labels')

    plt.xlabel('Predicted labels') 

    plt.show()
#Plotting the Confusion Matrix

draw_cm( y_train_pred_final['Converted_IND'], y_train_pred_final['final_predicted_1'], "GnBu")
conf_matrix = confusion_matrix(y_train_pred_final['Converted_IND'], y_train_pred_final['final_predicted_1'] )



lg_metrics(conf_matrix)
# Classification Record : Precision, Recall and F1 Score

print( metrics.classification_report( y_train_pred_final['Converted_IND'], y_train_pred_final['final_predicted_1'] ) )
print("F1 Score: {}".format(f1_score(y_train_pred_final['Converted_IND'], y_train_pred_final['final_predicted_1'])))
# Function to plot ROC Curve

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
# recoring the values FPR, TPR and Thresholds:

fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final['Converted_IND'], y_train_pred_final['Converted_Prob'] , drop_intermediate = False )
#plotting the ROC curve 

draw_roc(y_train_pred_final['Converted_IND'], y_train_pred_final['Converted_Prob'])
p, r, thresholds = precision_recall_curve(y_train_pred_final['Converted_IND'], y_train_pred_final['Converted_Prob'])
# Plotting the Precision-Recall Trade off Curve

plt.figure(figsize=(15,8))

sns.set_style("whitegrid")

sns.set_context("paper")

plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.axvline(x=0.404, color='b', linestyle='--') # additing axline

plt.xticks(np.arange(0,1,step=0.02),size=8)

plt.yticks(size=12)



plt.show()
# plotting the Train dataset again with 0.42 as cutoff

y_train_pred_final['final_predicted_2'] = y_train_pred_final['Converted_Prob'].map( lambda x: 1 if x > 0.404 else 0)

y_train_pred_final.head()
#Plotting the Confusion Matrix

draw_cm( y_train_pred_final['Converted_IND'], y_train_pred_final['final_predicted_2'], "GnBu")
#Data based on cutoff received from Precision-Recall Trade off

conf_matrix = confusion_matrix(y_train_pred_final['Converted_IND'], y_train_pred_final['final_predicted_2'] )

lg_metrics(conf_matrix)
# Classification Record : Precision, Recall and F1 Score

print( metrics.classification_report( y_train_pred_final['Converted_IND'], y_train_pred_final['final_predicted_2'] ) )
# Scaling the test dataset :

X_test[['Total Time Spent on Website','Page Views Per Visit','TotalVisits']] = scaler.transform(X_test[['Total Time Spent on Website','Page Views Per Visit','TotalVisits']])

X_test.head()
# Selecting only the columns used in final model of Train Dataset

X_test = X_test[features]

X_test.head()
#adding contant value

X_test_sm = sm.add_constant(X_test)

X_test_sm.columns
# Predicting the final test model 

y_test_pred = log_model9.predict(X_test_sm)
#checking the top 10 rows

y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_test_pred = pd.DataFrame(y_test_pred)

y_test_pred.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)

y_test_df.head()
# Putting CustID to index

y_test_df['Prospect_IND'] = y_test_df.index



# Removing index for both dataframes to append them side by side 

y_test_pred.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)



# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_test_pred],axis=1)

y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})

y_pred_final= y_pred_final.rename(columns={ 'Converted' : 'Converted_IND'})



# Rearranging the columns

y_pred_final = y_pred_final.reindex(['Prospect_IND','Converted_IND','Converted_Prob'], axis=1)

y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.335 else 0)

y_pred_final.head()
#Plotting the Confusion Matrix

draw_cm( y_pred_final['Converted_IND'], y_pred_final['final_predicted'], "GnBu")
conf_matrix = confusion_matrix(y_pred_final['Converted_IND'], y_pred_final['final_predicted'])



lg_metrics(conf_matrix)
# Invoking the functio to draw ROC curve



draw_roc( y_pred_final['Converted_IND'], y_pred_final['Converted_Prob'])
# Classification Record : Precision, Recall and F1 Score

print( metrics.classification_report( y_pred_final['Converted_IND'], y_pred_final['final_predicted'] ) )
# Let's assign Lead_score for the leads in Test Data Set : 

y_pred_final['lead_score']=(y_pred_final['Converted_Prob']*100).astype("int64")

y_pred_final.sort_values(by='Converted_Prob',ascending=False)
# checking the data from top 

y_pred_final.head(5)
# checking the data from bottom 

y_pred_final.tail(5)
# Let's look into final model features and coefficients 

pd.options.display.float_format = '{:.2f}'.format

final_parameters = log_model9.params[1:]

final_parameters
#Getting a relative coeffient value for all the features wrt the feature with the highest coefficient



top_predictors = final_parameters

top_predictors = 100.0 * (top_predictors / top_predictors.max())

top_predictors
# Plotting the predictors based on their relative importance

top_predictors_sort = np.argsort(top_predictors,kind='quicksort',order='list of str')

fig = plt.figure(figsize = (12,8))

pos = np.arange(top_predictors_sort.shape[0]) + .5



fig1 = plt.figure(figsize=(10,6))

ax = fig1.add_subplot(1, 1, 1)

ax.barh(pos, top_predictors[top_predictors_sort])

ax.set_yticks(pos)

ax.set_yticklabels(np.array(X_train[features].columns)[top_predictors_sort], fontsize=13)

ax.set_xlabel('Top Predictors Relative Importance', fontsize=15)

plt.show()