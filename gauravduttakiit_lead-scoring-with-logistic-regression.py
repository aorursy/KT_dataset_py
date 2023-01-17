# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# visulaisation

from matplotlib.pyplot import xticks

%matplotlib inline
# Data display coustomization

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
lead = pd.read_csv(r"/kaggle/input/lead-scoring-dataset/Lead Scoring.csv")

lead.head()
word=pd.read_excel(r"/kaggle/input/lead-scoring-dataset/Leads Data Dictionary.xlsx")

word.head()
pd.set_option('display.max_colwidth', -1)

word.drop('Unnamed: 0',inplace=True,axis=1)

word.columns = word.iloc[1]

word = word.iloc[2:]

word.reset_index(drop=True, inplace=True)

word.head(len(word))
lead.drop(['Asymmetrique Activity Index','Asymmetrique Profile Index',

           'Asymmetrique Activity Score','Asymmetrique Profile Score',

           'Tags','Lead Quality','Lead Profile'], 1, inplace = True)
lead_dub = lead.copy()



# Checking for duplicates and dropping the entire duplicate row if any

lead_dub.drop_duplicates(subset=None, inplace=True)

lead_dub.shape
lead.shape
lead.shape
lead.info()
lead.describe()
# As we can observe that there are select values for many column.

#This is because customer did not select any option from the list, hence it shows select.

# Select values are as good as NULL.



# Converting 'Select' values to NaN.

lead = lead.replace('Select', np.nan)

lead.head()
lead.isnull().sum()
round(100*(lead.isnull().sum()/len(lead.index)), 2)
# we will drop the columns having more than 60% NA values.

lead = lead.drop(lead.loc[:,list(round(100*(lead.isnull().sum()/len(lead.index)), 2)>60)].columns, 1)
round(100*(lead.isnull().sum()/len(lead.index)), 2)
#dropping Lead Number and Prospect ID since they have all unique values



lead.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)
lead.head()
round(100*(lead.isnull().sum()/len(lead.index)), 2)
# City
lead.City.value_counts()

lead.City.describe()
plt.figure(figsize = (10,5))

ax= sns.countplot(lead['City'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

plt.show()
# Around 57.8% of the data available  is Mumbai so we can impute Mumbai in the missing values.
lead['City'] = lead['City'].replace(np.nan, 'Mumbai')
plt.figure(figsize = (10,5))

ax= sns.countplot(lead['City'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

plt.show()
# Specailization
lead.Specialization.describe()
lead.Specialization.value_counts()
plt.figure(figsize = (20,5))

ax= sns.countplot(lead['Specialization'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

plt.show()
# It maybe the case that lead has not entered any specialization if his/her option is not availabe on the list,

#  may not have any specialization or is a student.

# Hence we can make a category "Others" for missing values. 

lead['Specialization'] = lead['Specialization'].replace(np.nan, 'Others')
plt.figure(figsize = (20,5))

ax= sns.countplot(lead['Specialization'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

plt.show()
round(100*(lead.isnull().sum()/len(lead.index)), 2)
# What matters most to you in choosing a course
lead['What matters most to you in choosing a course'].describe()
lead['What matters most to you in choosing a course'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(lead['What matters most to you in choosing a course'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
# Blanks in the this column may be imputed by 'Better Career Prospects'.
lead['What matters most to you in choosing a course'] = lead['What matters most to you in choosing a course'].replace(np.nan, 'Better Career Prospects')
plt.figure(figsize = (10,5))

ax= sns.countplot(lead['What matters most to you in choosing a course'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
# Occupation
lead['What is your current occupation'].describe()
lead['What is your current occupation'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(lead['What is your current occupation'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
# 86% entries are of Unemployed so we can impute "Unemployed" in it.
lead['What is your current occupation'] = lead['What is your current occupation'].replace(np.nan, 'Unemployed')
plt.figure(figsize = (10,5))

ax= sns.countplot(lead['What is your current occupation'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
# Country
lead['Country'].describe()
lead['Country'].value_counts()
plt.figure(figsize = (20,5))

ax= sns.countplot(lead['Country'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
# Country is India for most values so let's impute the same in missing values.

lead['Country'] = lead['Country'].replace(np.nan, 'India')
plt.figure(figsize = (20,5))

ax= sns.countplot(lead['Country'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
round(100*(lead.isnull().sum()/len(lead.index)), 2)
lead.isnull().sum()
# Rest missing values are under 1.5% so we can drop these rows.

lead.dropna(inplace = True)
round(100*(lead.isnull().sum()/len(lead.index)), 2)
lead.isnull().sum()
data_retailed= len(lead)* 100 / len(lead_dub)

print("{} % of original rows is available for EDA".format(round(data_retailed,2)))
lead.shape
# Converted is the target variable, Indicates whether a lead has been successfully converted (1) or not (0).
Converted = round((sum(lead['Converted'])/len(lead['Converted'].index))*100,2)



print("We have almost {} %  Converted rate".format(Converted))



plt.figure(figsize = (10,5))

ax= sns.countplot(x = "Lead Origin", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
plt.figure(figsize = (25,5))

ax= sns.countplot(x = "Lead Source", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['Lead Source'] = lead['Lead Source'].replace(['google'], 'Google')

lead['Lead Source'] = lead['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',

  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "Lead Source", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
plt.figure(figsize = (20,5))

plt.subplot(1,2,1)

ax= sns.countplot(x = "Do Not Email", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.subplot(1,2,2)

ax= sns.countplot(x = "Do Not Call", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])
plt.figure(figsize = (10,5))

sns.violinplot(lead['TotalVisits'])

plt.show()
# As we can see there are a number of outliers in the data.

# We will cap the outliers to 95% value for analysis.
percentiles = lead['TotalVisits'].quantile([0.05,0.95]).values

lead['TotalVisits'][lead['TotalVisits'] <= percentiles[0]] = percentiles[0]

lead['TotalVisits'][lead['TotalVisits'] >= percentiles[1]] = percentiles[1]
plt.figure(figsize = (10,5))

sns.violinplot(lead['TotalVisits'])

plt.show()
plt.figure(figsize = (10,5))

sns.violinplot(y = 'TotalVisits', x = 'Converted', data = lead)

plt.show()
lead['Total Time Spent on Website'].describe()
plt.figure(figsize = (10,5))

sns.violinplot(lead['Total Time Spent on Website'])

plt.show()
plt.figure(figsize = (10,5))

sns.violinplot(y = 'Total Time Spent on Website', x = 'Converted', data = lead)

plt.show()
lead['Page Views Per Visit'].describe()
plt.figure(figsize = (10,5))

sns.violinplot(lead['Page Views Per Visit'])

plt.show()
# As we can see there are a number of outliers in the data.

# We will cap the outliers to 95% value for analysis.
percentiles = lead['Page Views Per Visit'].quantile([0.05,0.95]).values

lead['Page Views Per Visit'][lead['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]

lead['Page Views Per Visit'][lead['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]
plt.figure(figsize = (10,5))

sns.violinplot(lead['Page Views Per Visit'])

plt.show()
plt.figure(figsize = (10,5))

sns.violinplot(y = 'Page Views Per Visit', x = 'Converted', data = lead)

plt.show()
lead['Last Activity'].describe()
lead['Last Activity'].value_counts()
plt.figure(figsize = (25,5))

ax= sns.countplot(x = "Last Activity", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
# Let's keep considerable last activities as such and club all others to "Other_Activity"

lead['Last Activity'] = lead['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 

                                                       'Visited Booth in Tradeshow', 'Approached upfront',

                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'],

                                                      'Other_Activity')
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "Last Activity", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead.Country.describe()
lead.Country.value_counts()
plt.figure(figsize = (25,5))

ax= sns.countplot(x = "Country", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead.Specialization.describe()
lead.Specialization.value_counts()
lead['Specialization'] = lead['Specialization'].replace(['Others'], 'Other_Specialization')
plt.figure(figsize = (25,5))

ax= sns.countplot(x = "Specialization", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['What is your current occupation'].describe()
lead['What is your current occupation'].value_counts()
lead['What is your current occupation'] = lead['What is your current occupation'].replace(['Other'], 'Other_Occupation')
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "What is your current occupation", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['What matters most to you in choosing a course'].describe()
lead['What matters most to you in choosing a course'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "What matters most to you in choosing a course", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead.Search.describe()
lead.Search.value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "Search", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead.Magazine.describe()
lead.Magazine.value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "Magazine", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['Newspaper Article'].describe()
lead['Newspaper Article'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "Newspaper Article", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['X Education Forums'].describe()
lead['X Education Forums'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "X Education Forums", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['Newspaper'].describe()
lead['Newspaper'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "Newspaper", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['Digital Advertisement'].describe()
lead['Digital Advertisement'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "Digital Advertisement", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['Through Recommendations'].describe()
lead['Through Recommendations'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "Through Recommendations", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['Receive More Updates About Our Courses'].describe()
lead['Receive More Updates About Our Courses'].value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "Receive More Updates About Our Courses", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['Update me on Supply Chain Content'].describe()
lead['Update me on Supply Chain Content'].value_counts()
plt.figure(figsize = (10,6))

ax= sns.countplot(x = "Update me on Supply Chain Content", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['Get updates on DM Content'].describe()
lead['Get updates on DM Content'].value_counts()
plt.figure(figsize = (10,6))

ax= sns.countplot(x = "Get updates on DM Content", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['I agree to pay the amount through cheque'].describe()
lead['I agree to pay the amount through cheque'].value_counts()
plt.figure(figsize = (10,6))

ax= sns.countplot(x = "I agree to pay the amount through cheque", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['A free copy of Mastering The Interview'].describe()
lead['A free copy of Mastering The Interview'].value_counts()
plt.figure(figsize = (10,6))

ax= sns.countplot(x = "A free copy of Mastering The Interview", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead.City.describe()
lead.City.value_counts()
plt.figure(figsize = (10,6))

ax= sns.countplot(x = "City", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['Last Notable Activity'].describe()
lead['Last Notable Activity'].value_counts()
plt.figure(figsize = (20,6))

ax= sns.countplot(x = "Last Notable Activity", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead = lead.drop(['What matters most to you in choosing a course','Search',

                  'Magazine','Newspaper Article','X Education Forums','Newspaper',

           'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',

                  'Update me on Supply Chain Content',

           'Get updates on DM Content','I agree to pay the amount through cheque',

                  'A free copy of Mastering The Interview','Country'],1)
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (10,5))

sns.heatmap(lead.corr(), annot = True, cmap="rainbow")

plt.show()
print(lead['Last Activity'].value_counts())

print(lead['Last Notable Activity'].value_counts())
#Values under 'Last Activity' are coverved as values under 'Last Notable Activity'. Either of one can be dropped. 

lead.drop(['Last Notable Activity'], 1, inplace = True)
print("Original Columns {} % Retained".format(round((100* len(lead.columns)/len(lead_dub.columns)),2)))
print("Original Rows {} % Retained".format(round((len(lead)*100)/len(lead_dub),2)))

print("Original Data {} % Retained".format(round((len(lead) * 

                                                     len(lead.columns))*100/(len(lead_dub.columns)*len(lead_dub)),2)))
lead.shape
lead.head()
# List of variables to map



varlist =  ['Do Not Email', 'Do Not Call']



# Defining the map function

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})



# Applying the function to the housing list

lead[varlist] = lead[varlist].apply(binary_map)

lead.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(lead[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',

                              'City']], drop_first=True)



dummy1.head()
# Adding the results to the master dataframe

lead = pd.concat([lead, dummy1], axis=1)

lead.head()
lead = lead.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization',

                  'What is your current occupation','City'], axis = 1)



lead.head()

lead.shape
from sklearn.model_selection import train_test_split



# Putting feature variable to X

X = lead.drop(['Converted'], axis=1)
X.head()
X.shape
# Putting response variable to y

y = lead['Converted']
y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=333)
X_train.head()
X_train.shape
X_test.head()
X_test.shape
y_train.head()
y_train.shape
y_test.head()
y_test.shape
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])



X_train.head()
# Checking the Converted Rate

Converted = round((sum(lead['Converted'])/len(lead['Converted'].index))*100,2)

print("We have almost {} %  Converted rate after successful data manipulation".format(Converted))
import statsmodels.api as sm
# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



from sklearn.feature_selection import RFE

rfe = RFE(logreg,18)             # running RFE with 18 variables as output

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col1 = col.drop('What is your current occupation_Housewife',1)
col1
X_train_sm = sm.add_constant(X_train[col1])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col1].columns

vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col1 = col1.drop('Lead Source_Reference',1)
X_train_sm = sm.add_constant(X_train[col1])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col1].columns

vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col1 = col1.drop('What is your current occupation_Unemployed',1)
X_train_sm = sm.add_constant(X_train[col1])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
vif = pd.DataFrame()

vif['Features'] = X_train[col1].columns

vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col1 = col1.drop('What is your current occupation_Student',1)
X_train_sm = sm.add_constant(X_train[col1])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
vif = pd.DataFrame()

vif['Features'] = X_train[col1].columns

vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col1 = col1.drop('Last Activity_Unreachable',1)
X_train_sm = sm.add_constant(X_train[col1])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
vif = pd.DataFrame()

vif['Features'] = X_train[col1].columns

vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col1 = col1.drop('Last Activity_Unsubscribed',1)
X_train_sm = sm.add_constant(X_train[col1])

logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm6.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics



# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
# Checking VIF
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col1].columns

vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)

# Calculate false postive rate - predicting Converted when customer does not have Converted

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

fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, 

                                         y_train_pred_final.Converted_prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)

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
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'],figsize = (10,5))

plt.grid(True)

plt.show()
#### From the curve above, 0.34 is the optimum point to take it as a cutoff probability



y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.34 else 0)



y_train_pred_final.head()
y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))



y_train_pred_final.head()
# Let's check the overall accuracy.

trainaccuracy= metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)

trainaccuracy




confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )

confusion2







TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our model

trainsensitivity= TP / float(TP+FN)

trainsensitivity
# Let us calculate specificity

trainspecificity= TN / float(TN+FP)

trainspecificity
# Calculate false postive rate - predicting Converted when customer does not have Converted

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print(TN / float(TN+ FN))
#Using sklearn utilities for the same
from sklearn.metrics import precision_score, recall_score
precision= precision_score(y_train_pred_final.Converted , y_train_pred_final.predicted)

precision
recall=recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted)

recall
trainF1_score= 2 * (precision * recall) / (precision + recall)

trainF1_score
from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
X_test[['TotalVisits','Total Time Spent on Website',

        'Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits',

                                                                'Total Time Spent on Website','Page Views Per Visit']])



X_train.head()
X_test = X_test[col1]

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
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})
# Let's see the head of y_pred_final

y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.34 else 0)
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))

Lead_Score=y_pred_final.copy()

y_pred_final.head()
Lead_Score.reset_index(level=0, inplace=True)

Lead_Score.drop(['Converted', 'Converted_prob', 'final_predicted'], 1, inplace = True)

Lead_Score.head()
Lead=lead_dub.copy()

Lead.reset_index(level=0, inplace=True)

Lead.drop(['Lead Origin', 'Lead Source','Do Not Email', 'Do Not Call', 'Converted', 'TotalVisits',

       'Total Time Spent on Website', 'Page Views Per Visit', 'Last Activity',

       'Country', 'Specialization', 'How did you hear about X Education',

       'What is your current occupation',

       'What matters most to you in choosing a course', 'Search', 'Magazine',

       'Newspaper Article', 'X Education Forums', 'Newspaper',

       'Digital Advertisement', 'Through Recommendations',

       'Receive More Updates About Our Courses',

       'Update me on Supply Chain Content', 'Get updates on DM Content',

       'City', 'I agree to pay the amount through cheque',

       'A free copy of Mastering The Interview', 'Last Notable Activity'], 1, inplace = True)

Lead.head()
Lead_Score=pd.merge(Lead,Lead_Score,on='index')

Lead_Score.drop(['index'], 1, inplace = True)

Lead_Score.head()
Lead_Score.sort_values(["Lead_Score"], ascending = False,inplace=True)

Lead_Score.head()
# Let's check the overall accuracy.

testaccuracy= metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)

testaccuracy
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )

confusion2
# Let's see the sensitivity of our lmodel

testsensitivity=TP / float(TP+FN)

testsensitivity
# Let us calculate specificity

testspecificity= TN / float(TN+FP)

testspecificity
precision= precision_score(y_pred_final.Converted , y_pred_final.final_predicted)

precision
recall=recall_score(y_pred_final.Converted , y_pred_final.final_predicted)

recall
testF1_score= 2 * (precision * recall) / (precision + recall)

testF1_score
Lead_Score
# Let us compare the values obtained for Train & Test:

print("Train Data Accuracy    :{} %".format(round((trainaccuracy*100),2)))

print("Train Data Sensitivity :{} %".format(round((trainsensitivity*100),2)))

print("Train Data Specificity :{} %".format(round((trainspecificity*100),2)))

print("Train Data F1 Score    :{}  ".format(round((trainF1_score),2)))

print("Test Data Accuracy     :{} %".format(round((testaccuracy*100),2)))

print("Test Data Sensitivity  :{} %".format(round((testsensitivity*100),2)))

print("Test Data Specificity  :{} %".format(round((testspecificity*100),2)))

print("Test Data F1 Score     :{}  ".format(round((testF1_score),2)))
from sklearn.metrics import classification_report
print (classification_report(y_train_pred_final['Converted'], y_train_pred_final['final_predicted']))
print (classification_report(y_pred_final.Converted, y_pred_final.final_predicted))