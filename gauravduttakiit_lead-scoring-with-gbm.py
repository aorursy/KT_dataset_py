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
# Lead Quality: Indicates the quality of lead based on the data and intuition the the employee who has been assigned to the lead
lead['Lead Quality'].value_counts()

lead['Lead Quality'].describe()
plt.figure(figsize = (10,5))

ax= sns.countplot(lead['Lead Quality'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

plt.show()
# As Lead quality is based on the impression employee & the lead, 

#if anything is left blank we can impute 'Not Sure' in NaN safely.



lead['Lead Quality'] = lead['Lead Quality'].replace(np.nan, 'Not Sure')

plt.figure(figsize = (10,5))

ax= sns.countplot(lead['Lead Quality'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

plt.show()
# Asymmetrique Activity Index  |

# Asymmetrique Profile Index   \   An index and score assigned to each customer

# Asymmetrique Activity Score  |    based on their activity and their profile

# Asymmetrique Profile Score   \
fig, axs = plt.subplots(2,2, figsize = (10,9))

plt1 = sns.countplot(lead['Asymmetrique Activity Index'], ax = axs[0,0])

for p in plt1.patches:

    plt1.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt2 = sns.violinplot(lead['Asymmetrique Activity Score'], ax = axs[0,1])

plt3 = sns.countplot(lead['Asymmetrique Profile Index'], ax = axs[1,0])

for p in plt3.patches:

    plt3.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt4 = sns.violinplot(lead['Asymmetrique Profile Score'], ax = axs[1,1])

plt.tight_layout()
# There is too much variation in thes parameters so its not reliable to impute any value in it. 

# 45% null values means we need to drop these columns.
lead = lead.drop(['Asymmetrique Activity Index','Asymmetrique Activity Score',

                  'Asymmetrique Profile Index','Asymmetrique Profile Score'],1)
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
plt.figure(figsize = (10,5))

ax= sns.countplot(lead['Specialization'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

plt.show()
# It maybe the case that lead has not entered any specialization if his/her option is not availabe on the list,

#  may not have any specialization or is a student.

# Hence we can make a category "Others" for missing values. 

lead['Specialization'] = lead['Specialization'].replace(np.nan, 'Others')
plt.figure(figsize = (10,5))

ax= sns.countplot(lead['Specialization'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

plt.show()
round(100*(lead.isnull().sum()/len(lead.index)), 2)
# Tags
lead.Tags.describe()
lead.Tags.value_counts()
plt.figure(figsize = (10,5))

ax= sns.countplot(lead['Tags'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

plt.show()
# Blanks in the tag column may be imputed by 'Will revert after reading the email'.
lead['Tags'] = lead['Tags'].replace(np.nan, 'Will revert after reading the email')
plt.figure(figsize = (10,5))

ax= sns.countplot(lead['Tags'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

plt.show()
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
plt.figure(figsize = (10,5))

ax= sns.countplot(lead['Country'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
# Country is India for most values so let's impute the same in missing values.

lead['Country'] = lead['Country'].replace(np.nan, 'India')
plt.figure(figsize = (10,5))

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
plt.figure(figsize = (20,5))

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
lead.Tags.describe()
lead.Tags.value_counts()
plt.figure(figsize = (30,6))

ax= sns.countplot(x = "Tags", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
# Let's keep considerable last activities as such and club all others to "Other_Activity"

lead['Tags'] = lead['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',

                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',

                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',

                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',

                                    'University not recognized'], 'Other_Tags')
plt.figure(figsize = (20,6))

ax= sns.countplot(x = "Tags", hue = "Converted", data = lead)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

ax.set_yscale('log')

plt.show()
lead['Lead Quality'].describe()
lead['Lead Quality'].value_counts()
plt.figure(figsize = (10,6))

ax= sns.countplot(x = "Lead Quality", hue = "Converted", data = lead)

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
print("Original Columns {} % Retained".format(round((100* len(lead.columns)/len(lead_dub.columns)),2)))
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

                              'Tags','Lead Quality','City','Last Notable Activity']], drop_first=True)

dummy1.head()
# Adding the results to the master dataframe

lead = pd.concat([lead, dummy1], axis=1)

lead.head()
lead = lead.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization',

                  'What is your current occupation','Tags','Lead Quality','City','Last Notable Activity'], axis = 1)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=75)
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
X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])



X_train.head()
# Checking the Converted Rate

Converted = round((sum(lead['Converted'])/len(lead['Converted'].index))*100,2)

print("We have almost {} %  Converted rate after successful data manipulation".format(Converted))
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

model = GradientBoostingClassifier(n_estimators=150,max_depth=6)
# fit the model with the training data

model.fit(X_train,y_train)
# predict the target on the train dataset

predict_train = model.predict(X_train)

predict_train
trainaccuracy = accuracy_score(y_train,predict_train)

print('accuracy_score on train dataset : ', trainaccuracy)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.tail()
features_to_remove = vif.loc[vif['VIF'] >= 4.99,'Features'].values

features_to_remove = list(features_to_remove)

print(features_to_remove)
X_train = X_train.drop(columns=features_to_remove, axis = 1)

X_train.head()
X_test = X_test.drop(columns=features_to_remove, axis = 1)

X_test.head()
# fit the model with the training data

model.fit(X_train,y_train)
# predict the target on the train dataset

predict_train = model.predict(X_train)

predict_train
accuracytrain = accuracy_score(y_train,predict_train)

print('accuracy_score on train dataset : ', accuracytrain)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train, predict_train )

print(confusion)
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
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

draw_roc(y_train,predict_train)
#Using sklearn utilities for the same
from sklearn.metrics import precision_score, recall_score

precision_score(y_train,predict_train)
recall_score(y_train,predict_train)
# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data\n\n',predict_test)
confusion2 = metrics.confusion_matrix(y_test, predict_test )

print(confusion2)
# Let's check the overall accuracy.

testaccuracy= accuracy_score(y_test,predict_test)

testaccuracy
# Let's see the sensitivity of our lmodel

testsensitivity=TP / float(TP+FN)

testsensitivity
# Let us calculate specificity

testspecificity= TN / float(TN+FP)

testspecificity
# Let us compare the values obtained for Train & Test:

print("Train Data Accuracy    :{} %".format(round((trainaccuracy*100),2)))

print("Train Data Sensitivity :{} %".format(round((trainsensitivity*100),2)))

print("Train Data Specificity :{} %".format(round((trainspecificity*100),2)))

print("Test Data Accuracy     :{} %".format(round((testaccuracy*100),2)))

print("Test Data Sensitivity  :{} %".format(round((testsensitivity*100),2)))

print("Test Data Specificity  :{} %".format(round((testspecificity*100),2)))