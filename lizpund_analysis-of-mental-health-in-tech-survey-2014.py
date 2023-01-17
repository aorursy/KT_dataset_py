# Import packages
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis, show

# Display plots inline in the notebook
%matplotlib inline 

# Ignore warning related to pandas_profiling (https://github.com/pandas-profiling/pandas-profiling/issues/68)
import warnings
warnings.filterwarnings('ignore') 

# Display all dataframe columns in outputs (it has 27 columns, which is wider than the notebook)
# This sets it up to dispaly with a horizontal scroll instead of hiding the middle columns
pd.set_option('display.max_columns', 100) 

# Load in the dataset as mh
mh = pd.read_csv("../input/survey.csv")
# Display the data type for each variable
mh.dtypes
# Display the first five rows of the data set
mh.head()
# Display a random 10 rows from the data set
mh.sample(10)
# Get a quick overview of all of the variables using pandas_profiling
import pandas_profiling
pandas_profiling.ProfileReport(mh)
# Explore the numeric variable Age to confirm whether all values are within a reasonable range and if any are NaN.
print("'Age'")
print("Minimum value: ", mh["Age"].min())
print("Maximum value: ", mh["Age"].max())
print("How many values are NaN?: ", pd.isnull(mh['Age']).sum())
# Learn more about the variable Gender, which appears not to be standardized with 49 distinct responses.

# Count the number of distinct responses and list them:
print("Count of distinct responses for Gender:", len(set(mh['Gender'])))
print("Distinct responses for Gender:", set(mh['Gender']))
mh.columns = map(str.lower, mh.columns)

# Confirm that all variable names are now lower case
mh.dtypes
# Convert "timestamp" data type from object to datetime
mh['timestamp'] = pd.to_datetime(mh['timestamp'])

# Confirm that it worked
mh.dtypes
# Create a new column "age_clean" that replaces out-of-range ages with "NaN"
# The oldest living person on record lived to be 122 years, 164 days (Jeanne Calment, 1875-1997)
def clean_age(age):
    if age>=0 and age<=123:
        return age
    else:
        return np.nan
mh['age_clean'] = mh['age'].apply(clean_age)

# Check out the new column and make sure it looks right

print("'Age'")
print("Minimum value: ", mh["age_clean"].min())
print("Maximum value: ", mh["age_clean"].max())
print("How many values are NaN?: ", pd.isnull(mh['age_clean']).sum())
print("Frequency table for age_clean:")
mh["age_clean"].value_counts().sort_index(0)
# Plot a histogram of the respondents' ages (remove any NaN values)

sns.set(color_codes=True)
sns.set_palette(sns.color_palette("muted"))

sns.distplot(mh["age_clean"].dropna());
# Recategorize each response into these categories: Male (cis), Male (trans), Female (cis), Female (trans), Other

# Standardize the capitalization in the responses by making all characters lower case
mh['gender'] = mh['gender'].str.lower()

# Make a copy of the column to preserve the original data. I will work with the new column going forward.
mh['gender_new'] = mh['gender']

# Assign each type of response to one of the five categories

male = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr","cis man", "cis male"]
trans_male = [None]
trans_female = ["trans-female",  "trans woman", "female (trans)"]
female = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]
other = ["non-binary", "nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "neuter", "queer", "ostensibly male, unsure what that really means", "queer/she/they", "something kinda male?"]

mh['gender_new'] = mh['gender_new'].apply(lambda x:"Male (Cis)" if x in male else x)
mh['gender_new'] = mh['gender_new'].apply(lambda x:"Female (Cis)" if x in female else x)
mh['gender_new'] = mh['gender_new'].apply(lambda x:"Female (Trans)" if x in trans_female else x)
mh['gender_new'] = mh['gender_new'].apply(lambda x:"Male (Trans)" if x in trans_male else x)
mh['gender_new'] = mh['gender_new'].apply(lambda x:"Other" if x in other else x)
mh.drop(mh[mh.gender_new == 'p'].index, inplace=True)
mh.drop(mh[mh.gender_new == 'a little about you'].index, inplace=True)

# Make a crosstab to view the count for each of the new categories
print("Gender:")
print("How many values are NaN?: ", pd.isnull(mh['gender_new']).sum())
print("")
print("Frequency table for gender_new:\n", mh["gender_new"].value_counts().sort_index(0))
print("")

# Confirm that no entries were missed in the sorting above: display the size of the old and new variables, and of the entire dataset
print("If we didn't miss any entries, these numbers will be the same:")
print("gender =", len(mh['gender']), "values")
print("gender_new =", len(mh['gender_new']), "values")
print("Dataset Entries =", len(mh), "values")
# Create a bar chart comparing gender

mh['gender_new'].value_counts().plot(kind='bar')
# Create a new dataframe with the two columns and assign numbers in place of their categories

df = pd.DataFrame({'treatment': mh['treatment'], 'work_interfere': mh['work_interfere']}, dtype='category')
df_num = df.apply(lambda x: x.cat.codes)

# Run a correlation calculation 
print("Pearson:", df_num.corr())
print("")
print("Spearman:", df_num.corr('spearman'))
print("")
print("Kendall:", df_num.corr('kendall'))
plt.figure(figsize=(10,7))
sns.countplot(x="benefits", hue="treatment", hue_order = ["Yes", "No"], data=mh)
plt.title("Does your employer provide mental health benefits?",fontsize=16)
plt.suptitle("Seeking Treatment v. Work Benefits", fontsize=20)
plt.xlabel("")
plt.show()
 # Generate a chart comparing mental health benefits and treatment
    
plt.figure(figsize=(10,7))
sns.countplot(x="treatment", hue="benefits", hue_order = ["Yes", "No", "Don't know"], data=mh)
plt.suptitle("Seeking Treatment v. Work Benefits  (Inverted)", fontsize=20)
plt.title("Have you sought treatment for a mental health condition?",fontsize=16)
plt.xlabel("")
plt.ylabel("")
plt.show()
plt.figure(figsize=(10,7))
sns.countplot(x="family_history", hue="treatment", hue_order = ["Yes", "No"], data=mh)
plt.suptitle("Family History v. Seeking Treatment", fontsize=20)
plt.title("Do you have a family history of mental illness?", fontsize=16)
plt.xlabel("")
plt.ylabel("")
plt.show()
# Generate crosstabs of "family history" and "treatment" (the "observed" values)

import scipy.stats as stats

tab_famhist_treatment = pd.crosstab(mh["family_history"], mh["treatment"], margins = True)
tab_famhist_treatment.columns = ["Treatment Yes","Treatment No","row_totals"]
tab_famhist_treatment.index = ["Fam Hist Yes","Fam Hist No","col_totals"]

observed = tab_famhist_treatment.iloc[0:2,0:2]   # Get table without totals for later use

tab_famhist_treatment
# Generate the "expected" values to compare against our "observed" values

expected =  np.outer(tab_famhist_treatment["row_totals"][0:2],
                     tab_famhist_treatment.loc["col_totals"][0:2]) / 1257

expected = pd.DataFrame(expected)

expected.columns = ["Treatment Yes","Treatment No"]
expected.index = ["Fam Hist Yes","Fam Hist No"]

expected
# Run the Chi-Squared test

chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()
print(chi_squared_stat)

# Note: We call .sum() twice: once to get the column sums and a second time to 
# add the column sums together, returning the sum of the entire 2D table.
crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 1)   # *

# *Note: The degrees of freedom for a test of independence equals the product of 
# the number of categories in each variable minus 1. In this case we have a 2x2 table 
# so df = 1x1 = 1.

print("Critical value")
print(crit)

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=1)
print("P value")
print(p_value)
# Display the distinct countries represented in the data (quantity and names)

print("Country Count =", len(set(mh['country'])))
print("Country Names =", set(mh['country']))
# Display quantity and names of distinct countries represented in the data (quantity and names)

print("State Count =", len(set(mh['state'])))
print("State Names =", set(mh['state']))
print(mh['state'].describe())
# Create a frequency chart for "country"

plt.figure(figsize=(10, 7))
sns.countplot(y='country', order = mh['country'].value_counts().index, data=mh)
plt.title('Survey Responses by Country', fontsize=20)
plt.xlabel('')
plt.ylabel('')
plt.show()
#### Survey Responses by state

total = float(len(mh))
plt.figure(figsize=(20, 7))
ax = sns.countplot(x='state', order = mh['state'].value_counts().index, data=mh)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
plt.title('Responses by State', fontsize=20)
plt.xlabel('')
plt.ylabel('')
plt.show()
# Define how to recategorize each state into one of the US Census Bureau regions: West, Midwest, South, Northeast

# Mke a copy of the column to preserve the original data. Work with the new column going forward.
mh['region'] = mh['state']

# Then, in the new column, assign each type of response to one of the new categories

west = ["WA", "OR", "CA", "NV", "ID", "MT", "WY", "UT", "AZ", "NM", "CO"]
midwest = ["ND", "SD", "NE", "KS", "MN", "IA", "MO", "WI", "IL", "IN", "OH", "MI"]
northeast = ["ME",  "NH", "VT", "MA", "CT", "RI", "NY", "PA", "NJ"]
south = ["MD", "DE", "DC", "WV",  "VA", "NC","SC", "GA", "FL", "KY", "TN", "AL", "MS", "AR", "LA", "OK", "TX"]

mh['region'] = mh['region'].apply(lambda x:"West" if x in west else x)
mh['region'] = mh['region'].apply(lambda x:"Midwest" if x in midwest else x)
mh['region'] = mh['region'].apply(lambda x:"Northeast" if x in northeast else x)
mh['region'] = mh['region'].apply(lambda x:"South" if x in south else x)

# Make a crosstab to view the count for each of the new categories
region_tab = pd.crosstab(index=mh["region"], columns="count")

print(region_tab)

# Confirm that we didn't miss any entries
print("If we didn't miss anything, this will equal 1257:", len(mh['region']))

region_tab.plot(kind="bar", 
                 figsize=(20,7),
                 stacked=True)
#### Survey Responses by region

total = float(len(mh))
plt.figure(figsize=(20, 7))
ax = sns.countplot(x='region', order = mh['region'].value_counts().index, data=mh)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 
plt.title('Responses by Region', fontsize=20)
plt.xlabel('')
plt.ylabel('')
plt.show()
plt.figure(figsize=(10,7))
sns.countplot(x="region", hue="work_interfere", hue_order = ["Never", "Rarely", "Sometimes", "Often"], data=mh)
plt.suptitle("Work Interfere v. Region (U.S.)", fontsize=20)
plt.title("Frequency of mental health conditions in the U.S. by region", fontsize=16)
plt.xlabel("")
plt.ylabel("")
plt.show()
mh['work_interfere']
# Convert the work_interfere responses into a new variable, 

# Make a copy of the column to preserve the original data. Work with the new column going forward.
mh['ill'] = mh['work_interfere']

# Transform all NaN to "No" (which means, not currently experiencing a mental health condition)
mh['ill'] = mh['ill'].replace(np.nan, 'No', regex=True)

# Assign each type of response to one of two categories

notill = ["No"]
ill = ["Never", "Rarely", "Sometimes", "Often"]

mh['ill'] = mh['ill'].apply(lambda x:"Yes" if x in ill else x)
mh['ill'] = mh['ill'].apply(lambda x:"No" if x in notill else x)

# Make a crosstab to view the count for each of the new categories
ill_tab = pd.crosstab(index=mh["ill"], columns="count")

print(ill_tab)

# Confirm that we didn't miss any entries
print("If we didn't miss anything, this will equal 1257:", len(mh['ill']))

ill_tab.plot(kind="bar", 
                 figsize=(20,7),
                 stacked=True)
# Display the relationship between "ill" and "region"

plt.figure(figsize=(10,7))
sns.countplot(x="region", hue="ill", hue_order = ["Yes", "No"], data=mh)
plt.suptitle("Mental Health Conditions v. Region (U.S.)", fontsize=20)
plt.title("Frequency of mental health conditions in the U.S. by region", fontsize=16)
plt.xlabel("")
plt.ylabel("")
plt.show()
# Convert the mental_health_consequence responses into a new variable, 

# Make a copy of the column to preserve the original data. Work with the new column going forward.
mh['attitudes'] = mh['mental_health_consequence']

# Assign each type of response to one of two categories
positive = ["No"]
negative = ["Yes"]
moderate = ['Maybe']

mh['attitudes'] = mh['attitudes'].apply(lambda x:"Positive" if x in positive else x)
mh['attitudes'] = mh['attitudes'].apply(lambda x:"Negative" if x in negative else x)
mh['attitudes'] = mh['attitudes'].apply(lambda x:"Moderate" if x in moderate else x)


# Make a crosstab to view the count for each of the new categories
attitudes_tab = pd.crosstab(index=mh["attitudes"], columns="count")

print(attitudes_tab)

# Confirm that we didn't miss any entries
print("If we didn't miss anything, this will equal 1257:", len(mh['attitudes']))

print(attitudes_tab.plot(kind="bar", 
                 figsize=(20,7),
                 stacked=True))


# Display the relationship between "mental_health_consequence" and "region"

plt.figure(figsize=(10,7))
sns.countplot(x="region", hue="attitudes", hue_order = ["Positive", "Moderate", "Negative"], data=mh)
plt.suptitle("Mental Health Attitudes v. Region (U.S.)", fontsize=20)
plt.title("Attitudes towards mental health in the U.S. by region", fontsize=16)
plt.xlabel("")
plt.ylabel("")
print(plt.show())