## RESEARCH METHODS UNIVARIATE AND BIVARIATE ASSIGNMENTS KERNEL

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plots and graphs
import seaborn as sns # plots and graphs
import scipy
from scipy import stats # statistical analyses

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Step 1: Define the dataset as 'dfGSS'
dfGSS=pd.read_csv('/kaggle/input/2020sp-gss-data/2020SP_GSS_data.csv', na_values=['.', '96', '97', '98', '99'], dtype='float64')
# Let's take a look at what the data in our dataset looks like
print(dfGSS.head(10))
# Step 1: List column names
dfGSS.columns.values
# Step 2: Create a list with the names of the variables we are interested in getting descriptive statistics for. 
# Fill in VAR1, VAR2, VAR3, VAR4, & VAR5 with the variables you are interested in exploring for the worksheet. 
# Make sure the names you type below exactly match how the name is written in step 1's output.
lVar=['race', 'age', 'income', 'fear', 'grass', 'news', 'evidu']
# Step 3: Print frequency distribution. 
# Replace the label VARIABLE_LIST with the name of the list from step 2. 
for var in lVar:
    lFreq = dfGSS.loc[:,var].value_counts()
    lLen = dfGSS.loc[:,var].value_counts(normalize=True) * 100
    dfFreq = pd.concat([lFreq, lLen], axis=1)\
        .set_axis(['frequency', 'percent'], axis=1, inplace=False).sort_index()
    print(var, '\n', dfFreq, '\n')
# Step 4: Plot a histogram for numeric (interval and ratio level) variables.
# You do not need to run this line of code if none of your variables are interval or ratio level variables.
my_variable = 'age' # Replace VAR with the name of a categorical variable.
plt.hist(dfGSS[my_variable])
plt.xlabel(my_variable)
plt.ylabel('Frequency')
plt.title('Histogram of '+my_variable)
# Print a histogram for each of the numeric variables from the list lVar in step 2.
# Hint: 
# 1) Hover your cursor over this cell.
# 2) Click the "+ Code" button below the cell.
# 3) Copy and paste this code into the new cell:

my_variable = 'VAR' # Replace VAR with the name of a categorical variable.
plt.hist(dfGSS[my_variable])
plt.xlabel(my_variable)
plt.ylabel('Frequency')
plt.title('Histogram of '+my_variable)

# 4) Follow the directions to replace the necessary information.
# 5) Repeat until you have a cell and histogram output for each numeric variable in lVar.
my_variable = 'income' # Replace VAR with the name of a categorical variable.
plt.hist(dfGSS[my_variable])
plt.xlabel(my_variable)
plt.ylabel('Frequency')
plt.title('Histogram of '+my_variable)

# Code for bar chart labels
test={}
test['rank']={1:'Top',10:'Bottom'}
test['letdie1']={0:'No',1:'Yes'}
test['polhitok']={0:'No',1:'Yes'}
test['fear']={0:'No',1:'Yes'}
test['owngun']={0:'No', 1:'Yes'}
test['news']={1:'Everyday',2:'Few times a week',3:'Once a week',4:'Less than once wk',5:'Never'}
test['helpsick']={1:'Govt should help', 2:2, 3:'Agree with both', 4:4, 5:'People help selves'}
test['health1']={1:'Excellent',2:'Very good',3:'Good',4:'Fair',5:'Poor'}
test['concourt']={1:'Complete confidence',2:'A great deal of confidence',3:'Some confidence',4:'Very little confidence',5:'No confidence at all'}
test['evidu']={0:'No',1:'Yes'}
test['class_']={1:'Lower class',2:'Working class',3:'Middle class',4:'Upper class',5:'No class'}
test['socbar']={1:'Almost daily',2:'Sev times a week',3:'Sev times a mnth',4:'Once a month',5:'Sev times a year',6:'Once a year',7:'Never'}
test['conjudge']={1:'A great deal',2:'Only some',3:'Hardly any'}
test['age']={89:'89 or older'}
test['sex']={0:'Male',1:'Female'}
test['race']={1:'Black',2:'White',3:'Other'}
test['income']={1:'Lt $1000',2:'$1000 to 2999',3:'$3000 to 3999',4:'$4000 to 4999',5:'$5000 to 5999',6:'$6000 to 6999',7:'$7000 to 7999',8:'$8000 to 9999',9:'$10000 - 14999',10:'$15000 - 19999',11:'$20000 - 24999',12:'$25000 or more'}
test['region']={1:'New England',2:'Middle atlantic',3:'E. nor. central',4:'W. nor. central',5:'South atlantic',6:'E. sou. central',7:'W. sou. central',8:'Mountain',9:'Pacific'}
test['polviews']={1:'Extremely liberal',2:'Liberal',3:'Slightly liberal',4:'Moderate',5:'Slightly conservative',6:'Conservative',7:'Extremely conservative'}
test['natcrime']={1:'Too little',2:'About right',3:'Too much'}
test['natdrug']={1:'Too little',2:'About right',3:'Too much'}
test['natcrimy']={1:'Too little',2:'About right',3:'Too much'}
test['natdrugy']={1:'Too little',2:'About right',3:'Too much'}
test['cappun']={0:'Oppose',1:'Favor'}
test['gunlaw']={0:'Oppose',1:'Favor'}
test['courts']={1:'Too harsh',2:'About right',3:'Not harsh enough'}
test['grass']={0:'Not legal',1:'Legal'}
# Step 5: Plot bar charts for categorical (nominal, binary, ordinal) variables.
my_variable = 'fear' # Replace VAR with the name of a categorical variable.
count=dfGSS[my_variable].value_counts()
sns.barplot(count.index, count.values, alpha=0.9, color='steelblue')
plt.title('Bar chart of '+my_variable)
plt.xlabel(my_variable)
plt.ylabel('Frequency')
locs, labels = plt.xticks()
my_xticks = [test[my_variable][i] for i in [float(x.get_text()) for x in (labels)]]
plt.xticks(locs, my_xticks, rotation=90)
plt.show()
# Print a bar chart for each of the categorical variables (nominal, ordinal, or interval) from the list lVar in step 2.
# Hint: 
# 1) Hover your cursor over this cell.
# 2) Click the "+ Code" button below the cell.
# 3) Copy and paste this code into the new cell:

my_variable = 'VAR' # Replace VAR with the name of a categorical variable.
count=dfGSS[my_variable].value_counts()
sns.barplot(count.index, count.values, alpha=0.9, color='steelblue')
plt.title('Bar chart of '+my_variable)
plt.xlabel(my_variable)
plt.ylabel('Frequency')
locs, labels = plt.xticks()
my_xticks = [test[my_variable][i] for i in [float(x.get_text()) for x in (labels)]]
plt.xticks(locs, my_xticks, rotation=90)
plt.show()

# 4) Follow the directions to replace the necessary information.
# 5) Repeat until you have a cell and bar chart output for each categorical variable in lVar.
my_variable = 'grass' # Replace VAR with the name of a categorical variable.
count=dfGSS[my_variable].value_counts()
sns.barplot(count.index, count.values, alpha=0.9, color='steelblue')
plt.title('Bar chart of '+my_variable)
plt.xlabel(my_variable)
plt.ylabel('Frequency')
locs, labels = plt.xticks()
my_xticks = [test[my_variable][i] for i in [float(x.get_text()) for x in (labels)]]
plt.xticks(locs, my_xticks, rotation=90)
plt.show()
my_variable = 'news' # Replace VAR with the name of a categorical variable.
count=dfGSS[my_variable].value_counts()
sns.barplot(count.index, count.values, alpha=0.9, color='steelblue')
plt.title('Bar chart of '+my_variable)
plt.xlabel(my_variable)
plt.ylabel('Frequency')
locs, labels = plt.xticks()
my_xticks = [test[my_variable][i] for i in [float(x.get_text()) for x in (labels)]]
plt.xticks(locs, my_xticks, rotation=90)
plt.show()
my_variable = 'evidu' # Replace VAR with the name of a categorical variable.
count=dfGSS[my_variable].value_counts()
sns.barplot(count.index, count.values, alpha=0.9, color='steelblue')
plt.title('Bar chart of '+my_variable)
plt.xlabel(my_variable)
plt.ylabel('Frequency')
locs, labels = plt.xticks()
my_xticks = [test[my_variable][i] for i in [float(x.get_text()) for x in (labels)]]
plt.xticks(locs, my_xticks, rotation=90)
plt.show()
# Step 6: Print descriptive statistics
# Replace the label VARIABLE_LIST with the name of the list from step 2.
dfUni= pd.DataFrame(index=('count', 'mean', 'median', 'mode', 'stdev', 'min', '25%', '50%', '75%', 'max'))
place=0

for var in lVar:
    lCount = dfGSS.loc[:,var].count()
    lMean = dfGSS.loc[:,var].mean()
    lMedian = dfGSS.loc[:,var].median()
    lMode = dfGSS.loc[:,var].mode().values[0]
    lStd = dfGSS.loc[:,var].std()
    lMin = dfGSS.loc[:,var].min()
    a, b, c = dfGSS.loc[:,var].quantile([.25, .5, .75])
    lMax = dfGSS.loc[:,var].max()
    s = (lCount, lMean, lMedian, lMode, lStd, lMin, a, b, c, lMax)
    dfUni.insert(place, var, s)
    place+1

print(dfUni)
# Step 7: Run a crosstab
# Replace INDEPENDENT_VARIABLE and DEPENDENT_VARIABLE with the names of 
# your independent and dependent variables from the worksheet.
contingency_table = pd.crosstab(
    dfGSS.loc[:,'INDEPENDENT_VARIABLE'],
    dfGSS.loc[:,'DEPENDENT_VARIABLE'],
    margins = True
)
contingency_table
# Step 8: Calculate chi-square. 
# You do not have to change anything in this code.
chi2, p, dof, ex = stats.chi2_contingency(contingency_table, correction=False)
stats.chi2_contingency(contingency_table)

# Converting the p-value from scientific format to a decimal
def ConvertToFloat(inputNumber):
    return "{0:.4f}".format(inputNumber)

# Print the chi-square and p-value
print("Chi-squared:", chi2, "p-value:", ConvertToFloat(p))
# Step 9: Run a correlation with p-value between two variables 
# Replace INDEPENDENT_VARIABLE and DEPENDENT_VARIABLE with the names of 
# your independent and dependent variables from the worksheet.
varA=dfGSS.loc[:,'INDEPENDENT_VARIABLE']
varB=dfGSS.loc[:,'DEPENDENT_VARIABLE']

# Hiding NaN values from arrays
bad = ~np.logical_or(np.isnan(varA), np.isnan(varB))
x=np.asarray(varA).compress(bad)
y=np.asarray(varB).compress(bad)

# Correlation statistics
r,p =stats.pearsonr(x,y)

# Print Pearson's r and p-value
print("r:", r, "p-value:", ConvertToFloat(p))