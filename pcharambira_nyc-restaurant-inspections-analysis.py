# Let us import what we need to the analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import datetime 
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
data = pd.read_csv('../input/DOHMH_New_York_City_Restaurant_Inspection_Results.csv')
# Lets us look at a few records
data.head()
# Get the coulmns in the dataset
data.columns
# checking missing data in data 
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()
data['SCORE'].describe()
# Plot a histogram
data.SCORE.hist(figsize=(10,4))
plt.title("Boxplot for the Scores", fontsize=15)
plt.xlabel('Score', fontsize = 12)
# Have a look at a distribution plot of the Score
fig, ax = plt.subplots()
fig.set_size_inches(15, 4)
sns.distplot(data.SCORE.dropna())
plt.title("Distribution Plot of the Scores", fontsize=15)
# Let us look at a violin plot of the scores
fig, ax = plt.subplots()
fig.set_size_inches(15, 4)
sns.violinplot(data.SCORE.dropna())
plt.title("Violin plot of the Scores", fontsize=15)
data.GRADE.value_counts()
# A look at the histogram of the Grades.
data.GRADE.hist(figsize = (15, 4))
plt.title("Histogram of the Grades", fontsize=15)
plt.xlabel('Grades', fontsize = 12)
# Lets look at scores by grades
fig, ax = plt.subplots()
fig.set_size_inches(15, 4)
sns.boxplot(data.SCORE.dropna(), data.GRADE)
plt.title('Boxplot by Grade', fontsize = 15)
# Look at whih Boroughs have the highest number of inspections
data.BORO.value_counts()
# Here is a look at a histogram of the numbers we just saw above.
data.BORO.hist(figsize = (15, 4))
plt.title('Boxplot of the count of inspections per Borough', fontsize = 15)
plt.xlabel('Borough', fontsize = 12)
# Breakdown scores by borough
fig, ax = plt.subplots()
fig.set_size_inches(15, 4)
sns.boxplot(data.SCORE.dropna(), data.BORO)
plt.title('Boxplot by Borough', fontsize = 15)
# Contingency table for Grade and Borough
boro_grade = pd.crosstab(data.GRADE, data.BORO, margins = True)
boro_grade
# Plot of grade by borough
pd.crosstab(data.BORO, data.GRADE).plot(kind="bar", figsize=(15,8), stacked=True)
plt.title('Grade Distribution by Borough', fontsize = 15)
# Test if the grades are independent of the borough
boro_grade.columns = ["BRONX","BROOKLYN","MANHATTAN", "QUEENS", "STATEN ISLAND" ,"All"]

boro_grade.index = ["A","B","C","Not Yet Graded","P", "Z", "All"]

observed = boro_grade.ix[0:6,0:5]   # Get table without totals for later use

expected =  np.outer(boro_grade["All"][0:6],
                     boro_grade.ix["All"][0:5]) / 1000

expected = pd.DataFrame(expected)

expected.columns = ["BRONX","BROOKLYN","MANHATTAN", "QUEENS", "STATEN ISLAND"]
expected.index = ["A","B","C","Not Yet Graded","P", "Z"]

chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()

print("Chi Squared Stat")
print(chi_squared_stat)

crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 20)   # (5-1) * (6-1)

print("Critical value")
print(crit)

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=20)
print("P value")
print(p_value)

stats.chi2_contingency(observed= observed)
data['CUISINE DESCRIPTION'].value_counts()
# Let us look at the scores by cuisine
score_cuisine = pd.concat([data['CUISINE DESCRIPTION'], data['SCORE']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x = 'CUISINE DESCRIPTION', y="SCORE", data = score_cuisine)
plt.xticks(rotation=90);
data.ACTION.value_counts()
# Histogram of the Action taken
data.ACTION.hist(figsize = (15,8))
plt.title('Histogram of the Action taken', fontsize = 15)
plt.xlabel('Action', fontsize = 12)
plt.xticks(rotation=90)
data['CRITICAL FLAG'].value_counts()
# Graphical representation of the critical flag
data['CRITICAL FLAG'].hist(figsize=(15,4))
plt.title('Histogram of the Critical Flag', fontsize = 15)
plt.xlabel('Flag', fontsize = 12)
# Critical Flag by Borough
pd.crosstab(data.BORO, data['CRITICAL FLAG']).plot(kind="bar", figsize=(15,8), stacked=True)
plt.title('Critical Flag by Borough', fontsize = 15)
# Critical Flag by Cuisine
pd.crosstab(data['CUISINE DESCRIPTION'], data['CRITICAL FLAG']).plot(kind="bar", figsize=(18,18), stacked=True)
plt.title('Critical Flag by Cuisine', fontsize = 15)
# Let us look at the scores by critical flag
score_flag = pd.concat([data['CRITICAL FLAG'], data['SCORE']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x = 'CRITICAL FLAG', y="SCORE", data = score_flag)
plt.title('Score by Critical Flag', fontsize = 15)
plt.xticks(rotation=90);
data['INSPECTION TYPE'].value_counts()
import re
from nltk.corpus import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()# lowercase text  
    text = REPLACE_BY_SPACE_RE.sub(' ',text)# replace REPLACE_BY_SPACE_RE symbols by space in text    
    text = BAD_SYMBOLS_RE.sub('',text)# delete symbols which are in BAD_SYMBOLS_RE from text    
    temp = [s.strip() for s in text.split() if s not in STOPWORDS]# delete stopwords from text
    new_text = ''
    for i in temp:
        new_text +=i+' '
    text = new_text
    return text.strip()
# Let us create a word cloud for the violation description
temp_data = data.dropna(subset=['VIOLATION DESCRIPTION'])
# converting into lowercase
temp_data['VIOLATION DESCRIPTION'] = temp_data['VIOLATION DESCRIPTION'].apply(lambda x: " ".join(x.lower() for x in x.split()))
temp_data['VIOLATION DESCRIPTION'] = temp_data['VIOLATION DESCRIPTION'].map(text_prepare)


from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(temp_data['VIOLATION DESCRIPTION'].values))
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.title("Top Words Used for the Violation Descriptions", fontsize=25)
plt.axis("off")
plt.show() 