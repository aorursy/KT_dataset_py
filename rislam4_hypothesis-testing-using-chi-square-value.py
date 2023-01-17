# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
# Importing dataset
filename = '/kaggle/input/jeopardy/jeopardy.csv'
jeopardy = pd.read_csv(filename)

pd.set_option('display.max_colwidth', 1000)
jeopardy.head()
jeopardy.shape
jeopardy.info()
jeopardy.columns
# Fixing the white space before column name.
jeopardy.columns = ['Show Number', 'Air Date', 'Round', 'Category', 'Value', 'Question', 'Answer']
# Value columns
pd.set_option('display.max_rows', 100)
jeopardy.Value.value_counts()
# Function to remove dollar sign and 'None' value
def convert(x):
    import re
    if x == 'None':
        return 0
    else:
        x = re.sub('[^\d]', '', x)         # '^\d' means except digit remove all
        x = int(x)
        return x

jeopardy['Value'] = jeopardy.Value.apply(convert)
# Changing the 'Air Date' data type to datetime
jeopardy['Air Date'] = pd.to_datetime(jeopardy['Air Date'])   # chanigng to date time format
jeopardy.head()
# Function to remove characters except digit and letters
def normalize_test(x):
    import re
    x = x.lower()
    x = re.sub('[^\w\s]', '', x)    # [^\w\s] means 
    return x

# Cleaning the 'Question' column and inserting next to it.
cleaned_question = jeopardy.Question.apply(normalize_test)
jeopardy.insert(6, 'Cleaned Question', cleaned_question)

# Cleaning the answer column and inserting next to it.
cleaned_answer = jeopardy.Answer.apply(normalize_test)
jeopardy.insert(8, 'Cleaned Answer', cleaned_answer)
jeopardy.head(2)
jeopardy.info()
# Writing a function to count matched words.
def ans_in_question(x, y):
    ans_in_ques = []
    for q, a in zip(x, y):
        c = 0
        split_answer = a.split()
        split_question = q.split()
        
        # Removing the below common words from answer
        unused = ['a', 'an', 'the']
        for item in unused:
            if item in split_answer:
                split_answer.remove(item)
                
        # Counting the matched words
        for item in split_answer:
            if item in split_question:
                c += 1
            else:
                c = c
        # Proportion of matched words
        if len(split_answer) == 0:
            ans_in_ques.append(0)
        else:
            ans = c / len(split_answer)
            ans_in_ques.append(ans)
    
    return ans_in_ques

# Creating the matched word count column
jeopardy['Ans in Ques'] = ans_in_question(jeopardy['Cleaned Question'], jeopardy['Cleaned Answer'])
jeopardy.head(2)
jeopardy['Ans in Ques'].nunique()
# available plot style
plt.style.available
plt.figure(figsize= (5,3), dpi= 95)
plt.style.use('fivethirtyeight')

jeopardy['Ans in Ques'].plot.kde(title= 'Distribution', linewidth= 1.5)

plt.show()
# Mean of the 'Ans in Ques'
jeopardy['Ans in Ques'].mean()
jeopardy.head(2)
# Taking a list to store unique question.
question_overlap = []
terms_used = set()

# Comparing question with the previous one.
for item in jeopardy['Cleaned Question']:
    
    split_question = item.split()
    split_question = [q for q in split_question if len(q) > 4]
    c = 0
    for word in split_question:

        if word in terms_used:
            c += 1
        terms_used.add(word)

    if len(split_question) == 0:
        question_overlap.append(0)
    else:
        ans = c / len(split_question)
        question_overlap.append(ans)
# Inserting the new column next to 'Cleaned Question'
jeopardy.insert(7, 'Question Overlap', question_overlap)
jeopardy.head(2)
plt.figure(figsize= (5,3), dpi= 95)
plt.style.use('fivethirtyeight')

jeopardy['Question Overlap'].plot.kde(title= 'Distribution', linewidth= 1.5)

plt.show()
jeopardy['Question Overlap'].mean()
# Function to determine high low value
def high_low_value(value):
    if value >= 800:
        return 1
    else:
        return 0

# Creating a new column
jeopardy["High Value"] = jeopardy['Value'].apply(high_low_value)
jeopardy.tail(2)
# Proportion of high value
jeopardy['High Value'].value_counts(normalize= True) * 100
# Function to determine the count of a word in 'High Value' and 'Low Value'
def count_word_in_high_low(word):
    high_count = 0
    low_count = 0
    for i, row in jeopardy.iterrows():
        split_question = row['Cleaned Question'].split()
        if word in split_question: 
            if row['High Value'] == 1:
                high_count += 1
            else:
                low_count += 1
    return high_count, low_count
# We obtained this set of all unique words earlier
terms_used = terms_used
import random
terms_used_list = list(terms_used)

comparison_terms = []
for i in range(15):
    r = random.choice(terms_used_list)   # This function pick a word randomly
    comparison_terms.append(r)

print('Randomly Picked Words: ', comparison_terms, end= '\n\n')

# Observed set
observed_set = []

for item in comparison_terms:
    obs = count_word_in_high_low(item) 
    observed_set.append(obs)
    
print('Observed Set: ', observed_set)
jeopardy['High Value'].value_counts()
# Actual count of high and low value
high_value_total_count = jeopardy['High Value'].value_counts()[1]
low_value_total_count = jeopardy['High Value'].value_counts()[0]
# Importing the function to determine chi-square
from scipy.stats import chisquare

for item in observed_set:
    add = sum(item)
    
    # Calculation to get expected value
    high = (add / len(jeopardy)) * high_value_total_count
    low = (add / len(jeopardy)) * low_value_total_count
    
    # Assigning observed and expected set
    observed = list(item)
    expected = [high, low]
    
    chi_square, p_value = chisquare(observed, expected)
    print('Chi-square: ',chi_square, end= '\t\t\t')
    print('P_value: ', p_value)
jeopardy.head(3)
# Unique category
jeopardy['Category'].nunique()
jeopardy.Category.value_counts().sort_values(ascending= False)
# Taking the values as a list
value = list(jeopardy.Category.value_counts().sort_values(ascending= False))

# finding the categories which are more than 29.
for i, item in enumerate(value) :
    head = i
    if item < 30:
        break;
print(f'First {head} values are above or equal to 30.')

# PLotting the category
plt.figure(figsize=(10,4), dpi= 95)
jeopardy.Category.value_counts().sort_values(ascending= False).head(head).plot.bar(title= 'Top 10 Categories')

plt.show()
