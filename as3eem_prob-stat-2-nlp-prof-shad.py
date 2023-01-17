# Mandatory Libraries only
import pandas as pd # to read data: CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/australian-election-2019-tweets/auspol2019.csv')
data.head()
# we need only tweets so let's fetch 'full_text' feature
tweets = data['full_text'][:5000]

# analytics
print("length of tweets: ",len(tweets))
print("Sample Tweet One: \n",tweets[464])
print("-----------------------------")
print("Sample Tweet Two: \n",tweets[4354])
# TOKENISE
# since libraries are not allowed I am not using nltk or regex
# without library it could be done using split function

for _ in range(len(tweets)):
    tweets[_] = tweets[_].lower().strip().split()
    
print("Sample Tokenised Tweet One: \n",tweets[464])
print("-----------------------------")
print("Sample Tokenised Tweet Two: \n",tweets[4354])
# F1: word count:
def word_count_F1(token):
    return len(token)
# negation words: words conataining any word related to not | https://www.grammarly.com/blog/negatives/
negation_words = ["No","Not","None","No one","Neither","Doesn’t","Isn’t","Wasn’t","Shouldn’t","Wouldn’t","Couldn’t","Won’t","Can’t","Don’t"]
# since REGEX library usage is not allowed I have done this with O(n) complexity with looping over all tokens formed otherwise with regex it could have been done more precisely

# F2: 
# a. question mark (Yes or No)
# b. exclamation mark (Yes or No)
# c. period - two or more consecutive dots (Yes or No)
# d. URL (Yes or No)
# e. negation words (Yes or No)

def check_F2(token):
    question_m = 0
    exclamation_m = 0
    period = 0
    url=0
    negation = 0
    
    for tok in token:
        if not question_m and '?' in tok:
            question_m+=1
        if not exclamation_m and '!' in tok:
            exclamation_m+=1
        if not period and '...' in tok:
            period+=1
        if not url and ('http' in tok or '.com' in tok):
            url+=1
        if not negation and (tok in negation_words):
            negation+=1
            
    return question_m, exclamation_m, period, url, negation
# Checking for special characters 
# F3. Special Symbol count. (Integer)

def special_char_F3(token):
    special_char_count = 0
    
    for tok in token:
        for char in tok:
            if not (char.isalpha() or char.isdigit()):
                special_char_count+=1   
    return special_char_count

# dataframe to organise answers to problem statement

df = pd.DataFrame(columns=['tweet', 'F1', 'F2a','F2b','F2c','F2d','F2e','F3'])
df.head()
# OUTPUT for all tweets:

for _ in range(len(tweets)):
    l = [data['full_text'][_]]
    
    l.append(word_count_F1(tweets[_]))
    
    qm,em,per,url,neg = check_F2(tweets[_])
    l.extend((bool(qm),bool(em),bool(per),bool(url),bool(neg)))
    
    l.append(special_char_F3(tweets[_]))
    
    df = df.append({'tweet':l[0],'F1':l[1],'F2a':l[2],'F2b':l[3],'F2c':l[4],'F2d':l[5],'F2e':l[6],'F3':l[7]}, ignore_index=True)
df.head()
