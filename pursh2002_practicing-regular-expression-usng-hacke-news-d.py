# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd 
hn = pd.read_csv("/kaggle/input/hacker-news-posts/HN_posts_year_to_Sep_26_2016.csv")

hn.info()
hn.head()
import re

m = re.search("and", "hand")
print(m)
m = re.search("and", "antidote")
print(m)
string_list = ["Julie's favorite color is Blue.",
               "Keli's favorite color is Green.",
               "Craig's favorite colors are blue and red."]

pattern = "Blue"

for s in string_list:
    if re.search(pattern,s):
        print("Match")
    else:
        print("No Match")

blue_mentions = 0
pattern = "Blue"

for s in string_list:
    if re.search(pattern, s):
        blue_mentions += 1

print(blue_mentions)
blue_mentions = 0
pattern = "[Bb]lue"

for s in string_list:
    if re.search(pattern, s):
        blue_mentions += 1

print(blue_mentions)
import re 
titles = hn["title"].tolist()
python_mentions = 0 
pattern = "[Pp]ython"

for s in titles:
    if re.search(pattern,s):
        python_mentions += 1
print(python_mentions)


eg_list = ["Julie's favorite color is green.",
           "Keli's favorite color is Blue.",
           "Craig's favorite colors are blue and red."]

eg_series = pd.Series(eg_list)

print(eg_series)
pattern = "[Bb]lue"
pattern_contained = eg_series.str.contains(pattern)
print(pattern_contained)

pattern_count = pattern_contained.sum()
print(pattern_count)
pattern_count = eg_series.str.contains(pattern).sum()
print(pattern_count)

patterns = "[Pp]ython"
titles = hn["title"]
python_mentions = titles.str.contains(patterns)
python_mentions.sum()
python_mentions.head()
py_titles = titles[python_mentions]
print(py_titles.head())
py_titles = titles[titles.str.contains("[Pp]ython")]
print(py_titles.head())
titles = hn["title"] 
ruby_titles = titles.str.contains("[Rr]uby")
ruby_titles = titles[ruby_titles]
ruby_titles
email_bool = titles.str.contains("e-?mail")
email_count = email_bool.sum()
email_count
email_titles = titles[email_bool]
email_titles
pattern = "\[\w+\]"
tag_titles = titles[titles.str.contains(pattern)]
tag_count = tag_titles.shape[0]
tag_count
tag_5 = tag_titles.head()
print(tag_5)
tag_titles
print('hello\b world')
print('hello\\b world')
print(r'hello\b world')
pattern = r"(\[\w+\])"
tag_5_matches = tag_5.str.extract(pattern)
print(tag_5_matches)
pattern = r"\[(\w+)\]"
tag_5_matches = tag_5.str.extract(pattern)
print(tag_5_matches)
tag_5_freq = tag_5_matches[0].value_counts()
print(tag_5_freq)

#pattern = r"\[\w+\]"
pattern = r"\[(\w+)\]"
tag_freq = titles.str.extract(pattern)[0].value_counts()

tag_freq

def first_10_matches(pattern):
    """
    Return the first 10 story titles that match
    the provided regular expression
    """
    all_matches = titles[titles.str.contains(pattern)]
    first_10 = all_matches.head(10)
    return first_10
# first_10_matches(r"[Jj]ava")
def first_10_matches(pattern):
    all_matches = titles[titles.str.contains(pattern)]
    first_10 = all_matches.head(10)
    return first_10
pattern = r"[Jj]ava[^Ss]"
java_titles = titles[titles.str.contains(pattern)]
first_10_matches(pattern)


string = "Sometimes people confuse JavaScript with Java"
pattern_1 = r"Java[^S]"

m1 = re.search(pattern_1, string)
print(m1)
# regular expression returns None
pattern_2 = r"\bJava\b"
m2 = re.search(pattern_2, string)
print(m2)
def first_10_matches(pattern):
    all_matches = titles[titles.str.contains(pattern)]
    first_10 = all_matches.head(10)
    return first_10
pattern = r"\b[Jj]ava\b"
java_titles = titles[titles.str.contains(pattern)]
first_10_matches(pattern)
test_cases = pd.Series([
    "Red Nose Day is a well-known fundraising event",
    "My favorite color is Red",
    "My Red Car was purchased three years ago"
])
print(test_cases)
test_cases.str.contains(r"^Red")
test_cases.str.contains(r"Red$")
pattern_beginning = r"^\[\w+\]"
beginning_count = titles.str.contains(pattern_beginning).sum()

pattern_ending =  r"\[\w+\]$"
ending_count = titles.str.contains(pattern_ending).sum()
beginning_count
email_tests = pd.Series(['email', 'Email', 'eMail', 'EMAIL'])
email_tests.str.contains(r"email")
import re
email_tests.str.contains(r"email",flags=re.I)
import re

email_tests = pd.Series(['email', 'Email', 'e Mail', 'e mail', 'E-mail',
              'e-mail', 'eMail', 'E-Mail', 'EMAIL'])

pattern = r"\be[\-\s]?mails?\b"
email_mentions = titles.str.contains(pattern,flags=re.I).sum()
email_mentions