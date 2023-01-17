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
import re

hn = pd.read_csv("/kaggle/input/hacker-news-posts/HN_posts_year_to_Sep_26_2016.csv")
titles = hn['title']
titles
hn.head()
pattern = r"[Pp]ython"
python_counts = titles.str.contains(pattern).sum()
print(python_counts)
pattern = r'python'
python_counts = titles.str.contains(pattern,flags = re.I).sum()
print(python_counts)
pattern = r'[Ss][Qq][Ll]'
sql_counts = titles.str.contains(pattern).sum()
print(sql_counts)
pattern = r'[Ss][Qq][Ll]'
sql_counts = titles.str.contains(pattern,flags = re.I).sum()
print(sql_counts)
pattern = r"(SQL)"
sql_capitalizations = titles.str.extract(pattern, flags=re.I)
print(sql_capitalizations)
sql_capitalizations_freq = sql_capitalizations[0].value_counts()
print(sql_capitalizations_freq)
pattern = r"(\w+SQL)"
sql_flavors = titles.str.extract(pattern,flags= re.I)
sql_flavors_freq = sql_flavors[0].value_counts()
print(sql_flavors_freq)
hn_sql = hn[hn['title'].str.contains(r"\w+SQL", flags=re.I)].copy() 
hn_sql
hn_sql["flavor"] = hn_sql["title"].str.extract(r"(\w+SQL)", re.I)
hn_sql["flavor"] = hn_sql["flavor"].str.lower()
sql_pivot = hn_sql.pivot_table(index="flavor",values='num_comments', aggfunc='mean')
sql_pivot
pattern = r"[Pp]ython ([\d\.]+)"

py_versions = titles.str.extract(pattern)
py_versions_freq = dict(py_versions[0].value_counts())
py_versions_freq
def first_10_matches(pattern):
    """
    Return the first 10 story titles that match
    the provided regular expression
    """
    all_matches = titles[titles.str.contains(pattern)]
    first_10 = all_matches.head(10)
    return first_10

first_10_matches(r"\b[Cc]\b")
def first_10_matches(pattern):
    all_matches = titles[titles.str.contains(pattern)]
    first_10 = all_matches.head(10)
    return first_10
    

pattern = r"\b[Cc]\b [^ . +]"
first_10_matches(pattern)

test_cases = ['Red_Green_Blue',
              'Yellow_Green_Red',
              'Red_Green_Red',
              'Yellow_Green_Blue',
              'Green']
def run_test_cases(pattern):
    for tc in test_cases:
        result = re.search(pattern, tc)
        print(result or "NO MATCH")
run_test_cases(r"Green(?=_Blue)")
run_test_cases(r"Green(?!_Red)")
run_test_cases(r"(?<=Red_)Green")
run_test_cases(r"(?<!Yellow_)Green")
run_test_cases(r"Green(?=.{5})")
first_10_matches(r"\b[Cc]\b[^.+]")
pattern = r"(?<!Series\s)\b[Cc]\b(?![\+\.])"
c_mentions = titles.str.contains(pattern).sum()
c_mentions 
test_cases = [
              "I'm going to read a book.",
              "Green is my favorite color.",
              "My name is Aaron.",
              "No doubles here.",
              "I have a pet eel."
             ]
for tc in test_cases:
    print(re.search(r"(\w)\1", tc))
test_cases = pd.Series(test_cases)
test_cases

print(test_cases.str.contains(r"(\w)\1"))
pattern = r"\b(\w+)\s\1\b"

repeated_words = titles[titles.str.contains(pattern)]
repeated_words
re.sub(pattern, repl, string, flags=0)
string = "aBcDEfGHIj"

print(re.sub(r"[A-Z]", "-", string))
sql_variations = pd.Series(["SQL", "Sql", "sql"])

sql_uniform = sql_variations.str.replace(r"sql", "SQL", flags=re.I)
print(sql_uniform)
email_variations = pd.Series(['email', 'Email', 'e Mail',
                        'e mail', 'E-mail', 'e-mail',
                        'eMail', 'E-Mail', 'EMAIL'])
pattern = r"e[\-\s]?mail"
email_uniform = email_variations.str.replace(pattern, "email", flags=re.I)
email_uniform
titles_clean = titles.str.replace(pattern, "email", flags=re.I)
titles_clean
test_urls = pd.Series([
 'https://www.amazon.com/Technology-Ventures-Enterprise-Thomas-Byers/dp/0073523429',
 'http://www.interactivedynamicvideo.com/',
 'http://www.nytimes.com/2007/11/07/movies/07stein.html?_r=0',
 'http://evonomics.com/advertising-cannot-maintain-internet-heres-solution/',
 'HTTPS://github.com/keppel/pinn',
 'Http://phys.org/news/2015-09-scale-solar-youve.html',
 'https://iot.seeed.cc',
 'http://www.bfilipek.com/2016/04/custom-deleters-for-c-smart-pointers.html',
 'http://beta.crowdfireapp.com/?beta=agnipath',
 'https://www.valid.ly?param',
 'http://css-cursor.techstream.org'
])
pattern = r"https?://([\w\-\.]+)"
test_urls_clean = test_urls.str.extract(pattern,flags=re.I)
test_urls_clean
domains = hn["url"].str.extract(pattern,flags= re.I)
top_domains = domains[0].value_counts().head(5)
top_domains
created_at = hn['created_at'].head()
print(created_at)
pattern = r"(.+)\s(.+)"
dates_times = created_at.str.extract(pattern)
print(dates_times)
pattern = r"(https?)://([\w\.\-]+)/?(.*)"

test_url_parts = test_urls.str.extract(pattern, flags=re.I)
test_url_parts
url_parts = hn['url'].str.extract(pattern, flags=re.I)
url_parts
#e.g
created_at = hn['created_at'].head()

pattern = r"(.+) (.+)"
dates_times = created_at.str.extract(pattern)
print(dates_times)
pattern = r"(?P<date>.+) (?P<time>.+)"
dates_times = created_at.str.extract(pattern)
print(dates_times)
pattern = r"(?P<protocol>https?)://(?P<domain>[\w\.\-]+)/?(?P<path>.*)"
url_parts = hn['url'].str.extract(pattern,flags=re.I)
url_parts
