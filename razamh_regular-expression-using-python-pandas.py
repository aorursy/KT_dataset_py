# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
hn = pd.read_csv("/kaggle/input/hacker_news.csv")

hn.head()
title = hn["title"]

title.head(10)
# creat a pattern for regular expression

pattern = r"[Pp]ython" # [] is a set
title.str.contains(pattern).sum()
# To see title have Python or not.

title[title.str.contains(pattern)]
# RE using Python

import re
python = 0



for i in title:

    if re.search(pattern,i):

        python += 1
python 
pattern = r"[12][0-9][0-9][0-9]"

pattern = r"[12][0-9]{3}"
# Now check the "email", "e_mail" in (hn) dataframe

pattern = r"e_?mail"
title.str.contains(pattern).sum()
title[title.str.contains(pattern)]
# To check [pdf] & [videos] in hn

pattern = r"(\[\w+\])"
title[title.str.contains(pattern)].head()
title.str.extract(pattern).head()
title.str.extract(pattern).iloc[100]
# If we have this type of sentence then

# "Javascript"

# "javaScript"

# "Java"

# "java"

pattern = "[Jj]ava[^Ss]"
"I am Java lover"

"I am Java lover and JavaScript"

"I am Javaprogramming lover"

"I am Java"
if re.search(pattern,"I am Java"):

    print("I found")
pat = r"\b[Jj]ava\b" # word boundry character 
if re.search(pat,"I am Java"):

    print("I found")
pat = r"\b[Jj]ava\w*\b"
if re.search(pat,"I am Javaprogramming lover"):

    print("I found")
pattern = r"^\[\w+\]"
title[title.str.contains(pattern)].head()
pattern = r"\[\w+\]$"



title[title.str.contains(pattern)].head()
pat = r"\b[Cc]\b[^+]"
title[title.str.contains(pat)].head()
pat = r"\b(?<!Series\s)[Cc]\b"
title[title.str.contains(pat)].head()