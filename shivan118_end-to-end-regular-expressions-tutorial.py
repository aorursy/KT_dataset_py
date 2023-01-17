import re

string = "tiger is the national animal of india "

pattern = "tiger"



# re.match function work on the only first function of the string

me = re.match(pattern, string)  

print(me)
import re

string = "tiger is the national animal of india "

pattern = "tiger"

pattern2 = "lion"



# re.match function work on the only first function of the string

me = re.match(pattern2, string)  

print(me)
string = "tiger is the national animal of india "

pattern = "national"



# re.search function works on the searc any where of the string.

me = re.search(pattern, string)

print(me)
string = "tiger is the national animal of india "

pattern = "national"

print(me.group(0))
string = "tiger is the national animal of india tiger is the national animal of india"

pattern = "national"



me = re.findall(pattern, string)

print(me)
me = re.finditer(pattern, string)   # ITER function returns  the indexes of the function present in the string.

for m in me:

    print(m.start())
string = "Ron was born on 12-09-1992 and he was addmited to school 15-12-1999"

pattern = "\d{2}-\d{2}-\d{4}"                          # we will use  while card specil character

me = re.findall(pattern, string)

print(me)
print(re.sub(pattern, "Monday", string))  # 
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
path = "/kaggle/input/tweets.csv"
dataset = pd.read_csv(path, encoding = "ISO-8859-1")



dataset.head()
for index, tweet in enumerate(dataset["text"][10:15]):

    print(index+1,".",tweet)
import re 



text = "RT @Joydas: Question in Narendra Modi App where PM is taking feedback if people support his #DeMonetization strategy https://t.co/pYgK8Rmg7r"

clean_text = re.sub(r"RT ", "", text)



print("Text before:\n", text)

print("Text after:\n", clean_text)

text = "@Jaggesh2 Bharat band on 28??<ed><U+00A0><U+00BD><ed><U+00B8><U+0082>Those who  are protesting #demonetization  are all different party leaders"

clean_text = re.sub(r"<U\+[A-Z0-9]+>", "", text)



print("Text before:\n", text)

print("Text after:\n", clean_text)

text = "RT @harshkkapoor: #DeMonetization survey results after 24 hours 5Lacs opinions Amazing response &amp; Commitment in fight against Blackmoney"

clean_text = re.sub(r"&amp;", "&", text)



print("Text before:\n", text)

print("Text after:\n", clean_text)
#List platforms that have more than 100 tweets

platform_count = dataset["statusSource"].value_counts()

top_platforms = platform_count.loc[platform_count>100]

top_platforms

def platform_type(x):

    ser = re.search( r"android|iphone|web|windows|mobile|google|facebook|ipad|tweetdeck|onlywire", x, re.IGNORECASE)

    if ser:

        return ser.group()

    else:

        return None



#reset index of the series

top_platforms = top_platforms.reset_index()["index"]



#extract platform types

top_platforms.apply(lambda x: platform_type(x))
text = "RT @Atheist_Krishna: The effect of #Demonetization !!\r\n. https://t.co/A8of7zh2f5"

hashtag = re.search(r"#\w+", text)



print("Tweet:\n", text)

print("Hashtag:\n", hashtag.group())
text = """RT @kapil_kausik: #Doltiwal I mean #JaiChandKejriwal is "hurt" by #Demonetization as the same has rendered USELESS <ed><U+00A0><U+00BD><ed><U+00B1><U+0089> "acquired funds" No wo"""

hashtags = re.findall(r"#\w+", text)



print("Tweet:\n", text)

print("Hashtag:\n", hashtags)

text = """@Joydas: Question in Narendra Modi App where PM is taking feedback if people support his #DeMonetization strategy https://t.co/pYgK8Rmg7r"""

remove = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

print("Remove:\n", remove)