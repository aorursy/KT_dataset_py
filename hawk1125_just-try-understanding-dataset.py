import numpy as np 
import pandas as pd 
import re
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn import foo.bar # to be loaded by functions 

%matplotlib inline

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
class config :
    DATASET_PATH = "/kaggle/input/tweet-sentiment-extraction/"
    TRAIN_PATH   = "/kaggle/input/tweet-sentiment-extraction/train.csv"
    TEST_PATH    = "/kaggle/input/tweet-sentiment-extraction/test.csv"    
    SAMPLE_PATH  = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"    
df_train  = pd.read_csv(config.TRAIN_PATH)
df_test   = pd.read_csv(config.TEST_PATH)
df_sample = pd.read_csv(config.SAMPLE_PATH)
df_train.describe()
df_train.head()

df_test.describe()
df_test.head()
df_sample.describe()
df_sample.shape
class tweet : 
    
    def generate(self, str_) : 
        # in case, memorize argument's raw string
        self.rawstring = str_
        
        # basically, preprocessed string must be used.
        self.text = str_.lower()
        self.words = self.text.replace("[\.,!]","").split(" ")
        
        # check reply-to 
        reply = re.search("(\@\S+)", self.text)
        self.reply   = bool(reply)
        if self.reply :
            self.replyto = reply.groups()
        else :
            self.replyto = ()
        del reply

        # extract hashtags
        hashtags = re.search("(#\S+)", self.text)
        if bool(hashtags) : 
            self.hashtags = hashtags.groups()
        else :
            self.hashtags = ()
        del hashtags 
        
        # eliminate hashtags from words
        self.words = [ 
            w for w in self.words 
            if w not in self.hashtags
        ]

        # eliminate reply-to from words
        self.words = [ 
            w for w in self.words 
            if w not in self.replyto
        ]
        
        return self 

    
tw = tweet()

tw1 = tw.generate("Hello World!! @BillG #ProgrammingBeginner")
tw1.rawstring, tw1.words, tw1.hashtags, tw1.replyto, tw1.reply, 

