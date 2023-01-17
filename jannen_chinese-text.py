import pandas as pd
import re

# Display whole text of dataframe field and don't cut it
pd.set_option('display.max_colwidth', -1)    
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df = pd.concat([df_train ,df_test],sort=True)
# return 1 or 0 - if text includes one or more these characters or not: u'[\u4e00-\u9fff]
def is_chinese(x):
    if re.search(u'[\u4e00-\u9fff]', x):
          return 1
    else: return 0

# no return, but print the sentence if chinese found
def is_chinese_print(x):
    if re.search(u'[\u4e00-\u9fff]', x):
          print('found chinese character in ' + x)
# test it
is_chinese(('What does 有毒 mean in Chinese?'))
is_chinese_print('What does 有毒 mean in Chinese?')
is_chinese('english only')
df_train['chinese']= df_train['question_text'].apply(lambda x: is_chinese(x))
df_train[(df_train['chinese']==1)][0:10]
df_train['chinese'].sum()
df_train[df_train['chinese']==1]['target'].sum()
df_train.drop('qid',axis=1)[(df_train['chinese']==1) & (df_train['target']==1)]