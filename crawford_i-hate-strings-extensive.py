import pandas as pd
import re
df = pd.DataFrame()
df['text'] = ['Meow',
              'HAHAH',
              'not funny', 
              str('from the film \"Aria\"''     6hAqj75PAI3m4bDdL0RVvF\u9866\u9876\u6743'),
              '420',
              '1337',
              '69',
              '6.9'
             ]
words = df['text'].str.contains(r'[A-z]')
df = df[~words]
df