# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/yerevanvoterssql/VotersYVN.csv')

print(df.shape)
df.describe(include=['object', 'bool'])
df[(df['LastName'] == 'Հարությունյան') & (df['FirstName'] == 'Անահիտ') & (df['MiddleName'] == 'Աշոտի')][['LastName','FirstName','MiddleName']]
df.groupby(by= ['LastName','FirstName','MiddleName'])['LastName','FirstName','MiddleName'].size().sort_values( ascending=False).head(20)
df[(df['LastName'] == 'Գրիգորյան') & (df['FirstName'] == 'Արթուր') & (df['MiddleName'] == 'Աշոտի')]
#pd.set_option('display.max_rows', 1000)
#df.sort_values(['LastName'],ascending=True).head(20)
pd.set_option('display.max_rows', 30000)
df.groupby(by= ['FirstName'])['FirstName'].size().sort_index().head(20)
pd.set_option('display.max_rows', 30000)
df.groupby(by= ['LastName'])['LastName'].size().sort_index().head(20)
#FirstNames sorted  at first by frequency , than  alphabetically
pd.set_option('display.max_rows', 30000)
ser = df.groupby(by= ['FirstName'])['FirstName'].size()
ser.iloc[np.lexsort([ ser.values])].head(20)
#LastNames sorted  at first by frequency , than  alphabetically
pd.set_option('display.max_rows', 30000)
ser = df.groupby(by= ['LastName'])['LastName'].size()
ser.iloc[np.lexsort([ ser.values])].head(20)
#MiddleNames sorted by frequency 
pd.set_option('display.max_rows', 30000)
df.groupby(by= ['MiddleName'])['MiddleName'].size().sort_values(ascending=True).head(20).sort_index()
#if in middleNames that means it is Male name 
#I suppose we can find female names by  left joining all Firstnames with MiddleNames  and consider remaining names as female names , which must be justified manually 
pd.set_option('display.max_rows', 1000)
df['LastNameLen'] = df['LastName'].apply(len)
#df.sort_values(by = ['LastNameLen','LastName'],ascending=True)[['LastName','FirstName','MiddleName']].head(1000)
df.sort_values(by = ['LastNameLen','LastName'],ascending=True).drop_duplicates(subset ='LastName', keep='last')[['LastName','FirstName','MiddleName']].head(20)
#ser= df['LastName'].apply(len).sort_values(ascending=True).head(300)
#df.iloc[ser.index].sort_values(by = ['LastName'],ascending=True)
pd.set_option('display.max_columns', 4000)
np.set_printoptions(threshold=np.nan)
df['FirstNameLen'] = df['FirstName'].apply(len)
print(df.sort_values(by = ['FirstNameLen','FirstName'],ascending=True).drop_duplicates(subset =['FirstName'], keep='first')['FirstName'].values[:20])
#[(df["FirstName"] != 'Անի') & (df["FirstName"] != 'Ադա') & (df["FirstName"] != 'Արա')  & (df["FirstName"] != 'Եւա') & (df["FirstName"] != 'Գոռ') ]
#ser= df['FirstName'].apply(len).sort_values(ascending=True).head(300)
#df.iloc[ser.index].sort_values(by = ['FirstName'],ascending=True)
pd.set_option('display.max_columns', 4000)

dfn = df['FirstName'].append(df['MiddleName'].str[:-1])

#dfn['NameLen'] = df['FirstName'].apply(len)
dfn.shape#tail()

names = dfn.sort_values(ascending=True).drop_duplicates()

names.shape
np.set_printoptions(threshold=np.nan)
print(names.values)
pd.set_option('display.max_rows', 1000)
df['LastNameLen'] = df['LastName'].apply(len)
df.sort_values(by = ['LastNameLen','LastName'],ascending=False).drop_duplicates(subset ='LastName', keep='last').head(10)[['LastName','FirstName','MiddleName']]

pd.set_option('display.max_rows', 1000)
df['FirstNameLen'] = df['FirstName'].apply(len)
df.sort_values(by = ['FirstNameLen','FirstName'],ascending=False).drop_duplicates(subset ='FirstName', keep='last')[['LastName','FirstName','MiddleName']].head(20)

#ser = df.groupby(by= ['BirthYear','FirstName'])['BirthYear','FirstName'].size().sort_values( ascending=False)#.drop_duplicates(subset ='FirstName', keep='first')
#ser.sort_index()
pd.set_option('display.max_rows', 30000)
df['BirthYear'] = df['BirthDay'].str.split('/').str[2]
df.groupby(by= ['BirthYear','FirstName'])['BirthYear','FirstName'].size().sort_index().to_frame().groupby(by= ['BirthYear']).idxmax(axis=0)#.sort_values( ascending=False).first()


df.groupby(by= ['BirthYear','FirstName'])['BirthYear','FirstName'].size().sort_values(ascending=False).to_frame().head(20)#.groupby(level=0, axis=1).max().sort_index#.sort_values(by = 'BirthYear',ascending = True)#.shape#drop_duplicates(subset ='', keep='first')
pd.set_option('display.max_rows', 100)
df['BirthDay'].value_counts().head(10)
df[~df["BirthDay"].str.contains('00/00')]['BirthDay'].value_counts().head(10)
df['BirthDate'] = pd.to_datetime(df['BirthDate'])
df.info()
df['Age'] = (pd.Timestamp.today() - df['BirthDate']).dt.days/365
df['Age'].describe()
df.sort_values(by=['Age', 'LastName',],ascending=[False, True])[['LastName','FirstName','MiddleName','BirthDate','Age']].head(10)
df[df['Age'].apply(lambda a: a>100)].count()[0] #distribution by age
!pip install --upgrade pip
!pip install transliterate
from transliterate import get_translit_function

translit_hy = get_translit_function('hy')
print(translit_hy(u"Լօրեմ իպսում դօլօր սիտ ամետ", reversed=True))
df.sort_values(by=['Age', 'LastName',],ascending=[False, True])[['LastName','FirstName','MiddleName','BirthDate','Age']].head(10)['LastName'].iloc[0]
df.sort_values(by=['Age', 'LastName',],ascending=[False, True])[['LastName','FirstName','MiddleName','BirthDate','Age']].head(10)['LastName'].apply(lambda x: translit_hy(x,reversed=True))
df['FirstName'] = df['FirstName'].apply(lambda x: str.replace(x, 'եւ', 'եվ'))
df['FirstNameLatin'] = df['FirstName'].apply(lambda x: translit_hy(x,reversed=True))
#import string
df['FirstNameLatin'] = df['FirstNameLatin'].apply(lambda x: str.replace(x, 'vo', 'o'))
df['FirstNameLatin'] = df['FirstNameLatin'].apply(lambda x: str.replace(x, "'", ''))
# word cloud library
from wordcloud import WordCloud
# matplotlib library
import matplotlib.pyplot as plt
from PIL import Image

plt.subplots(figsize = (128,128))
mask = np.array(Image.open("../input/pictures567/colorful.jpg"))
#mask = np.array(Image.open("../input/pictures1/Yerevan28000.jpeg"))
wordcloud = WordCloud (
                    background_color = 'white',
                    width = 2047,
                    height = 1536,
                    max_words=500,
                    mask=mask,
                    contour_width = .1, 
                    contour_color="orange",
                    relative_scaling =1
                        ).generate(' '.join(df.FirstNameLatin))
plt.imshow(wordcloud) # image show
plt.axis('off') # to off the axis of x and y

plt.savefig('Yerevan2800.jpg')
plt.show()