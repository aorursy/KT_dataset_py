import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd
candy = pd.read_excel('/kaggle/input/candy-data/candyhierarchy2017.xlsx')
candy.head()
candy.columns
candy = candy.rename(columns = {'Q1: GOING OUT?' : 'going_out', 'Q2: GENDER' : 'gender', 'Q3: AGE': 'age', 'Q4: COUNTRY' : 'country',

       'Q5: STATE, PROVINCE, COUNTY, ETC' : 'area', 'Q10: DRESS' : 'dress', 'Q11: DAY': 'day',

       'Q12: MEDIA [Daily Dish]' : 'media_DailyDish', 'Q12: MEDIA [Science]': 'media_Science', 'Q12: MEDIA [ESPN]' : 'media_ESPN',

       'Q12: MEDIA [Yahoo]': 'media_Yahoo'})
candy.columns
candy['Unnamed: 113'].unique()
candy.drop(columns = ['Internal ID','Unnamed: 113', 'Click Coordinates (x, y)'], inplace = True)
candy.shape
candy.dropna(subset = ['going_out', 'gender', 'age', 'country', 'area'], how = 'all', inplace = True)

candy.reset_index(drop = True, inplace = True)
candy.shape
candy.going_out = candy.going_out.fillna('Not Sure')

candy.going_out.unique()
candy.gender.value_counts()
# Adding 11 NaN genders to type 3 - I'd rather not say seems to be the closest to unknown

candy[candy.gender == "I'd rather not say"].shape  #checking for spaces in text - found none
candy.gender = candy.gender.fillna("I'd rather not say")

candy.gender.value_counts()
candy.country.unique()
candy.country.value_counts(dropna = False).sort_values(ascending = False)
candy.country.isna().sum()
candy.country = candy.country.fillna('Unknown')
set([x for x in candy.country if 'u' in str(x)])  # unique values with 'u'
USA = [x for x in candy.country if (('u' in str(x) or 'U' in str(x)) and 'ingdom' not in str(x)\

     and 'urope' not in str(x) and 'stralia' not in str(x) and 'South Korea' not in str(x) and 'South africa' not in str(x) and 'uk' not in str(x))]
candy.country = candy.country.replace(to_replace = USA, value = 'USA')
candy.country.unique()
candy.country = candy.country.replace(to_replace = ['america','Ahem....Amerca',"'merica",'North Carolina ','cascadia',\

                                                   'New York','A','California','New Jersey','America','Alaska',\

                                                    'N. America'], value = 'USA')
canada = [x for x in candy.country if 'anada' in str(x).strip() or 'ANADA' in str(x) or 'Can' in str(x)]
candy.country = candy.country.replace(to_replace = canada, value = 'Canada')
candy.country.value_counts()
other = [x for x in candy.country.unique()]
other.remove('USA')

other.remove('Canada')
other
candy.country = candy.country.replace(to_replace = other, value = 'Other')
candy.country.value_counts()
candy.columns
candy = candy.astype({'going_out':'category', 'gender':'category', 'country':'category', 'dress':'category', 'day':'category'})
candy.describe(include = 'category')
def melt1(row):

    for c in data.columns:

        if row[c] == 1:

            return c
data = candy[candy.columns[-4:]]
data
new_col = data.apply(melt1, axis = 1)
candy['media_preference'] = new_col
candy.drop(columns = ['media_DailyDish','media_Science','media_ESPN','media_Yahoo'], inplace = True)
candy.media_preference.value_counts(dropna = False)
#Dividing questions and other columns

'''

candy_options = [i for i in candy.columns if 'Q6' in i or 'Q7' in i or 'Q8' in i or 'Q9' in i]

other_columns = [i for i in candy.columns if 'Q6' not in i and 'Q7' not in i and 'Q8' not in i and 'Q9' not in i]

'''



personal_info_cols = candy.columns[:6]

questionare_cols = candy.columns[5:]
candy.columns
responses = len(questionare_cols) - candy[questionare_cols].isna().sum(axis = 1)
candy['responses'] = responses
candy.head(3)
candy_questions = [x for x in candy.columns if 'Q6' in str(x)]
candy_questions
data = pd.DataFrame(candy[candy_questions])
data.shape
re = ['type_'+ str(x) for x in range(1,104)]



dic = {}

for i in range(len(data.columns)):

    dic[data.columns[i]] = re[i]
candy = candy.rename(columns = dic)

data = data.rename(columns = dic)
data = data.dropna(axis = 0, how = 'all')

data = data.reset_index(drop = True)
data.shape
data.head(4)
d = data.melt()
d.head(5)
import seaborn as sns



sns.countplot(data = d[:4000], x = 'variable', hue = 'value')