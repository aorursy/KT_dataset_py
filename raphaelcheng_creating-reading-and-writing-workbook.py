import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
data={'Apples':[30],'Bananas':[21]}
df = pd.DataFrame(data ,columns=['Apples','Bananas'])
print(df)
import pandas as pd
data={'Apples':[35,41],'Bananas':[21,34]}
df = pd.DataFrame(data ,index=['2017 sales','2018 sales'],columns=['Apples','Bananas'])
print(df)
# Your code hereimport pandas as pd
import pandas as pd
sf=pd.Series(['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'],name='Dinner')
print(sf)
import  pandas as pd
wineCsv = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
winPd = pd.DataFrame(wineCsv,columns=(['country', 'description', 'designation', 'points','price', 'province', 'region_1', 'region_2', 'variety', 'winery']))
print(winPd)

q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
import pandas as pd

q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

q6_df.to_csv("cows_and_goats.csv")