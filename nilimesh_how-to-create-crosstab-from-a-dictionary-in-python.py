# ========================================================================================

# Applied Data Science Recipes @ https://wacamlds.podia.com

# Western Australian Center for Applied Machine Learning and Data Science (WACAMLDS)

# ========================================================================================



print()

print(format('How to create crosstabs from a Dictionary in Python','*^82'))    

import warnings

warnings.filterwarnings("ignore")



# load libraries

import pandas as pd

    

raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 

            'company': ['infantry', 'infantry', 'cavalry', 'cavalry', 'infantry', 'infantry', 'cavalry', 'cavalry','infantry', 'infantry', 'cavalry', 'cavalry'], 

            'experience': ['veteran', 'rookie', 'veteran', 'rookie', 'veteran', 'rookie', 'veteran', 'rookie','veteran', 'rookie', 'veteran', 'rookie'],

            'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 

            'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],

            'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}



df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'experience', 'name', 'preTestScore', 'postTestScore'])

print(); print(df)



# Create a crosstab table by company and regiment

df1 = pd.crosstab(df.regiment, df.company, margins=True)

print(); print(df1)    



# Create more crosstabs

df2 = pd.crosstab([df.company, df.experience], df.regiment,  margins=True)

print(); print(df2)



df3 = pd.crosstab([df.company, df.experience, df.preTestScore], df.regiment,  margins=True)

print(); print(df3)    