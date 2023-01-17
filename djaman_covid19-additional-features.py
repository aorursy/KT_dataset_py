import pandas as pd 



import plotly.express as px



import matplotlib.pyplot as plt 



%matplotlib inline

df = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
df['combine state'] = df['Country/Region'].fillna('') + str(': ') +df['Province/State'].fillna('')
drop_list =['Province/State','Country/Region']
new_df = df.drop(drop_list, axis = 1)
young = pd.read_csv('../input/population-dataset/Population_ages_0-14.csv', skiprows = 3)

adult = pd.read_csv('../input/population-dataset/Population_ages_15-64.csv', skiprows = 3)

elderly = pd.read_csv('../input/population-dataset/Population_ages_65_and_above.csv', skiprows= 3)
slice_young = young.loc[:,["Country Name","2016","2017","2018"]]

slice_adult = adult.loc[:,["Country Name","2016","2017","2018"]]

slice_elderly = elderly.loc[:,["Country Name","2016","2017","2018"]]

y_merged = df.merge(right = slice_young , how ="left", left_on="Country/Region", right_on= "Country Name",left_index= True)

y_merged.drop("Country Name", axis = 1, inplace= True)

y_merged.rename(columns={"2016": "2016_young_pop", "2017": "2017_young_pop","2018":"2018_young_pop"},inplace = True)

a_merged = y_merged.merge(right = slice_adult , how ="left", left_on="Country/Region", right_on= "Country Name",left_index= True)

a_merged.drop("Country Name", axis = 1, inplace= True)

a_merged.rename(columns={"2016": "2016_adult_pop", "2017": "2017_adult_pop","2018":"2018_adult_pop"},inplace = True)
o_merged = a_merged.merge(right = slice_elderly , how ="left", left_on="Country/Region", right_on= "Country Name",left_index= True)

o_merged.drop("Country Name", axis = 1, inplace= True)

o_merged.rename(columns={"2016": "2016_eld_pop", "2017": "2017_eld_pop","2018":"2018_eld_pop"},inplace = True)

o_merged.head()