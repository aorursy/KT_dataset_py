%matplotlib inline 

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 



import seaborn as sns 



from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
data = pd.read_csv('../input/states_all.csv')
data.shape
print (data.columns) 

data.head()
data.info()
data.shape
print (data['YEAR'].max())

print (data['YEAR'].min())
data = data.dropna(subset=['ENROLL'])
data.shape
1492-1229
del data['PRIMARY_KEY']
data.set_index('STATE')

data.head()
#picking columns that are relevant to the scoring 

data.isnull().any()
scores = ['AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE','AVG_READING_4_SCORE','AVG_READING_8_SCORE']
scores_df = data[scores].dropna().copy()

print (scores_df.isna().sum())
scores_df.shape
scores_df.isna().sum()
scores_df.index
X = StandardScaler().fit_transform(scores_df)

X
kmeans = KMeans(n_clusters=4)

model = kmeans.fit(X)

print("model\n", model)
centers = model.cluster_centers_

centers
def pd_centers(featuresUsed, centers):

    colNames = list(featuresUsed)

    colNames.append('prediction')



    # Zip with a column called 'prediction' (index)

    Z = [np.append(A, index) for index, A in enumerate(centers)]



    # Convert to pandas data frame for plotting

    P = pd.DataFrame(Z, columns=colNames)

    P['prediction'] = P['prediction'].astype(int)

    return P
from pandas.plotting import parallel_coordinates
from itertools import cycle, islice
def parallel_plot(data):

	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))

	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])

	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')
P = pd_centers(scores, centers)

P
parallel_plot(P[P['AVG_MATH_8_SCORE'] < 1])
parallel_plot(P[P['AVG_READING_8_SCORE'] < 1])
data.head()
data[data['YEAR'].isin([2015])].sort_values(by = 'AVG_MATH_4_SCORE')
print ('Minnesota enrollment for 2015 = 807044')

print ('Alabama enrollment for 2015 = 734974')
revenue_data = data[['STATE','YEAR','TOTAL_REVENUE','FEDERAL_REVENUE','STATE_REVENUE','LOCAL_REVENUE']] 

revenue_data.tail()
year = 2016 

state1 = 'MINNESOTA'





filter1 = revenue_data['STATE'].str.contains(state1)  

filter2 = revenue_data['YEAR'].isin([year])



rev = revenue_data[filter1 & filter2] 

type(rev)
year = 2016 

state1 = 'ALABAMA'





filter1 = revenue_data['STATE'].str.contains(state1)  

filter2 = revenue_data['YEAR'].isin([year])



rev1 = revenue_data[filter1 & filter2] 

type(rev1)
df = pd.concat([rev,rev1])

##uSE THE PANDAS.PLOT.BAR() 11PM AT NIGHT GOING TO SLEEP 

df
del df['YEAR']
df

df_melt = pd.melt(df,id_vars=['STATE'] , var_name='revenue')

df_melt

                  
plt.figure(figsize=(10,6))

sns.barplot(x = 'revenue', y= 'value', hue='STATE', data=df_melt)
expenditure_data = data[['STATE','YEAR','TOTAL_EXPENDITURE', 'INSTRUCTION_EXPENDITURE',

       'SUPPORT_SERVICES_EXPENDITURE', 'OTHER_EXPENDITURE',

       'CAPITAL_OUTLAY_EXPENDITURE']]

expenditure_data.head()
year = 2016 

state1 = 'MINNESOTA'





filter3 = expenditure_data['STATE'].str.contains(state1)  

filter4 = expenditure_data['YEAR'].isin([year])



exp = expenditure_data[filter3 & filter4] 

exp
year = 2016 

state1 = 'ALABAMA'





filter3 = expenditure_data['STATE'].str.contains(state1)  

filter4 = expenditure_data['YEAR'].isin([year])



exp1 = expenditure_data[filter3 & filter4] 

exp1
df1 = pd.concat([exp,exp1])

df1
df1 = df1.drop('YEAR', 1)
df1
df_melt = pd.melt(df1,id_vars=['STATE'] , var_name='expenditure')

df_melt

  
df_melt
plt.figure(figsize=(15,10))

sns.barplot(x = 'expenditure', y= 'value', hue='STATE', data=df_melt)