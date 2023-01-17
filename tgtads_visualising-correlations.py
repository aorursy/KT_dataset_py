import numpy as np

import pandas as pd



# preparing data

responses = pd.read_csv("../input/responses.csv", index_col=False)



# df1 = responses.dropna(subset=['Giving', 'God']) # drop rows where essential data is missing

# print(np.corrcoef(df1['Giving'], df1['God']))



# df2 = responses.dropna(subset=['Dreams', 'Empathy'])

# print(np.corrcoef(df2['Dreams'], df2['Empathy']))



# droplist = ['Smoking', 'Alcohol', 'Punctuality', 'Lying', 'Internet usage', 'Gender', 'Left - right handed', 'Education', 'Only child', 'Village - town', 'House - block of flats']



# df3 = responses.drop(droplist, axis=1).dropna() # drop nonint columns and missing data



responses = responses.dropna()



music = responses.ix[:,1:19]

movies = responses.ix[:,19:31]

interests = responses.ix[:,31:46]

hobbies = responses.ix[:,46:63]

phobias = responses.ix[:,63:73]

health = responses.ix[:,73:76]

traits = responses.ix[:,76:133]

spending = responses.ix[:,133:140]

demographics = responses.ix[:,140:150]
from numpy.random import rand



# random dataframe for testing



def random_df(rows, columns):

    data = rand(rows, columns)

    df = pd.DataFrame(data)

    df.columns = [chr(x + ord('a')) for x in df.columns.values]

    return df



rdf = random_df(1000, 6)
# seaborn > stock matplotlib



import seaborn as sns



R = spending.ix[:,1:].corr()

ax = sns.heatmap(R)
# seaborn for distribution analysis

import seaborn as sns



col = 'Weight'



bx = sns.distplot(responses[col])



x = np.random.normal(np.mean(responses[col]), np.std(responses[col]), size=10000)

cx = sns.distplot(x, hist=False);



# import matplotlib.mlab as mlab

# import math



# x = np.linspace(rdf['a'].range[0], rdf['a'].range[1], 1000)

# cx = sns.distplot(mlab.normpdf(x, 0, math.sqrt(1)))



# cx = sns.distplot(responses['Height'])
# from scipy.stats import norm

import seaborn as sns

import numpy as np



x = np.random.normal(0, 1, size=100)

sns.distplot(x);
# identifying predictors/inputs for low self-esteem



# identifying highly correlated variables first



import seaborn as sns



R = traits.ix[:,:].corr()

ax = sns.heatmap(R)



# Loneliness and happiness are the most negatively correlated combination

# final judgement and god are highly postively correlated.



# happiness in life and energy levels also highly correlated.
# analytical/semantic clustering of variables is needed



# print(traits.columns.values)



self_esteem = traits.ix[:,['Self-criticism',

                          'Loneliness',

                          'Changing the past',

                          'Unpopularity',

                          'Happiness in life',

                          'Personality']]



# inverting them for a more negative outlook

self_esteem['Happiness in life'] = -1*self_esteem['Happiness in life']+5

self_esteem['Personality'] = -1*self_esteem['Personality']+5



import seaborn as sns



R = self_esteem.ix[:,:].corr()

ax = sns.heatmap(R, annot=True, fmt="0.00", square=True)



# nothing is correlated near +/-0.8, therefore 

# they do not need to be eliminated or combined 

# as independent variables when modelling
# traits['Personality'] will the dependent variable



# next step is to attempt to model.



# are there any correlations of the demographic data with this?



cols = ['Personality', 

        'Age', 

        'Height', 

        'Weight', 

        'Number of siblings', 

        'Gender', 

        'Left - right handed', 

        'Education', 

        'Only child', 

        'Village - town', 

        'House - block of flats']



modvars = responses[cols]



# convert categorical data to numbers

catcols = ['Gender', 

           'Left - right handed', 

           'Education', 

           'Only child', 

           'Village - town', 

           'House - block of flats']



modvar_dum = pd.get_dummies(modvars, columns=catcols)



# print(modvar_dum.head(1))



import seaborn as sns



R = modvar_dum.corr()

pre = (1-np.identity(R.shape[0]))*np.abs(R) # get absolute values and mask the diagonal



ax = sns.heatmap((pre >= 0.8), annot=False, fmt="0.00", square=True)



# very weak correlations with Personality
# what I would like to do, instead is, repeat the above process for all non-demographic fields,

# look at only the first row of the correlation coef matrix and identify where abs(corr(non-demographic field, demographic fields)) > 0.8
# The next step is to look at all the numerical dependent variables in the dataset

# and see if there are correlations with the demographic fields > +/- 0.8



# convert categorical data to numbers

catcols = ['Gender', 

           'Left - right handed', 

           'Education', 

           'Only child', 

           'Village - town', 

           'House - block of flats']

demographics_dum = pd.get_dummies(demographics, columns=catcols)



# pseudocode: 

# for column in response[dependent_variables]

# form a df with column and demographics_dum

# R = df.corr()

# pre = (1-np.identity(R.shape[0]))*np.abs(R)

# if any col in (pre[:1] >= 0.8), get colname

# 
