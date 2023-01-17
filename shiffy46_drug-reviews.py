# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import dateutil.parser as parser

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/drugsComTrain_raw.csv')
df2 = pd.read_csv('../input/drugsComTest_raw.csv')
df.head()
df.describe()
df2.describe()
df2.head()
#find a correlation among the rating and useful count
df.plot.scatter(x='rating',y='usefulCount')
#get the top 15 reviews in the list based on usefulness count
df.nlargest(15,'usefulCount')
#see the most used drugs to treat conditions
df['drugName'].value_counts()
#find the most common conditions that are treated
commonCdf = df['condition'].value_counts().nlargest(15)
print(type(commonCdf))
commonCdf
#checking out the test data
commonCdf2 = df2['condition'].value_counts().nlargest(15)
commonConditions = pd.DataFrame(commonCdf2)
commonConditions
#establish a dataframe to focus on birth control
BirthControlDf = df[['condition','rating','drugName']]
#locate each instance of birth control use in our dataframe
BirthControlDf=BirthControlDf.loc[BirthControlDf['condition'] == 'Birth Control']
#get the number of times each drug was used for birth control
BirthControlDf['drugName'].value_counts().nlargest(15)
#dataframe to show the average satifcation of using Etonogestrel for birth control(top choice of drug)
EtonogestrelBCDf = BirthControlDf.loc[BirthControlDf['drugName']=='Etonogestrel']
EtonogestrelBCDf['rating'].mean()
#add a column to dataframe with years extracted from the date
df['year'] = [parser.parse(date).date().year for date in df['date']]
df2['year'] = [parser.parse(date).date().year for date in df2['date']]
# Change in annual average ratings.

ConditionsDict = {}
ConditionsDictYearly = {}
conditions = list(commonCdf.index)[:5]
for condition in conditions:
    cdf = df[['condition','rating','drugName','year']].loc[df['condition'] == condition]
    cdf.reset_index(drop=True)
    drugs = list(cdf['drugName'].value_counts().nlargest(5).index)
    ConditionsDict[condition] = [cdf.loc[cdf['drugName'] == drug].reset_index(drop=True) for drug in drugs]
    ConditionsDictYearly[condition] = [cdf.loc[cdf['drugName'] == drug].groupby(['year']).mean().reset_index() for drug in drugs]
for condition in ConditionsDict:
    drugs = [cdf.at[0, 'drugName'] for cdf in ConditionsDict[condition]]
    fig = plt.figure(figsize=(10, 8))
    cdf = ConditionsDictYearly[condition]
    for i in range(len(drugs)):
        plt.plot(cdf[i]['year'], cdf[i]['rating'], label=drugs[i])
    plt.xlabel('Year')
    plt.ylabel('Ratings')
    plt.legend(loc='lower left')
    plt.title('Average annual ratings of the five most popular drugs for ' + condition)
