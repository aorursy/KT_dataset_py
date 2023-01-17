import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt

df = pd.read_csv('../input/FAO.csv',encoding = 'ISO-8859-1')



fields = ['Area','Element']

countries = list(df.Area.unique())

yearexp = r'\d{4}'

years = [int(re.findall(yearexp,x)[0]) for x in df.columns if len(re.findall(yearexp,x))>0]



def year_slice(year):

    global df,fields

    yearStr = 'Y' + str(year)

    df2 = df[fields+[yearStr]].copy()

    df2['year'] = year

#df2.rename(columns={'old column': 'new column'},inplace=True)

    df2.rename(columns={yearStr:'amount'},inplace=True)

    df2.set_index('Area',inplace=True)

    def get_ffr(country):

        cnty = df2.loc[country]

        hasfood = 'Food' in cnty.Element.unique()

        hasfeed = 'Feed' in cnty.Element.unique()

        if hasfeed:

            hasfeed = cnty[cnty['Element']=='Feed']['amount'].sum()>0

            ffr=0;

            if hasfood and hasfeed:

                ffr = cnty[cnty['Element']=='Food']['amount'].sum()/cnty[cnty['Element']=='Feed']['amount'].sum()

        return ffr

    ffr = [get_ffr(c) for c in countries]

    yr = [year for x in range(len(countries))]

    df3 = pd.DataFrame({'country':countries,'ffr':ffr,'year':yr})

    return df3

slices = [year_slice(yr) for yr in years]

df2 = pd.concat(slices,axis=0)

len(df2)



df2 = pd.concat(slices,axis=0)

df2.head()
x = years

y = [df2[df2['year']==y]['amount'].sum() for y in years]

plt.plot(x,y)

countries = ['United States of America','Sweden']

country_dict = dict(zip(countries,list(range(len(countries)))))

df3=df2[df2['country'].isin(countries)][['ffr','country']]

df3['id'] = df3.apply(lambda x:country_dict[x.country],axis=1)

df3['dummy']=1

del df3['country']

df3.head()





X_train, X_test = train_test_split(df3,test_size=0.2)

used_fields = [x for x in df3.columns if not x=='id']

gnb = GaussianNB()

gnb.fit(

    X_train[used_fields],

    X_train.id

)

y_pred = gnb.predict(X_test[used_fields])

result = pd.DataFrame({'pred':y_pred,'act':X_test.id})

result['correct'] = result.apply(lambda x:1 if x.pred==x.act else 0,axis=1)

result.groupby('act').mean()['correct']