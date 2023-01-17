import numpy as np

import pandas as pd

import sklearn as sk

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import re
df = pd.read_csv('../input/nyc-jobs.csv')

df.info()
pd.set_option("display.max_columns", 28)

df.head()
df = df.dropna(subset=['Salary Range From', 'Salary Range To', 'Salary Frequency','Full-Time/Part-Time indicator', 'Minimum Qual Requirements', 'Preferred Skills']) 

df = df[(df['Salary Frequency'] =='Annual') & (df['Full-Time/Part-Time indicator']  =='F') & (df['Salary Range From'] > 0) & (df['Salary Range To']  >0)]



df.info()
df.describe()
plt.hist(df['Salary Range From'], bins=50, alpha=  0.5, color='r', label='Salary Range From')

plt.hist(df['Salary Range To'],     bins=50, alpha = 0.5, color='b', label='Salary Range To')

plt.xlabel('Salary ($/year)')

plt.title('Distribution of salary')



plt.axvline(df['Salary Range From'].quantile(.75), color='r')

plt.axvline(df['Salary Range To'].quantile(.75), color='b')



plt.legend()

plt.show()
min75 = 73576

max75 = 108653



df.loc[  (df['Salary Range From'] > min75), 'Min_Salary75'] = 1

df.loc[~(df['Salary Range From'] > min75), 'Min_Salary75'] = 0



df.loc[ (df['Salary Range To'] > max75), 'Max_Salary75'] = 1

df.loc[~(df['Salary Range To'] > max75), 'Max_Salary75'] = 0



df['Min_Salary75'] = df['Min_Salary75'].astype(int)

df['Max_Salary75'] = df['Max_Salary75'].astype(int)



df.head()
def clensing(df_series):

    df = df_series.replace('[^a-zA-Z ]',' ', regex = True)

    df = df.str.lower()

    return df



df['MinQualReq'] = clensing(df['Minimum Qual Requirements'])

df['PrefSkills']     = clensing(df['Preferred Skills'])
df['PrefSkills'].head()
def calc_tfidf(docs, count, tfidf):

    bag = count.fit_transform(docs)

    t = tfidf.fit_transform(bag)

    return bag, t



def conc_text(texts, flags):

    pos = ""

    neg = ""

    for (t,f) in zip(texts.values, flags.values):

        if f >0:

            pos = pos + t + " "

        else:

            neg = neg + t + " "

    

    return [pos,neg]



tfidf = TfidfTransformer(use_idf = True, norm ='l2', smooth_idf = True)

count = CountVectorizer()



docs1 = conc_text(df['MinQualReq'], df['Min_Salary75'])

bag1, tfidf1 = calc_tfidf(docs1, count, tfidf)
bag1.shape
print(tfidf1.toarray())
def stats(count, tfidf):

    df1 = pd.DataFrame(list(count.vocabulary_.items()),columns=['word','id'])

    df1 = df1.sort_values('id').reset_index()

    dfx = pd.DataFrame(tfidf.toarray().T)

    dfx.columns = ['tf-idf for high salary', 'tf-idf for low salary']

    df1 = pd.concat([df1, dfx], axis=1)

    df1['diff'] = df1['tf-idf for high salary']- df1['tf-idf for low salary']

    return df1



df1 = stats(count,tfidf1)
df1.head()
df1.nlargest(20,'diff')
docs2 = conc_text(df['MinQualReq'], df['Max_Salary75'])

bag2, tfidf2 = calc_tfidf(docs2, count, tfidf)

df2 = stats(count, tfidf2)

df2.nlargest(20,'diff')
docs3 = conc_text(df['PrefSkills'], df['Min_Salary75'])

bag3, tfidf3 = calc_tfidf(docs3, count, tfidf)

df3 = stats(count, tfidf3)

df3.nlargest(20,'diff')
docs4 = conc_text(df['PrefSkills'], df['Max_Salary75'])

bag4, tfidf4 = calc_tfidf(docs4, count, tfidf)

df4 = stats(count, tfidf4)

df4.nlargest(20,'diff')