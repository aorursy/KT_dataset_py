!pip install psycopg2-binary

!pip install dateparser
import os

import json



import psycopg2



import pandas as pd





connection = psycopg2.connect(user = 'data_access',

                              password = 'H5yfSApxS5bk6f',

                              host = 'spotlightdata-covid-do-user-1518235-0.a.db.ondigitalocean.com',

                              port = '25060',

                              database = 'covid')



cur = connection.cursor()



query = ("SELECT MIN(d.date_published), MAX(d.date_published) "

         "FROM nanowire.data_analysis_nlp as dan "

         "join nanowire.data_analysis da on dan.data_analysis_id = da.id "

         "join nanowire.documents as d on d.document_uuid = da.source_uuid "

         "WHERE dan.data_group='keywords';")



cur.execute(query)



dates = cur.fetchone()



cur.close()



startDate = min(dates)

endDate = max(dates)



dateRange = pd.date_range(startDate, endDate, freq='M')



connection.close()



print('finished cell, collected {0} dates'.format(len(dateRange)))
import matplotlib.pyplot as plt

import dateparser



connection = psycopg2.connect(user = 'data_access',

                              password = 'H5yfSApxS5bk6f',

                              host = 'spotlightdata-covid-do-user-1518235-0.a.db.ondigitalocean.com',

                              port = '25060',

                              database = 'covid')



#define the query to grab keywords where the snippets contain 

docs_query = ("with entries as ( "

                "    select "

                "        date_trunc('month', sd.date_published)::date as date, count(sd.document_uuid)::int "

                "    from nanowire.documents as sd "

                "    group by date "

                "), range as ( "

                "    select "

                "        min(date) as min_date, "

                "        max(date) as max_date "

                "    from entries "

                ") "

                "select (((series.date) + interval '1 month') - interval '1 day')::date as date, coalesce(entries.count, series.count) as count "

                "from ( "

                "    select generate_series( "

                "        date_trunc('month', min_date), "

                "        date_trunc('month', max_date), "

                "        ('1 ' || 'month')::interval)::date as date, 0 as count from range "

                ") as series "

                "left join entries on series.date = entries.date "

                "order by date;")





docsDict = {'has_date':[],

            'has_documents':[]}





cur = connection.cursor()



cur.execute(docs_query)



docsCounts = [d for d in cur]



cur.close()

        

connection.close()



for doc in docsCounts:

    

    docsDict['has_date'].append(doc[0])

    docsDict['has_documents'].append(doc[1])





#store the document counts in a dictionary

countDf = pd.DataFrame(docsDict)

    

#show our collected document counts over time

plt.figure(figsize=(16, 12))

plt.plot(docsDict['has_date'], docsDict['has_documents'])

plt.xlabel('Date')

plt.ylabel('Count')

plt.title('Documents per month')

plt.show()
connection = psycopg2.connect(user = 'data_access',

                              password = 'H5yfSApxS5bk6f',

                              host = 'spotlightdata-covid-do-user-1518235-0.a.db.ondigitalocean.com',

                              port = '25060',

                              database = 'covid')



#collect all our words together so we can see how big our vocabulary is

wordsQuery = ("SELECT DISTINCT dan.text "

              "from nanowire.data_analysis_nlp as dan "

              "where data_group = 'keywords';")



cur = connection.cursor()



cur.execute(wordsQuery)



words = [w[0] for w in cur]



cur.close()



print("FOUND {0} WORDS".format(len(words)))



#work out the total number of documents

totalDocs = sum(docsDict['has_documents'])



#IDF = (Total number of documents/Number of documents with target word in)

IDFQuery = ("SELECT dan.text, {0}::float/COUNT(DISTINCT d.id) "

            "from nanowire.data_analysis_nlp as dan "

            "join nanowire.data_analysis da on dan.data_analysis_id = da.id "

            "join nanowire.documents as d on d.document_uuid = da.source_uuid "

            "where data_group = 'keywords' "

            "GROUP BY dan.text "

            "HAVING COUNT(DISTINCT d.id) > 50 "

            "ORDER BY COUNT(DISTINCT d.id) DESC;")



cur = connection.cursor()



cur.execute(IDFQuery.format(totalDocs))



IDF_scores = [s for s in cur]



cur.close()

connection.close()



for si, s in enumerate(IDF_scores):

    

    print(s)

    

    if si > 20:

        

        break

cutoff = 6.5

automaticStopwords = []

for word in IDF_scores:

    

    if word[1] > cutoff:

        break

    

    automaticStopwords.append(word[0])

    

print("SELECTED STOPWORDS")

print(automaticStopwords)
connection = psycopg2.connect(user = 'data_access',

                              password = 'H5yfSApxS5bk6f',

                              host = 'spotlightdata-covid-do-user-1518235-0.a.db.ondigitalocean.com',

                              port = '25060',

                              database = 'covid')



#define the query to grab keywords where the snippets contain 

range_query = ("select dan.text, COUNT(dan.text) "

                "from nanowire.data_analysis_nlp as dan "

                "join nanowire.data_analysis da on dan.data_analysis_id = da.id "

                "join nanowire.documents as d on d.document_uuid = da.source_uuid "

                "where data_group = 'keywords' "

                "AND d.date_published < '{0}' "

                "AND d.date_published >= '{1}' "

                "GROUP BY dan.text;")





storeDict = {'has_date':[],

            'has_value':[],

            'has_count':[]}



for datei, date in enumerate(dateRange):

    

    cur = connection.cursor()

    

    if datei % 50 ==0:

        print('REACHED ROW', datei)

    

    if datei == 0:

        cur.execute(range_query.format(date.strftime('%Y-%m-%d'), '1900-1-1'))

    else:

        cur.execute(range_query.format(date.strftime('%Y-%m-%d'), dateRange[datei-1].strftime('%Y-%m-%d')))

        

    for r in cur:

        

        storeDict['has_date'].append(date)

        storeDict['has_value'].append(r[0])

        storeDict['has_count'].append(r[1])

        

    cur.close()

        

connection.close()





df = pd.DataFrame(storeDict)



df.head()
from datetime import datetime



window_start = datetime(2015, 1, 1)

window_end = datetime(2020, 3, 1)



normalisedDict = {'has_date':[],

                  'has_value':[],

                  'has_score':[]}



for di, date in enumerate(dateRange):

    

    if di % 50 == 0:

        print("REACHED ROW", di)

    

    if date <= window_end and date >= window_start:

        

        filtered_df = df[df['has_date'] == date]

        

        date_count = countDf[countDf['has_date'] == date].iloc[0]['has_documents']



        for rowi, row in filtered_df.iterrows():

            

            normalisedDict['has_date'].append(date)

            normalisedDict['has_value'].append(row['has_value'])

            normalisedDict['has_score'].append(row['has_count']/date_count)



#drop the result into a dataframe for easy manipulation            

normalisedDf = pd.DataFrame(normalisedDict)



normalisedDf.head()
import time

import math

import numpy as np

from datetime import datetime

from nltk.corpus import stopwords





#extract our data from a csv

def extract_from_csv(df, dateCol, valueCol, countCol):

    

    df[dateCol] = pd.to_datetime(df[dateCol])

    

    #extract the dates using pandas tools to speed everything up

    dates = np.array(df[dateCol].unique())

    

    #convert to numpy dates to make things compatable

    dates = dates.astype('datetime64')



    #make sure the dates are in the right order

    dates.sort()

    

    #convert each of the words so that they're all in the right order

    words_list = {}

    for rowi, row in df.iterrows():

        

        d = np.datetime64(row[dateCol])

        k = row[valueCol]

        count = row[countCol]

        

        if isinstance(k, str):

        

            if k not in words_list.keys():

    

                words_list[k] = np.zeros(len(dates))

            

            ind = np.where(dates == d)



            words_list[k][ind] += float(count)



    return [dates, words_list]



#extract the rising words

def extract_rising(words_list, words_coefs):

    

    #combine everything into one list

    d = []

    for wi, word in enumerate(words_list):

        if math.isnan(words_coefs[wi][0]):

            d.append([word, 0, 1])

        else:

            d.append([word, words_coefs[wi][0], words_coefs[wi][1], words_coefs[wi][2]])

    

    #sort the words in ascending order of gradient co-efficient

    d.sort(key = lambda x: x[1], reverse=True)

    

    out = []

    

    for p in d:

        out.append({'term':p[0],

                    'm':float(p[1]), 

                    'c':round(float(p[2]), 3),

                    'error':round(float(p[3]), 3)})

    

    return out



#extract the falling words

def extract_falling(words_list, words_coefs):

    

    #combine everything into one list

    d = []

    for wi, word in enumerate(words_list):

        if math.isnan(words_coefs[wi][0]):

            d.append([word, 0, 1])

        else:

            d.append([word, words_coefs[wi][0], words_coefs[wi][1], words_coefs[wi][2]])

    

    #sort the words in ascending order of gradient co-efficient

    d.sort(key = lambda x: x[1], reverse=True)

    

    out = []

    

    for p in d[::-1]:

        out.append({'term':p[0],

                    'm':float(p[1]), 

                    'c':round(float(p[2]), 3),

                    'error':round(float(p[3]), 3)})

    

    return out



#this is the function that does the calculation of the m/c values

def linear_trend_finding(dates, words_list):

    

    #convert the dates into something the polyfit tool can actually work with

    dates = dates.astype(datetime)



    #make sure time starts at zero

    start = dates[0]

    

    delta = [x - start for x in dates]



    #I have no idea why in the name of god this can change. Everything is 

    #identical between test case and live but sometimes it is int and sometimes

    #it is a datetime

    if isinstance(delta[0], int):



        delta[:] = [x/(60*60*24) for x in delta]

        

    else:

        delta[:] = [(x.total_seconds())/(60*60*24) for x in delta]



    

    #find the m and c values for all the words the tool has been sent

    word_coefs = []

    for w in words_list.keys():

        

        #convert date to numerical data (hours from start to end of dataset)

        lin = np.polyfit(delta, words_list[w], deg=1, full=True)



        #the spearmans function understands dates

        word_coefs.append([lin[0][0], lin[0][1], lin[1][0]])

    

    #sort the results to find the most rising words

    rising = extract_rising(words_list, word_coefs)



    return {'results':rising}



################################################################

### End of utility functions, start of main class of toolbox ###

################################################################



#this class is designed to tie everything together

class trends_tool(object):

    

    #initialise the tool with stopwords

    def __init__(self):

        

        self.baseStops = set(stopwords.words('english'))

    

    #limit the number of words we're returning

    def apply_word_limits(self, result, nWords):

    

        if isinstance(nWords, int):



            if math.floor(nWords) * 2 < len(result['results']):



                #ensure the top and bottom trends are all as they should be

                positives = result['results'][:nWords]

                negatives = result['results'][-nWords:]



                positives[:] = [x for x in positives if x['m'] >= 0]



                negatives[:] = [x for x in negatives if x['m'] <= 0]



                #reverse the order of the negatives so they go low to high

                negatives = negatives[::-1]



            else:

                print("LESS WORDS IN DATASET THAN REQUESTED")



        return positives + negatives

        

    #remove the stopwords from the dataset

    def remove_stops(self, results, customStops):

        

        if customStops == []:

            return results

        

        #remove stopwords from the list



        results['results'][:] = [r for r in results['results'] if (r['term'] not in self.baseStops and r['term'] not in customStops)]



        return results



    def main(self, df, dateCol, valueCol, countCol, customStops = ['']):



        load_time_start = time.time()

        

        #extract the interesting information in the csv in a 'nice' format

        [dates, words_list] = extract_from_csv(df, dateCol, valueCol, countCol)

        

        loading_time = time.time() - load_time_start

        



        #check to see if there's even enough data to do anything with

        if len(dates) <= 3:

            raise Exception("INSUFFICIENT DATA TO PERFORM TREND ANALYSIS, PLEASE PROVIDE MORE DATA POINTS")

        

        #perform the trend finding

        result = linear_trend_finding(dates, words_list)

        

        #remove any stopwords in the limits

        result = self.remove_stops(result, customStops)



        #limit the number of words to be returned

        result = self.apply_word_limits(result, nWords=10)

        

        out = {'result':result}

        

        #add some metadata to the output        

        out['startDate'] = str(dates[0])

        out['endDate'] = str(dates[-1])

                

        return out



##################################

### End of function defintions ###

##################################

    

tool = trends_tool()



print('Initiated tool')
normal_trends_result = tool.main(normalisedDf, dateCol='has_date', valueCol='has_value', countCol='has_score', customStops = automaticStopwords + ['Figure'])#customStops = ['influenza', 'disease', 'infection', 'virus'])



#from pprint import pprint



#pprint(normal_trends_result['result'])



print('Collected trends results')
startDate = min(normalisedDf['has_date'])

endDate = max(normalisedDf['has_date'])



plotDateRange = list(normalisedDf.has_date.unique())



fig, ax = plt.subplots(figsize=(16, 12))



offset = 0



for ti, term in enumerate(normal_trends_result['result']):

    

    if ti > 0 + offset:



        values = normalisedDf[normalisedDf['has_value'] == term['term']]

        

        for date in plotDateRange:

            

            if date not in list(values['has_date']):



                values.loc[len(values)] = [date, term['term'], 0]

                

        values = values.sort_values(by=['has_date'])

        

        plt.plot(list(values['has_date']), list(values['has_score']), label=term['term'])



    

    if ti > 5 + offset:

        

        break

        

plt.xlabel('Date')

plt.ylabel('Count')

plt.xticks(rotation=45)

#every_nth = 24

#for n, label in enumerate(ax.xaxis.get_ticklabels()):

#    if n % every_nth != 0:

#        label.set_visible(False)

plt.legend()

plt.show()
fig, ax = plt.subplots(figsize=(16, 12))



rev_norm_results = normal_trends_result['result'][::-1]



offset = 0



for ti, term in enumerate(rev_norm_results):

    

    if ti > 0 + offset:



        values = normalisedDf[normalisedDf['has_value'] == term['term']]

        

        for date in plotDateRange:

            

            if date not in list(values['has_date']):



                values.loc[len(values)] = [date, term['term'], 0]

                

        values = values.sort_values(by=['has_date'])

        

        plt.plot(list(values['has_date']), list(values['has_score']), label=term['term'])



    

    if ti > 5 + offset:

        

        break

        

plt.xlabel('Date')

plt.ylabel('Count')

plt.xticks(rotation=45)

plt.legend()

plt.show()