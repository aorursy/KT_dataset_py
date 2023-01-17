import requests



def guardian_search(api_key, search_query):

    url = 'http://content.guardianapis.com/search?q='

    query = search_query.replace(' ', '%20') #removing spaces

    final_url = url + query + '&api-key=' + api_key #composing the URL we want to run

    response = requests.get(final_url) #collects the url

    data = response.content.decode('utf-8') #this gets it into a readable format

    return data
guardian_search('7fdfdabf-472f-4198-bfa3-89020e56707e', 'code')
guardian_search('7fdfdabf-472f-4198-bfa3-89020e56707e', 'banksy and london')
def guardian_search_version2(api_key, search_query, from_date):

    url = 'http://content.guardianapis.com/search?q='

    query = search_query.replace(' ', '%20') 

    final_url = url + query + '&from-date=' + from_date + '&page-size=100' + '&api-key=' + api_key

    response = requests.get(final_url)

    data = response.content.decode('utf-8')

    return data
guardian_search_version2('7fdfdabf-472f-4198-bfa3-89020e56707e', 'banksy and london', '2020-06-15')
example_json = guardian_search_version2('7fdfdabf-472f-4198-bfa3-89020e56707e', 'banksy and london', '2020-06-15')



import json



example = json.loads(example_json) 
print(example['response']['results'])
results_example = example['response']['results']



import pandas as pd



df = pd.DataFrame(results_example, columns=['id', 'type', 'sectionId', 'sectionName'])



print(df)
def guardian_search_wordcount(api_key, search_query, from_date):

    url = 'http://content.guardianapis.com/search?q='

    query = search_query.replace(' ', '%20') 

    final_url = url + query + '&from-date=' + from_date + '&show-fields=wordcount' + '&page-size=100' + '&api-key=' + api_key

    response = requests.get(final_url)

    data = response.content.decode('utf-8')

    return data
print(guardian_search_wordcount('7fdfdabf-472f-4198-bfa3-89020e56707e', 'banksy and london', '2020-06-15'))
wordcount_example_json = guardian_search_wordcount('7fdfdabf-472f-4198-bfa3-89020e56707e', 'banksy and london', '2020-06-15')



wordcount_example = json.loads(wordcount_example_json)



results_wordcount_example = wordcount_example['response']['results']



df_wordcount = pd.DataFrame(results_wordcount_example, columns=['id', 'type', 'sectionId', 'sectionName', 'fields'])



df_wordcount['wordcount'] = df_wordcount['fields'].astype(str).str.replace('{\'wordcount\': \'','').str.replace('\'}','')



del df_wordcount['fields']



print(df_wordcount)
def guardian_search_wordcount_formatted(api_key, search_query, from_date, output_type):

    url = 'http://content.guardianapis.com/search?q='

    query = search_query.replace(' ', '%20') 

    final_url = url + query + '&from-date=' + from_date + '&show-fields=wordcount' + '&page-size=100' + '&api-key=' + api_key

    response = requests.get(final_url)

    data1 = response.content.decode('utf-8')

    data2 = json.loads(data1)

    data3 = data2['response']['results']

    final_data = pd.DataFrame(data3, columns=['id', 'type', 'sectionId', 'sectionName', 'fields'])

    final_data['wordcount'] = final_data['fields'].astype(str).str.replace('{\'wordcount\': \'','').str.replace('\'}','')

    del final_data['fields']

    if (output_type == 'DataFrame'):

        return final_data

    else:

        return data1



df_wordcount = guardian_search_wordcount_formatted('7fdfdabf-472f-4198-bfa3-89020e56707e', 'banksy and london', '2020-06-15', output_type='DataFrame')



print(df_wordcount)