import json 

import os

articles = {}

# Convert string to Python dict 

for dirpath, subdirs, files in os.walk('/kaggle/input'):

    for x in files:

        if x.endswith(".json"):

            articles[x] = json.loads(open(os.path.join(dirpath, x)).read().lower())

more_than_1_abs = 0

for key in articles.keys():

    if len(articles[key]['abstract']) > 1:

        more_than_1_abs += 1

print(more_than_1_abs)



more_than_1_babs = 0

for key in articles.keys():

    if len(articles[key]['body_text']) > 1:

        more_than_1_babs += 1

print(more_than_1_babs)
abstracts_dct = {}

incr = 1

for key in articles.keys():

    if len(articles[key]['abstract']) == 1:

        #print(incr)

        abstracts_dct[key] = articles[key]['abstract'][0]['text']

        incr += 1

    else:

        local_iter = 0

        for lst in articles[key]['abstract']:

            abstracts_dct[key + str(local_iter)] = articles[key]['abstract'][local_iter]['text']

            local_iter += 1

        incr += 1

#abstracts_dct

print('The {} jsons had abstract text and processed into dictionary: abstracts_dct'.format(incr))
body_text_dct = {}

incr = 1

for key in articles.keys():

    if len(articles[key]['body_text']) == 1:

        #print(incr)

        body_text_dct[key] = articles[key]['body_text'][0]['text']

        incr += 1

    else:

        local_iter = 0

        for lst in articles[key]['body_text']:

            body_text_dct[key + str(local_iter)] = articles[key]['body_text'][local_iter]['text']

            local_iter += 1

        incr += 1



print('The {} jsons had body text and processed into dictionary: body_text_dct'.format(incr))
import pickle



with open('abstracts.pickle', 'wb') as f:

    pickle.dump(abstracts_dct, f)



with open('detail_ext.pickle', 'wb') as f:

    pickle.dump(body_text_dct, f)
import pickle

with open('/kaggle/input/covid-19/abstracts.pickle','rb') as f:

    abstracts_dct = pickle.load(f)

    

with open('/kaggle/input/covid-19/detail_ext.pickle','rb') as f:

    body_text_dct = pickle.load(f)