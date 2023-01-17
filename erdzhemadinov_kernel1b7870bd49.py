import numpy as np

import pandas as pd

import tqdm



import catboost as cat

from catboost import CatBoostClassifier



import numpy as np

from numpy import zeros

from collections import Counter

import operator

from functools import reduce



#from getpreprocess import GetPreprocess

import os

print(os.listdir("../input/tochka-data"))
%%time

edges = pd.read_csv('../input/tochka-data/edges.csv')

ids = pd.read_csv('../input/tochka-data/ids.csv')

vertices = pd.read_csv('../input/tochka-data/vertices.csv')
vertices['main_okved'] = vertices['main_okved'].astype(str)
np.random.seed(7777)
edges.shape
edges.head()
id_1 = edges['id_1'].tolist()

id_2 = edges['id_2'].tolist()
from gensim.models import Word2Vec



emb_l = []





for  index, i in enumerate(id_1):

    if index % 500000 == 0:

        print(index/ len(id_1))

    emb_l.append([str(id_1[i]), str(id_2[i]) ]) 



model = Word2Vec(emb_l, min_count=1, size= 100, workers=6, window = 2, sg = 1)
model['76064']
def cosine_distance (model, word,target_list , num) :

    cosine_dict ={}

    word_list = []

    a = model[word]

    for item in target_list :

        if item != word :

            try:

                b = model [item]

                cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

                cosine_dict[item] = cos_sim

            except:

                pass

    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 

    for item in dist_sort:

        word_list.append((item[0], item[1]))

    return word_list[0:num]
maker_mod = list(set(id_1 + id_2))





for i in range(0, len(maker_mod)):

    maker_mod[i] = str(maker_mod[i])
cosine_distance (model,'169982', maker_mod, 1000)
dicti = {}

for i in edges['id_1'].tolist():

    if i in dicti.keys():

        dicti[i] += 1

    else:

        dicti[i] = 1

for i in edges['id_2'].tolist():

    if i in dicti.keys():

        dicti[i] += 1

    else:

        dicti[i] = 1
freq = {}



count = []



total = 0



for i in ids.id:

    total += dicti[i]

    

    count.append(dicti[i])



check = 0

for index, i in enumerate(ids.id):

    

    freq[i] = int(int((10000000*(count[index]/total)))/100)

    check += freq[i]



    

#bug    

freq[524354] += 54  

check+= 54

print(check)
# for index, i in enumerate(freq.keys()):

#     print("#", index, freq[i], i)

first_column = []

second_column = []
verticles_list = vertices.id.tolist()



#set(ids.id).intersection(result.id_1)

ids_list = list(ids.id)





for i in freq.keys():

    freq[i] = 1000
for i in tqdm.tqdm(ids.id):

    cnt = freq[i]

    cnt+=1

    #print(i/len(ids.id))

    for example in cosine_distance (model, str(i), maker_mod, freq[i] + 200 ):

        

#         print(edges[ ((edges['id_1'] == int(example[0])) & (edges['id_2'] == i)) | 

#                   ((edges['id_2'] == int(example[0])) & (edges['id_1'] == i)) ].shape[0],  cnt)

        

        if (edges[ ((edges['id_1'] == int(example[0])) & (edges['id_2'] == i)) |

                  ((edges['id_2'] == int(example[0])) & (edges['id_1'] == i)) ].shape[0] == 0) & (cnt > 0) &  (int(example[0]) in verticles_list) &  (i in verticles_list) :

            cnt -= 1

            first_column.append(int(example[0]))

            second_column.append(i)

            if cnt%100 ==0:

                print("Estimate: ", cnt, " id: ", i)

        if cnt ==0:

            break

            

        

#         if (edges.loc[ (edges.id_1 == int(example[0]) & edges.id_2 == i) | 

#                  (edges.id_2 == int(example[0]) & edges.id_1 == i) ].shape[0] > 1):

#             print(id_1, id_2)









d = {'id_1': first_column, 'id_2': second_column}



result = pd.DataFrame(data=d)

result = result[~result.id_1.isin(ids.id)].head(100000)
result.shape
result.head()
# # для каждой вершины из ids с помощью catboost найдем 1000 самых вероятных ребер

# for i in tqdm.tqdm(ids.id):

#     # соберем датасет из всех возможных вершин

#     # вершины имеющие в исходных данных ребро с i обозначим 1, остальные 0

#     # учтем то, что вершина i может быть как среди id_1, так и среди id_2

#     df1 = edges[edges['id_1'] == i].reset_index()

#     df2 = edges[edges['id_2'] == i].reset_index()



#     df = df1[['id_2', 'id_1']].rename(columns={'id_1':'id_2', 'id_2':'id_1'}).append(df2[['id_1', 'id_2']])

#     df['target'] = 1

    

#     df = vertices.set_index('id').join(df.set_index('id_1')['target']).fillna(0)

    

    

#     X = df[['main_okved', 'region_code', 'company_type']]

#     y = df['target']

    

#     model = CatBoostClassifier(iterations=150, verbose=True)

#     cat_features = [0,1,2] # все признаки категориальные

    

#     model.fit(X, y, cat_features)



#     preds = model.predict_proba(X)[:,1]



#     df['preds'] = preds

#     df['id_2'] = i

#     print(df.head)

#     # возьмем первую 1000 предсказанных ребер, исключив те, про которые мы уже знали

#     res = df[df['target'] != 1].sort_values(by='preds', ascending=False).iloc[:1250].reset_index()[['id', 'id_2']]

#     res.columns = ['id_1', 'id_2']

    

#     result = result.append(res, ignore_index=True, sort=False)
result.to_csv('submission.csv', index = False)