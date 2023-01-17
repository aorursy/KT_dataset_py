%pip install jsondiff
import requests
from jsondiff import diff
aux_arr = []
def search(link, arr):

    r = requests.get(link)
    arr.append(r.json()['items'])
    if 'last' in r.links:
        print(r.links['next']['url'])
        search(r.links['next']['url'], arr)
    return arr

ret = search('https://api.github.com/search/repositories?q=covid19&sort=stars&order=desc&page=1&per_page=100', aux_arr)

ret[1][0]
# https://developer.github.com/v3/#pagination
# https://tools.ietf.org/html/rfc5988
r = requests.get('https://api.github.com/search/repositories?q=covid19&sort=stars&order=desc&page=33')
print([r.links, r.links['last']['url']])
#r.json()
# https://developer.github.com/v3/#pagination
# https://tools.ietf.org/html/rfc5988
#r2 = requests.get('https://api.github.com/search/repositories?q=corona&sort=stars&order=desc')
r2 = requests.get('https://api.github.com/search/repositories?q=sars-cov-2&sort=starts&order=desc')
#print(r2.json()['total_count'])


class Filters:
    def __init__():
        pass
    
def filter(arr):
    
    # metadata structure
    id = 'id'
    full_name = 'full_name'
    url = 'url'
    license = 'license'
    language = 'language'
    watchers_count = 'watchers_count'
    forks = 'forks'
    created_at = 'created_at'
    updated_at = 'updated_at'
    
    return [{
        id : e[id],
        full_name : e[full_name],
        url : e[url],
        license : e[license]['key'] if e[license] is not None else None,
        language : e[language],
        watchers_count : e[watchers_count],
        forks : e[forks],
        created_at : e[created_at],
        updated_at : e[updated_at]
        }
        #for e in response.json()['items']
        for e in arr
    ]


def filter2(arr, keys):
    obj = {
    }
    for e in keys:
        aux_arr = []

        lambda2(arr, aux_arr, e)
        obj[e] = aux_arr

    return obj
    
lambda3 = lambda arr, key : [e[key] for e in arr]
lambda2 = lambda arr, aux_arr, key :  [ aux_arr.extend(lambda3(e, key)) for e in arr ]
#filter2(dict.fromkeys(keys2, []), ret)
#test = dict.fromkeys(keys2, [])
arr_aux =[]
#print(keys2)
#lambda2([filter(e) for e in ret], arr_aux)
result_final = filter2([filter(e) for e in ret], list(keys2))
train = pd.DataFrame.from_dict(result_final)
train.to_csv('repos1.csv', index=False)
#check differences between results
#diff(r.json()['items'], r2.json()['items']).__len__()
#filter(ret[0])
import pandas as pd

keys2 = [filter(e) for e in ret][0][0].keys()
values = values_lambda(keys2, [filter(e) for e in ret][0][0])
print(keys2)
#print(values)
#print(ret[0][0])
def to_data_frame(keys, arr):
    
    for e in arr:
        for e2 in e:
            aux_arr.extend(values_lambda(keys, e2))
    return aux_arr
#print(aux_arr)
#to_data_frame(keys2, [filter(e) for e in ret])
#df = pd.DataFrame([values], columns = keys2)
#df

print(ret[0][0])
#arr
#[filter(e) for e in ret]

def has_licenses(count, arr):
    aux_arr = []
    for e in arr:
        for i in e:
            if i['license'] != None:
                aux_arr.append(i['license'])
    
    return aux_arr
    
licenses_arr = has_licenses(0, [filter(e) for e in ret])
print(licenses_arr.__len__())
obj = dict.fromkeys(licenses_arr, 0)

def count_licenses(arr, obj):
    for e in arr:
        for i in e:
            if i['license'] != None:
                obj[i['license']] += 1
    return obj

obj_ret = count_licenses([filter(e) for e in ret], obj)
print(obj_ret)
keys = obj_ret.keys()
values_lambda = lambda arr, obj : [ obj[e] for e in arr]
values = values_lambda(keys, obj_ret)
print(keys,values)
list(zip(keys,values))
import matplotlib
import matplotlib.pyplot as plt
import json
from pandas import pandas
obj_ret = dict(sorted(obj_ret.items(), key=lambda x: x[1], reverse=True))
keys = list(obj_ret.keys())
values = [obj_ret[keys[i]] for i in range(keys.__len__())]

df = pandas.Series(obj_ret)
#df = pandas.DataFrame({"License" : keys, "Count" : values})
plt.figure(figsize=(12,8))
df.plot(kind='bar')
#df.rename(columns={3: "Label", 4: "Country", 5: "Industry", 6: "Website"})
#df.to_csv('licenses.csv')
df = pandas.DataFrame({"License" : keys, "Count" : values})

#df.sort_values('Count',ascending=False).drop(labels=None, axis=1)
df.to_csv('licenses1.csv', index=False)

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.set_index.html