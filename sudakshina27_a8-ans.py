d = {"foo" : 42 }

print(d)
keys = ['Ten', 'Twenty', 'Thirty']

values = [10, 20, 30]

d = {} 

for key in keys: 

    for value in values: 

        d[key] = value 

        values.remove(value) 

        break  

print ("Resultant dictionary is : " +  str(d)) 
dict1 = {'Ten': 10, 'Twenty': 20, 'Thirty': 30}

dict2 = {'Thirty': 30, 'Forty': 40, 'Fifty': 50}

dict1.update(dict2)

print(dict1)
sampleDict = {'a': 100, 'b': 200, 'c': 300}

200 in sampleDict.values()
r = {'c1': 'violet', 'c2': 'indigo', 'c3': 'blue', 'c4': 'green', 'c5': 'yellow', 'c6': 'orange', 'c8': 'red'}

r['c7'] = r['c8']

print(r)
Data = {'Physics': 82,'Math': 65,'history': 75 }

v = list(Data.values())

k = list(Data.keys())

print(k[v.index(min(v))]) 