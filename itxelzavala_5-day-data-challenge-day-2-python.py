import numpy as np 
import pandas as pd 
import collections
import matplotlib.pyplot as plt

file = '../input/job_skills.csv'

df = pd.read_csv(file)
df.describe()
top_Company = collections.Counter(df['Company'])
top_Company
maximum = max(top_Company, key=top_Company.get) 
maximum
list_ = df['Category']#SEPARAR PALABRAS TOMAS LAS PRIMERAS DOS LETRAS PARA CONSTRUIR UN ID UNICO DE CATEGORIA
list_id = []
for x in list_:
    y = x.split()
    var1 = ''
    for _ in range(len(y)):
        var1 += (y[_][0:1])
    list_id.append(var1)
    
list_id

df['ID_Category'] = list_id
df.head()
data_filter = df[df['Company'] == maximum]
data_filter.head()
plt.hist(data_filter['ID_Category'], bins='auto', histtype ='step')
plt.title("Histogram of Category of the job of " + maximum)
plt.xlabel('Categories of the job')
plt.ylabel('Frequency')