file="../input/wp_3gram.txt"
import pandas as pd
import gc
f=open(file, 'r', encoding='latin-1')
f.seek(6000000000)
data = f.read(1000000000)
f.close()
del f
gc.collect()
no_word=len(data)
print("no of word in file :",no_word)

x=data.split("\n")
del data
gc.collect()
length=len(x)
print("no of list",length)
x_new=x
del x
gc.collect()
gram_value=3
dist = dict()
for line in x_new:
    y= line.split("\t")
    try:
        dist[' '.join(y[1:])] = int(y[0])
    except:
        print("ignore")
del x_new
gc.collect()

distpd_1= pd.DataFrame(list(dist.items()), columns=['bigram', 'frequency'])
del dist
gc.collect()
distpd_1.to_csv("3_gram_7.csv")
