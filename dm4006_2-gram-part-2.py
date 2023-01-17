file="../input/wp_2gram.txt"
import pandas as pd
import gc
f=open(file, 'r', encoding='latin-1')
#f.seek(0)
data = f.read()
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
x_new=x[50000000:]
del x
gc.collect()
dist = dict()
for line in x_new:
    y= line.split("\t")
    try:
        dist[' '.join(y[1:])] = int(y[0])
    except:
        print("ignore")
del x_new
gc.collect()

distpd_5= pd.DataFrame(list(dist.items()), columns=['bigram', 'frequency'])
del dist
gc.collect()
distpd_5.to_csv("2_gram_9.csv")
