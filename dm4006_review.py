f=open(file="../input/movies.txt",mode='r',encoding="latin-1")
f.seek(0)
data=f.read(4000000000)
import gc
len(data)
del f
gc.collect()
new_data=data.split("\n")
del data
gc.collect()
sumary="review/text: "
sum_len=len(sumary)
f_new=open("review.txt",'w')
for line in new_data:
        if line.startswith(sumary):
            str=line[sum_len:]
            f_new.write(str+"\n")            
f_new.close()
del new_data
gc.collect()
