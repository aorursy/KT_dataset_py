f=open(file="../input/movies.txt",mode='r',encoding="latin-1")
sumary="review/summary:"
sum_len=len(sumary)
f_new=open("review_sumary.txt",'w')
for line in f.readlines():
        if line.startswith(sumary):
            str=line[sum_len:]
            f_new.write(str)            
f_new.close()
f.close()
ff=open("review_sumary.txt",'r')
