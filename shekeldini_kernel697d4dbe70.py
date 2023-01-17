from math import *
a=[24,27,26,21,20,31,26,22,20,18,30,29,24,26]
b=[100,115,117,119,134,94,105,103,111,124,122,109,110,86]
a_aver=[]
b_aver=[]
a_aver_in_kvd=[]
b_aver_in_kvd=[]
result_a_aver_in_kvd=0
result_b_aver_in_kvd=0
result_A=0
result_B=0
multiply_aver=[]
sum_multiply_aver=0
#1
for i in a:
    result_A+=i

for i in b:
    result_B+=i
#2    
average_A=result_A/len(a)
average_B=result_B/len(b)

#3
for i in a:
    a_aver.append(average_A-i)
for i in b:
    b_aver.append(average_B-i)

#4
 
for i in a_aver:
    a_aver_in_kvd.append(i*i)
for i in b_aver:
    b_aver_in_kvd.append(i*i)


#5    
for i in a_aver_in_kvd:
    result_a_aver_in_kvd+=i

for i in b_aver_in_kvd:
    result_b_aver_in_kvd+=i

   
#6    
for x,y in zip (a_aver,b_aver):
    multiply_aver.append(x*y)

for i in multiply_aver:
    sum_multiply_aver+=i
result=sum_multiply_aver/sqrt(result_a_aver_in_kvd*result_b_aver_in_kvd)
result