for x in range(23,30):

    print(x,"girebilir")
msg="SELAMUN ALEYKUM"

print(msg)
for msg2 in msg:

    print(msg2)

    print("••••••")
for msg3 in msg.split():

    print(msg3)
list1=[1,2,3,4,5,6,7,8,9]

v_sumlist=sum(list1)

print("sum of list1 is:",v_sumlist)



print()

v_cumlist1=0

v_loopindex=0

for v_current in list1:

    v_cumlist1=v_cumlist1 + v_current

    print(v_loopindex, ". degisken",v_current,"dir")

    print("Cumulative is :",v_cumlist1)

    v_loopindex=v_loopindex+1

    

    
i =0

while(i<4):

    print("hi",i)

    i=i+3
print(list1)

print()



i=0

k=len(list1)



while(i<k):

    print(list1[i])

    i=i+1
vmin=-1

vmax=0



index=0

vlen=len(list1)



while (index<vlen):

    current=list1[index]

    

    if current > vmax:

        vmax=current

        

    if current < vmin:

        vmin=current

        

    index=index+1

    

    

print("max is",vmax)

print("min is", vmin)