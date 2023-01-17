for B in range(0,5):

    print("Hi " , B)
v_BoringMessage = "I AM BORED"

print(v_BoringMessage)
for b in v_BoringMessage:

    print(b)

    print("------")
for b in v_BoringMessage.split():

    print(b)
b_list1 = [1,3,5,7,9]

print(b_list1)

b_sum_list1 = sum(b_list1)

print("Sum of b_list1 is : " , b_sum_list1)



print()

b_cum_list1 = 0

b_loopindex = 0

for b_current in b_list1:

    b_cum_list1 = b_cum_list1 + b_current

    print(b_loopindex , " nd value is : " , b_current)

    print("Cumulative is : " , b_cum_list1)

    b_loopindex = b_loopindex + 1

    print("------")
B = 0

while(B < 9):

    print("Hi" , B)

    B = B+1
print(b_list1)

print()



b = 0

k = len(b_list1)



while(b<k):

    print(b_list1[b])

    b=b+1
#Let's find minimum and maximum number in list



l_list2 = [11,50,62,8445,6645964]

        

v_siralama = l_list2[0:4]



v_min = 0

if v_min > 0:

     v_min = v_siralama[0]

    

v_max = 0



v_index = 0

 



while (v_index < v_len):

    v_current = v_siralama[v_index]

    

    if v_current > v_max:

        v_max = v_current                                                                                              

    

    if v_current < v_min:

        v_min = v_current

    

    v_index = v_index+1

    



print ("Maximum number is : " , v_max)

print ("Minimum number is : " , v_min)
