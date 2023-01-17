b_list1 = [9,8,7,6,5,4]

print(b_list1)

print("Type of 'b_list1' is : " , type(b_list1))
b_list1_4 = b_list1[3]

print(b_list1_4)

print("Type of 'b_list1_4' is : " , type(b_list1_4))
b_list2 = ["Tank","Support","Dmg","Flank"]

print(b_list2)

print("Type of 'b_list2' is : " , type(b_list1))
b_list2_4 = b_list2[3]

print(b_list2_4)

print("Type of 'b_list2_4' is : " , type(b_list2_4))
b_list2_x3 = b_list2[-1]

print(b_list2_x3)
b_list2_2 = b_list2[0:2]

print(b_list2_2)
#Len

v_len_b_list2_2 = len(b_list2_2)

print("Size of 'b_list2_2' is : ",v_len_b_list2_2)

print(b_list2_2)
#Append

b_list2_2.append("Unknow class")

print(b_list2_2)



b_list2_2.append("Dmg")

print(b_list2_2)
#Reverse

b_list2_2.reverse()

print(b_list2_2)
#Sort

b_list2_2.sort()

print(b_list2_2)
#Remove



#First add 'Dmg' then Remove 'Dmg'

b_list2_2.append("Dmg")

print(b_list2_2)
b_list2_2.remove("Dmg")

print(b_list2_2)