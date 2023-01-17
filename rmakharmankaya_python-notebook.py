print("hello kaggle")
v_message = "hello"



v_name = "Irmak"

v_surname="harmankaya"

v_fullname=v_name+v_surname



v_var1="100"

v_var2="200"

v_varsum=v_var1+v_var2



print(v_message)

print()

print(v_fullname)

print()

print(v_varsum)

print("lenght of v_fullname:",len(v_fullname))

print()

print("full name is:",v_fullname.title())

print()

print("upper of full name is:",v_fullname.upper())

print()

print("lower of full name is:",v_fullname.lower())

print()

print("type of v_fullname:",type(v_fullname) )
v_chr1=v_fullname[3]

v_chr2=v_fullname[5]

print("v_chr1:",v_chr1,"v_chr2:",v_chr2)
v_num1= 8

v_num2= 2

v_numsum=v_num1+v_num2

print("v_num1:",v_num1,"and type",type(v_num1))

print("sum of v_num1 and v_num2:",v_numsum,"and type:",type(v_numsum))
v_num3=10.5

v_numsum=v_num1+v_num3

print("sum of v_num1 and v_num3:",v_numsum,"and type:",type(v_numsum))
def f_sayhello():

    print("hi.ı am coming from f_sayhello")

    

def f_sayhello2():

    print("hi.ı am coming from f_sayhello2")

    print("good")

    

f_sayhello()    
f_sayhello2()
def f_saymessage(v_message1):

    print(v_message1,"came from'f_saymessage'")

    

def f_getfullname(v_firstname,v_surname,v_age):

    print("welcome",v_firstname," ",v_surname,"your age:",v_age)
f_saymessage("how are you")
f_getfullname("ırmak","harmankaya",15)
def f_calc1(f_num1,f_num2,f_num3):

    v_sonuç=f_num1+f_num2+f_num3

    print("sonuç=",v_sonuç)
f_calc1(10,20,30)
def f_calc2(v_num1,v_num2,v_num3):

    v_out=v_num1+v_num2+v_num3*2

    print("hi from f_calc2")

    return v_out
v_gelen=f_calc2(1,2,3)

print("score is:",v_gelen)
def f_getschoolinfo(v_name,v_studentcount,v_city="istanbul"):

    print("name:",v_name,"st count",v_studentcount,"city:",v_city)
f_getschoolinfo("aaihl",521)

f_getschoolinfo("ankara fen",521,"ankara")
def f_flex1(v_name,*v_messages):

    print("hi",v_name,"your first message is:",v_messages[2])
f_flex1("ırmak","selam","naber","iyisindir inşallah")
v_result=lambda x:x*3

print("result is:",v_result(6))
def f_alan(kenar1,kenar2):

    print(kenar1*kenar2)
f_alan(5,6)
l_list1=[1,5,3,6,8,9]

print(l_list1)

print("type of'l_list1'is:",type(l_list1))
v_list1_4=l_list1[3]

print(v_list1_4)

print("type of'v_list1_4'is:",type(v_list1_4))
l_list2 = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

print(l_list2)

print("type of 'l_list2'is :",type(l_list1))
v_list2_4 = l_list2[3]

print(v_list2_4)

print("type of 'v_list2_4'is:",type(v_list2_4))

v_list2_x3 = l_list2[-3]

print(v_list2_x3)
l_list2_2 = l_list2[0:3]

print(l_list2_2)
v_len_l_list2_2 = len(l_list2_2)

print("size of 'l_list2_2'is:", v_len_l_list2_2)

print(l_list2_2)
l_list2_2.append("saturday")

print(l_list2_2)



l_list2_2.append("tuesday")

print(l_list2_2)
l_list2_2.reverse()

print(l_list2_2)
l_list2_2.sort()

print(l_list2_2)
l_list2_2.append("saturday")

print(l_list2_2)
l_list2_2.remove("saturday")

print(l_list2_2)
d_dict1 = {"home":"ev","school":"okul","student":"öğrenci"}

print(d_dict)

print(type(d_dict1))
v_school = d_dict1["school"]

print(v_school)

print(type(d_dict1))

v_keys = d_dict1.keys()

v_values = d_dict1.values()



print(v_keys)

print(type(v_keys))



print()

print(v_values)

print(type(v_values))


    