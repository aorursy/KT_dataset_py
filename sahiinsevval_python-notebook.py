v_message = "hello world"

print(v_message)



v_name = "sevval"

v_surname = "sahin" 

v_fullname = v_name + " " + v_surname

print(v_fullname)
v_num1 = "480"

v_num2 = "840"

v_sum1 = v_num1 + v_num2



print(v_sum1)



v_num1 = 480

v_num2 = 840

v_Sum1 = v_num1 + v_num2 



print(v_Sum1)
#len function:

print()

print("Lenght of v_fullname is" , len(v_fullname))



#title function:

print()

print("Full Name is" , v_fullname.title())



#upper function:

print()

print("Upper of Full Name is" , v_fullname.upper())



#lower function:

print()

print("Full Name is" , v_fullname.lower())



#type:

print()

print("Type of v_fullname is" , type(v_fullname))

v_chr1 = v_fullname[4]

v_chr2 = v_fullname[8]

print("v_chr1 : " , v_chr1 , "and v_chr2 : " , v_chr2)
#Integer

v_num3 = 40

v_num4 = 20

v_sum1 = v_num3 + v_num4

print(v_sum1)



print("v_num3 : " , v_num3 , "and type : " , type(v_num3))



print()

print("Sum of Num3 and Num4 is : " , v_sum1 , "and type : " , type(v_sum1))
#Float

v_num6 = 20.8

v_Sum2 = v_num6 + v_num3



print("Sum of Num6 and Num3 is : " , v_Sum2 , "and type : " , type(v_Sum2))
def w_SayHello():

    print("Hi. The message coming from Europa")



def w_sayhello():

    print("stay away from our planet")

    print("we dont lıke people")

    

w_SayHello()
w_sayhello()
def w_message(x_message1):

    print(x_message1 , "came from 'aliens'")

    

def w_Fullname(v_name , v_surname , v_color):

    print("Welcome" , v_name , " " , v_surname, "color of your eyes" ,  v_color)

    

    
w_message("wie geht's?")
w_Fullname("sevval" , "sahin" , "black")
def f_blabla(f_num1 , f_num2 , f_num3):

    v_Sonuc = f_num1 + f_num2 + f_num3

    print("Sonuç = " ," " , v_Sonuc)
f_blabla( 480 , 600 , 842)
# return function

def f_blabla2(v_num1 , v_num2 , v_num3):

    v_Out = v_num1+v_num2+v_num3*2

    print("Hi from f_blabla2")

    return v_Out
v_gelen =  f_blabla2(2,3,5)

print("Score is : " , v_gelen)
# Default Functions :

def x_okulbilgileri(v_name,v_surname,v_city = "Istanbul"):

    print("Name : " , v_name , " Surname : " , v_surname  , " City : " , v_city)
x_okulbilgileri('sevval','sahin')

x_okulbilgileri('nursima','sevinc','konya')
# Flexible Functions :



def f_Flex(v_Name , *v_messages):

    print("Hi " , v_Name , " your first message is : " , v_messages[1])
f_Flex('Sevval' , 'was machst du?' , 'ruf bitte mich an')
# Lambda Function :



v_result = lambda z : z*24

print("Result is : " , v_result(2))
def v_alan(kenar1,kenar2):

    print(kenar1*kenar2)

v_alan(8,6)


l_ökaryot = ["alg" , "amip" , "paramesyum" , "öglena" , "hayvan" , "bitki" , "mantar"]

print(l_ökaryot)

print("Type of 'l_ökaryot' is : " , type(l_ökaryot))
l_1 = l_ökaryot[2]



print(l_1),

print( "type of l_1 is : " , type(l_1)) 
l_asal = [2 , 3 , 5 , 7 , 11 , 13]

l_asal2 = l_asal[4]

print("type of l_asal2 is : " , type(l_asal2))
l_3 = l_ökaryot[-5]

print(l_3)
l_4 = l_ökaryot[2:6]

print(l_4)
#len

l_5 = len(l_4)

print("Size of l_4 is : ",l_5)

print(l_4)
#Append

l_4.append("cıvık mantar")

print(l_4)



l_4.append("öglena")

print(l_4)
#reverse

l_4.reverse()

print(l_4)

#sort

l_4.sort()

print(l_4)
#remove

l_4.remove("hayvan")

print(l_4)
d_1 = {"groß" : "big" , "süß" : "sweet" , "einfach" : "easy"}

print(d_1)

print(type(d_1))
v_adjectiv = d_1["groß"]

print(v_adjectiv)
#Keys & Values



v_keys = d_1.keys()

v_values = d_1.values()





print(v_keys)

print(type(v_keys))



print()

print(v_values)

print(type(v_values))
v_var1 = 13.58

v_var2 = 42.84



if v_var1 > v_var2:

    print(v_var1 , " is greater then " , v_var2)

elif v_var1 < v_var2:

    print(v_var1 , " is smaller then " , v_var2)

else :

    print("This 2 variables are equal")
# < , <= , > , >= , == , <>

def f_Comparison1(v_Comp1 , v_Comp2):

    if v_Comp1 > v_Comp2:

        print(v_Comp1 , " is greater then " , v_Comp2)

    elif v_Comp1 < v_Comp2:

        print(v_Comp1 , " is smaller then " , v_Comp2)

    else :

        print("These " , v_Comp1 , " variables are equal")

        

f_Comparison1(38,48)

f_Comparison1(66,876)

f_Comparison1(1124,11)
# using 'IN' with LIST





def f_IncludeOrNot(v_search, v_searchList):

    if v_search in v_searchList :

        print(v_search , " is in list.")

    else :

        print(v_search , " is not in list.")



l_list = list(d_1.keys())

print(l_list)

print(type(l_list))



f_IncludeOrNot("süß" , l_list)

f_IncludeOrNot("schwarz" , l_list)
for X in range(0,10):

    print("Hi " , X)
v_Message = "DON'T DO THİS "

print(v_Message)
for v_chrs in v_Message:

    print(v_chrs)

    print("*/**/*/**/*/**/*/**")
for v_chrs in v_Message.split():

    print(v_chrs)
v_list1 = [15 , 3963 , 586, 186 ,44 , 22015 , -12565]

print(v_list1)
print(v_list1)

v_sum_list1 = sum(v_list1)

print("Sum of v_list1 is : " , v_sum_list1)



print()

v_cum_list1 = 0

v_loopindex = 0

for v_current in v_list1:

    v_cum_list1 = v_cum_list1 + v_current

    print(v_loopindex , " nd value is : " , v_current)

    print("Cumulative is : " , v_cum_list1)

    v_loopindex = v_loopindex + 1

    print("------")
i = 0

while(i < 48):

    print("Hi" , i)

    i = i+1
print(v_list1)

print()



i = 0

k = len(v_list1)



while(i<k):

    print(v_list1[i])

    i=i+1
#Let's find minimum and maximum number in list



l_list2 = [5 , 36 , 896 , -12 , -7831 , -789]



v_min = 0

v_max = 0



v_index = 0

v_len = len(l_list2)



while (v_index < v_len):

    v_current = l_list2[v_index]

    

    if v_current > v_max:

        v_max = v_current

    

    if v_current < v_min:

        v_min = v_current

    

    v_index = v_index+1



print ("Maximum number is : " , v_max)

print ("Minimum number is : " , v_min)