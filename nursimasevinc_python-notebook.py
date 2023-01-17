print('hello word')



v_message= 'Hello'



print()

v_name='nursima'

v_surname='sevinc'

v_fullname=v_name+" "+v_surname



v_num1='200'

v_num2='300'

v_num3=v_num1+v_num2



print(v_message)



print()

print('fullname')



print()

print('v_num3:',v_num3)
#len fonction:

print('Length of v_fullname is :', len(v_fullname))



#title function:

print()

print('Full Name is : ' , v_fullname.title())



#upper fonction

print()

print('upper of Fullname is: ',v_fullname.upper())



#lower function:

print()

print('Full Name is : ' , v_fullname.lower())



#type

print()

print('type of Fullname is:',type(v_fullname))





v_chr1=v_fullname[5]

v_chr2=v_fullname[10]

print('v_chr1 : ' , v_chr1 , 'and v_chr2 : ' , v_chr2)
#Ingeter

v_num4=10

v_num5=20

v_numSum=v_num4 + v_num5

print('v_num4:',v_num4,'and type:', type(v_num4))



print()

print('Sum of num4 and num5 is:' ,v_numSum, 'and type:' ,type(v_numSum))
#Float

v_numa=10.5

v_numb=v_numa+v_num5

print('v_numa:', v_numa, 'and type:',type(v_numa))

print('Sum of num5 and numa is:' , v_numb,'and type:',type(v_numb))
def f_komut():

    print('su şişesini eline al')

    print('kapağını aç')

    print('suyu iç')

    

f_komut()    
def f_message(v_message):

    print(v_message)



f_message('ich heiße Nursima')    

    



        
def f_sum1(v_num1,v_num2):

    f_sum2=v_num1-v_num2

    print(v_num1,'-',v_num2,'=',f_sum2)

    

f_sum1(20,10)

#Let's create calculator of circle circumreference function



#Functions cam give a value with RETURN



def f_CircleCircumreference(v_Radius):

    v_CircumRefenece = 3 * 5.15 * v_Radius

    return v_CircumRefenece



v_Circle1 = f_CircleCircumreference(2)

print("Reference is : " , v_Circle1)
#Default Function



def f_Students1(v_Name , v_Surname , v_ShoeSize = 15):

    print("Shoe size of ",v_Name , " " , v_Surname , " is : " , v_ShoeSize)

    

f_Students1("nursima","sevinç")

f_Students1("tuğçe","aydın",16)
#Flexible Functions :



def f_SayMessage2(v_Name , *v_args):

    print("Hi ", v_Name , " Your 2nd message is : " , v_args[1])



f_SayMessage2("nursima" , "hello world","hello")
# Lambda Function :



v_result1 = lambda x : x*5

print("Result is : " , v_result1(5))
print("Type of 'f_Students1' is : " , type(f_Students1))

print("Type of 'v_result1' is : " , type(v_result1))
planetnum=[1,2,3,4,5,6,7,8]

print(planetnum)

print('type of planetnum is:',type(planetnum) )
planetnum1=planetnum[5]

print(planetnum1)

print('type of planetnum1 is:',type(planetnum1))
planet=['merkür','venüs','dünya','mars','jüpiter','satürn','uranüs','neptün']

print(planet)

print('type of planet is:',type(planet))
whichplanet=planet[2]

print(whichplanet)

print('type of whichplanet is:',type(whichplanet))
planets=planet[-3]

print(planets)

print('type of planets is:',type(planets))
planets1=planet[0:7]

print(planets1)

print('type of planets1 is',type(planets1))
#len

lenplanet=len(planets1)

print('size of "lenplanet" is:',lenplanet)

print(planets1)
# append

planets1.append('neptün')

print(planets1)



planets1.append('pluton')

print(planets1)
#reverse

planets1.reverse()

print(planets1)
planets1.reverse()

print(planets1)
#sort

planets1.sort()

print(planets1)
#Remove



#First add 'dünya' then Remove 'dünya'



planets1.append('dünya')

print(planets1)



planets1.remove('dünya')

print(planets1)



planet_hacim={"satürn":"100","dünya":"10","pluton":"1"}

print(planet_hacim)

print(type(planet_hacim))
planet1=planet_hacim["satürn"]

print(planet1)

print(type(planet1))
planet_keys=planet_hacim.keys()





print(planet_keys)

print(type(planet_keys))



print()



planet_values=planet_hacim.values()

print(planet_values)

print(type(planet_values))

      
asteroid1=100

dünya=10



if asteroid1<dünya:

    print("füze ateşleme")

elif asteroid1>dünya:

    print("füze ateşle")

else:

    print("füze ateşle")

def planet(p_1 ,p_2):

    if p_1 > p_2:

        print(p_1 , " is greater then " , p_2)

    elif p_1 < p_2:

        print(p_1 , " is smaller then " ,p_2)

    else :

        print("These " , p_1 , " variables are equal")

        

planet(10,100)        

planet(200,20)

planet(30,30)
def planets(planet_s,planet_d):

    if planet_s in planet_d:

        print("satürn dünyayı içine alır") 

       

    else:

        print("dünya içine satürnü almaz")

        

planet_l=list(planet_hacim.keys())

print(planet_l)

print(type(planet_l))



planets("satürn",planet_l)

planets("ay",planet_l)



        

        

for a in range(0,10):

    print("planet " , a)
v_message = "my planet"

print(v_message)
for v_chrs in v_message:

    print(v_chrs)

    print('.......')
for v_chrs in v_message.split():

    print(v_chrs)

    
l_list1=[9,8,7,6,5]

print(l_list1)
print(l_list1)

v_sum_list1 = sum(l_list1)

print("Sum of l_list1 is : " , v_sum_list1)



print()

v_cum_list1 = 0

v_loopindex = 0

for v_current in l_list1:

    v_cum_list1 = v_cum_list1 + v_current

    print(v_loopindex , " nd value is : " , v_current)

    print("Cumulative is : " , v_cum_list1)

    v_loopindex = v_loopindex + 1

    print("------")
i = 0

while(i < 5):

    print("Hi" , i)

    i = i+1
print(l_list1)

print()



i = 0

k = len(l_list1)



while(i<k):

    print(l_list1[i])

    i=i+1
#Let's find minimum and maximum number in list



l_list2 = [5,10,200,-100,-300,999,-111]



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
# Import library to use

import numpy as np
v_array1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

v_array2_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print('v_array1:', v_array1)

print( 'type of v_array1:' ,type(v_array1))
print('v_array2_np:', v_array2_np)

print("type of v_array2_np:",type(v_array2_np))

#shape

v_shape1=v_array2_np.shape

print("v_shape1:",v_shape1,"type of v_shape1:",type(v_shape1))
#reshape

v_array3_np=v_array2_np.reshape(3,5)

print(v_array3_np)
v_shape2=v_array3_np.shape

print("v_sahpe2:", v_shape2,"type of v_shape2",type(v_shape2))
#dimension 

v_dimen1=v_array3_np.ndim

print("v_dimen1:", v_dimen1 ,"type of v_dimen1:", type(v_dimen1))
v_dimen2=v_array2_np.ndim

print("v_dimen2:", v_dimen2 ,"type of v_dimen2:", type(v_dimen2))
# Dtype.name

v_dtype1 = v_array3_np.dtype.name

print("v_dtype1 : " , v_dtype1 , " and type is : " , type(v_dtype1))
#size

v_size1=v_array3_np.size

print("v_size1:", v_size1 , "type of v_size1:", type(v_size1))
# Let's create 3x4 array

v_array4_np = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(v_array4_np)

print("_________________")

print("Shape is : " , v_array4_np.shape)
#zeros

v_array5_np=np.zeros((3,5))

print(v_array5_np)
# Update an item on this array 

v_array5_np[0,0]=1

print(v_array5_np)
#ones 

v_array6_np=np.ones((3,3))

print(v_array6_np)
# Empty

v_array7_np = np.empty((2,3))

print(v_array7_np)
#arrenge

v_arrange1 = np.arange(1,10,2)

print(v_arrange1)

print("shape of v_arrange1:",v_arrange1.shape)
#linspace

v_linspace1=np.linspace(1,20,3)

v_linspace2=np.linspace(1,20,10)



print(v_linspace1)

print(v_linspace1.shape)



print()



print(v_linspace2)

print(v_linspace2.shape)
