print("Hello World")
v_message ="hello world"

print("Hi")
print(v_message)
v_name ="aysegul"

v_surname ="bucak"



v_fullname = v_name + v_surname

print(v_fullname)
v_fullname = v_name + " " + v_surname



print(v_fullname)
v_num1 = "200"

v_num2 = "300"

v_numSum1 = v_num1 + v_num2

print(v_numSum1)
#length

v_lenFull = len(v_fullname) 

print("v_fullname :" ,v_fullname," and lenght is : ",v_lenFull)
v_titleF = v_fullname.title()

print("v_fullname :", v_fullname ," and title is : " , v_titleF)
#upper :

v_upperF = v_fullname.upper()



#lower

v_lowerF = v_fullname.lower()

print("v_fullname : " , v_fullname , " Upper : " , v_upperF , "L")
v_2ch = v_fullname[12]

print(v_2ch)
v_num1 = 200

v_num2 = 300

v_sum1 = v_num1 + v_num2



print(v_sum1 , " and type : " , type(v_sum1))
#it will get error

#v_sum2 = v_num1 + v_name

#print(v_sum2)
v_num1 = v_num1 + 58

v_num2 = v_num2 - 24.4

v_sum1 = v_num1 + v_num2



print(v_num1)
print("v_sum1 : ",v_sum1 , "type : ", type(v_sum1))
v_fl1 = 24.4

v_fl2 = 14.6

v_s3 = v_fl1 + v_fl2



print(v_s3 , type(v_s3))
def f_SayHello():

    print("Hi. I am coming from f_SayHello")

    

def f_SayHello2():

    print("Hi. I am coming from f_SayHello2")

    print("Good")

    

f_SayHello()
f_SayHello2()
def f_sayMessage(v_Message1):

    print(v_Message1 , " came from 'f_sayMessage'")

    

def f_getFullName(v_FirstName , v_Surname , v_Age):

    print("Welcome " , v_FirstName , " " , v_Surname , " your age : " , v_Age)
f_sayMessage("How are you ?")
f_getFullName("Aysegul" , "BUCAK" , 16)
def f_Calc1(f_Num1 , f_Num2 , f_Num3):

    v_Sonuc = f_Num1 + f_Num2 + f_Num3

    print("Sonuç = " ," " , v_Sonuc)
f_Calc1(80 , 100 , 180)
#return function

def f_Calc2(v_Num1 , v_Num2 , v_Num3):

    v_Out =v_Num1 + v_Num2 + v_Num3*2

    print("Hi from f_Calc2")

    return v_Out
v_gelen = f_Calc2(1,2,3)

print("Score is : " , v_gelen)
#Default Functions :

def f_populationInfo(v_city,v_population,v_country = "TURKEY"):

    print("City : ", v_city , "Population :", v_population, "Country :",v_country)
f_populationInfo("Sivas","646.608")

f_populationInfo("Moskow","12.500.000","RUSSIA")
#Flexible Functions :



def f_Flex1(v_Name , *v_colors):

    print("Hi ", v_Name , "your favorite color is : " , v_colors[5])
f_Flex1("Aysegul" , "Blue","Purple","Brown","Green","Black","Gray") 
# Lambda Function :



v_result = lambda x : x*9

print("Result is :",v_result(58))
def f_total(v_edge1,v_edge2,v_edge3,v_edge4):

    v_perimeter =(v_edge1 + v_edge2 + v_edge3 + v_edge4)

    v_area = (v_edge1*v_edge3)

    print(v_perimeter + v_area)
f_total(9,9,6,6)
v_list1 = [3,6,9,7,8,2]

print(v_list1)

print("Type of 'v_list1' is : " , type(v_list1))
v_number1 = v_list1 [4]

print(v_number1)

print("Type of 'v_number1' is : " , type (v_number1))
v_planet1 = ["Mercury","Venus","Earth","Mars","jupiter","Saturn","Uranus","Neptune","Pluto"]

print(v_planet1)

print("Type of 'v_planet1' is : " , type(v_planet1))
v_planet1_2 = v_planet1[1]

print(v_planet1_2)

print("Type of 'v_planet1_2' is : " , type(v_planet1_2))
v_planet1_c1 = v_planet1 [-1]

print(v_planet1_c1)
v_planet_c2 = v_planet1 [:7]

print(v_planet_c2)
#Len

v_len_planet1 = len(v_planet1)

print("Size of 'v_planet1_c2' is : " , v_len_planet1)

print(v_planet1)
#Append

v_planet1.append ("Moon")

print(v_planet1)



v_planet1.append ("Venus")

print(v_planet1)
#Reverse

v_planet1.reverse()

print(v_planet1)
#Sort

v_planet1.sort()

print(v_planet1)
#Remove



#First add 'Neptune' then remove 'Neptune'



v_planet1.append ("Neptune")

print(v_planet1)
v_planet1.remove("Neptune")

print(v_planet1)
v_satellite = {"Earth":"Moon" , "Jupiter":"Callisto" , "Saturn":"Titan" , "Uranus":"Miranda" ,"Neptun":"Triton"}



print(v_satellite)

print(type(v_satellite))
v_Saturn = v_satellite["Saturn"]

print(v_Saturn)

print(type(v_Saturn))
#Keys & Valeus



v_keys = v_satellite.keys()

v_values = v_satellite.values()





print(v_keys)

print(type(v_keys))



print()

print(v_values)

print(type(v_values))
v_num1 = 98

v_num2 = 58



if v_num1 > v_num2:

    print(v_num1 , " is greater then " , v_num2)

    

elif v_num1 < v_num2:

    print(v_num1 , " is smaller then " , v_num2)

    

else :

    print("This 2 variables are equal")
# < , <= , > , >= , == , <> 



def v_Comparison1(v_num1 , v_num2):

    

    

    if v_num1 > v_num2:

        print(v_num1 , " is greater then " , v_num2)

        

    elif v_num1 < v_num2:

        print(v_num1 , " is smaller then " , v_num2)

        

    else : 

        print("These " , v_num1 , " variables are equal")

        

v_Comparison1(17,17)

v_Comparison1(45,78)

v_Comparison1(67,58)
# using 'IN' with LIST





def s_search (v_search , v_searchList):

    

    if v_search in v_searchList :

        print("Yeap! ",v_search , " is in list :)")

        

    else : 

        print( v_search , " is not in list :( ")

        

v_list = list(v_satellite.keys())

print(v_list)

print(type(v_list))



s_search("Saturn" , v_list)

s_search("Venus" , v_list)
for a in range(0,9):

    print( a , " İHTİZA King Of VEHİC")
v_best_queen = "ERMEDA IS MY BEST QUEEN"

print(v_best_queen)
for v_down in v_best_queen:

    print(v_down)

    print("_______")
for v_down in v_best_queen.split():

    print(v_down)
v_list6 = [9,6,7,16,3,58]

print(v_list6)
print(v_list6)

v_sum_list6 = sum(v_list6)

print("Sum of v_list6 is : " , v_sum_list6)



print()

v_cum_list6 = 0

v_loopindex = 0



for v_current in v_list6:

    

    v_cum_list6 = v_cum_list6 + v_current

    

    print(v_loopindex , " nd value is : " , v_current)

    print("Cumulative is : " , v_cum_list6)

    

    v_loopindex = v_loopindex + 1

    print("______")
h = 0 

while(h < 6):

    print("Glowing Fire" , h)

    h = h+1
print(v_list6)

print()



h = 0

k= len(v_list6)



while(h < k):

    print(v_list6[h])

    h = h+1
#Let's find minimum and maximum number in list



v_list7 = [9,13,8,-17,58,-271]



v_min = 0

v_max = 0



v_index = 0

v_len = len(v_list7)



while ( v_index < v_len):

    v_current = v_list7[v_index]

    

    if v_current > v_max:

        v_max = v_current

        

    if v_current < v_min:

        v_min = v_current

        

    v_index = v_index+1

    

print ("Max. number is : " , v_max)

print ("Min. number is : " , v_min)
#Import library to use



import numpy as np
v_array1 = [9,6,7,8,4,5,3,2,1,0,5,8,9,7,6,2,3,2,1,0]



v_array1_np =np.array([9,6,7,8,4,5,3,2,1,0,5,8,9,7,6,2,3,2,1,0])
print("v_array1 : " , v_array1)

print("Type of v_array1 : " , type(v_array1))
print("v_array2_np : " , v_array2_np)

print("Type of v_array1_np : " ,type(v_array1_np))

# Shape

v_shape1 = v_array2_np.shape

print("v_shape1 : " , v_shape1 , " and type is : " , type (v_shape1))
#Reshape

v_array3_np = v_array2_np.reshape(4,5)

print(v_array2_np)
v_shape2 =v_array3_np.shape

print("v_shape2 : " , v_shape2 , " and type is : " , type(v_shape2) )
#Dimension

v_dimen1 = v_array3_np.ndim

print("v_dimen : " , v_dimen , " type is : " , type(v_dimen))
# Dtype.name

v_dtype1 = v_array3_np.dtype.name

print("v_dtype : " , v_dtype , " and type is : " , type(v_dtype))
# Size

v_size1 = v_array3_np.size

print("v_size : " , v_size , " and type : " , type(v_size))
#Let's create 4x5 array

v_array4_np = np. array([[9,6,7,8,4],[5,3,2,1,0],[5,8,9,7,6],[2,3,2,1,0]])

print(v_array4_np)

print("_________")

print("Shape is : " , v_array4_np.shape)
# Let's do it with string







v_array_str = (['Merdüm i dîdeme', 'bilmem ne füsûn etti felek.',

'Giryemi kıldı füsûn', 'eşkimi hûn etti felek.',

'Şirler pençe i ','kahrımda olurken lerzân.',

'Beni bir gözleri âhûya', 'zebûn etti felek.',

               

'Bilmem ki gözlerime', 'nasıl bir büyü yaptı felek',

'Gözümü kan içinde bırakıp', 'aşkımı artırdı felek',

'Arslanlar pençemin', 'korkusundan tir tir titrerken',

'Beni bir gözleri ahuya', 'esir etti felek.'])



v_array_str = np.array (['Merdüm i dîdeme', 'bilmem ne füsûn etti felek.',

'Giryemi kıldı füsûn', 'eşkimi hûn etti felek.',

'Şirler pençe i', 'kahrımda olurken lerzân.',

'Beni bir gözleri âhûya', 'zebûn etti felek.',

                        

'Bilmem ki gözlerime', 'nasıl bir büyü yaptı felek',

'Gözümü kan içinde bırakıp', 'aşkımı artırdı felek',

'Arslanlar pençemin', 'korkusundan tir tir titrerken',

'Beni bir gözleri ahuya', 'esir etti felek.'])

print("v_array_str : " , v_array_str)

print("Type of v_array_str : " , type(v_array_str))
print("v_array_str : " , v_array_str)

print("Type of v_array_str: " , type(v_array_str))
v_shape_str = v_array_str.shape

print("v_shape_str : " , v_shape_str , " and type is : " , type(v_shape_str))
v_array_str = v_array_str.reshape(4,4)

print(v_array_str)

print(" YAVUZ SULTAN SELİM HAN")
#Zeros

v_array5_np = np.zeros((9,6))

print(v_array_np)



type(v_array5_np)
#Update an item on this array

v_array5_np[2,2] = 4

print(v_array5_np)
#Ones



v_array6_np = np.ones((9,6))

print(v_array6_np)
#Empty

v_array7_np = np.empty((6,3))

print(v_array7_np)
#Arrange

v_array8_np = np.arange(16,96,4)

print(v_array8_np)

print(v_array8_np.shape)
#Linspace



v_array9_np = np.linspace(10,100,2)

v_array9_np = np.linspace(9,81,20)



print(v_array9_np)

print(v_array9_np.shape)

print("_________")

print(v_array9_np)

print(v_array9_np.shape)
# Sum , Substract , Square

v_np = np.array([9,6,3])

v_np1 = np.array([7,5,2])



print(v_np - v_np1)

print(v_np + v_np1)

print(v_np1 - v_np)

print(v_np ** 3)
# Sinus



print(np.sin(v_np))
# True/False

v_np_TF = v_np < 9

print(v_np_TF)

print(v_np_TF.dtype.name)
# Element Wise Prodcut



v_np = np.array([9,6,3])

v_np = np.array([5,7,2])



print(v_np * v_np1)