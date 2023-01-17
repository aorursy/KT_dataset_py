print("hello world")
v_message = "hello world"

print("hi")
print(v_message)
v_name= "ilayda"

v_surname="yılmaz"

v_fullname= v_name + v_surname

print(v_fullname)
v_fullname=v_name + " " + v_surname

print(v_fullname)
v_num1="200"

v_num2="300"

v_numSum1=v_num1 + v_num2

print(v_numSum1)
#length

v_lenFull = len(v_fullname)

print("v_fullname:", v_fullname,"and length is:", v_lenFull)
#upper:

v_upperF = v_fullname.upper()

#lower

v_lowerF = v_fullname.lower()

print("v_fullname:", v_fullname, "uper:", v_upperF, "Lover:", v_lowerF)
v_2ch = v_fullname[11]

print(v_2ch)
v_num1 = 150

v_num2= 200

v_sum1= v_num1 + v_num2

print(v_sum1, "and type:" , type(v_sum1))

#it will get errror

#v_sum2 = v_num1 + v_name

print(v_sum2)
v_num1 = v_num1+ 50

v_num2 = v_num2 + 25.5

v_sum1 = v_num1 + v_num2

print(v_sum1)
print("v_sum1:", v_sum1,"type:", type(v_sum1))
v_fl1= 25.5

v_fl2=25.5

v_s3=v_fl1+ v_fl2

print(v_s3, "type:",type(v_s3))
def f_SayHello():

    print("hello, i am reading web design")

    

def f_SayHello2():

    print("I want artificial intelligence engineering")

    print("Good")

f_SayHello()

f_SayHello2()
def f_sayMessage(a_Message1):

    print(a_Message1 , " came from 'f_sayMessage'")

    

def f_getFullName(a_FirstName , a_Surname , a_Age):

    print("Welcome " , a_FirstName , " " , a_Surname , " your age : " , a_Age)

    
f_sayMessage("How are you ?")
f_getFullName("ilayda" , "yılmaz" , 16)
def f_Calc1(f_Num1 , f_Num2 , f_Num3):

    a_Sonuc = f_Num1 + f_Num2 + f_Num3

    print("Sonuç =", a_Sonuc)
f_Calc1(10 , 20 , 50)

def f_Calc2(a_Num1 , a_Num2 , a_Num3):

    a_Out = a_Num1+a_Num2+a_Num3*2

    print("Hi from f_Calc2")

    return a_Out

    
a_gelen =  f_Calc2(1,2,3)

print("Score is : " , a_gelen)
def f_getSchoolInfo(a_Name,a_StudentCount,a_City = "KAHRAMANMARAŞ"):

    print("Name : " , a_Name , " St Count : " , a_StudentCount 

          , " City : " , a_City)
f_getSchoolInfo("AKSU ANADOLU LİSESİ" , 269)

f_getSchoolInfo("RÜŞTÜ AKIN MTAL" , 432 , "İSTANBUL")
def f_Students1(v_isim, v_soyisim, v_ayakkabıNum=38):

    print("bilgileriniz:",v_isim, "" , v_soyisim, "is:", v_ayakkabıNum)

f_Students1("ilayda","yılmaz")

f_Students1("sevde", "hacıosmanoğlu",37)
def f_pi(v_Radius):

    v_pi = 5 * 3.14 * v_Radius

    return v_pi



v_Circle1 = f_pi(5)

print("Reference is : " , v_Circle1)
def f_SayMessage2(v_Name , *v_args):

    print("Hi ", v_Name , " Your 2nd message is : " , v_args[1])



f_SayMessage2("İlayda" , "Message 1", "Message 2", "Message 3", "Message 4")
v_result1 = lambda x : x*3

print("result is:", v_result1(9))


def f_alan(kenar1,kenar2):

    print(kenar1*kenar2)

f_alan(5,12)

sayilar =[2,4,6,8]

print(sayilar)

print("çift sayılar:",type(sayilar))
sayilar1 = sayilar[2]

print(sayilar1)

print("sayılar'ın tipi:",type(sayilar1))
sevdigimsarkicilar=["tuğkan","emircan iğrek", "gökşin derin","çağan şengül","yasir miy"]

print(sevdigimsarkicilar)

print("sevdiğim sanatçılar:",type(sevdigimsarkicilar))
sevdigimsarkicilar1 = sevdigimsarkicilar[0]

print(sevdigimsarkicilar1)

print("en sevdiğim şarkıcı:",type(sevdigimsarkicilar1))
sarkicilar = sevdigimsarkicilar[-2]

print(sarkicilar)
sarkicilar2 = sevdigimsarkicilar[0:2]

print("en çok dinlediklerim:",sarkicilar2)
len_sarkicilar = len (sevdigimsarkicilar)

print(sevdigimsarkicilar)

print("sevdiğim kaç şarkıcı var?",len_sarkicilar)
sevdigimsarkicilar.append("nil ipek")

print(sevdigimsarkicilar)
sevdigimsarkicilar.reverse()

print(sevdigimsarkicilar)
sevdigimsarkicilar.sort()

print(sevdigimsarkicilar)
sevdigimsarkicilar.append("tuğkan")

print(sevdigimsarkicilar)
sevdigimsarkicilar.remove("tuğkan")

print(sevdigimsarkicilar)
d_dict1 = {"artificial":"yapay" , "intelligence" : "zeka" , "engineer": "mühendisi"}



print(d_dict1)

print(type(d_dict1))
v_engineer = d_dict1["engineer"]

print(v_engineer)

print(type(v_engineer))
v_keys = d_dict1.keys()

v_values = d_dict1.values()





print(v_keys)

print(type(v_keys))



print()

print(v_values)

print(type(v_values))
sayi1 = 30

sayi2=5



if sayi1 > sayi2:

    print(sayi1 , " is greater then " ,sayi2)

elif sayi < sayi:

    print(sayi , " is smaller then " , sayi2)

else :

    print("This 2 variables are equal")
def f_Comparison1(v_Comp1 , v_Comp2):

    if v_Comp1 > v_Comp2:

        print(v_Comp1 , " is greater then " , v_Comp2)

    elif v_Comp1 < v_Comp2:

        print(v_Comp1 , " is smaller then " , v_Comp2)

    else :

        print("These " , v_Comp1 , " variables are equal")

        

f_Comparison1(22,33)

f_Comparison1(55,88)

f_Comparison1(77,99)
def f_IncludeOrNot(v_search, v_searchList):

    if v_search in v_searchList :

        print("Good news ! ",v_search , " is in list.")

    else :

        print(v_search , " is not in list. Sorry :(")



l_list = list(d_dict1.keys())

print(l_list)

print(type(l_list))



f_IncludeOrNot("artificial" , l_list)

f_IncludeOrNot("yazılım" , l_list)
for a in range(0,31):

    print("doğum gününüze " , a , "gün kalmıştır")
v_succeedMessage = "I can succeed"

print(v_succeedMessage)
for v_chrs in v_succeedMessage:

    print(v_chrs)

    print("-------")

for v_chrs in v_succeedMessage.split():

    print(v_chrs)
l_list1=[1,5,7,10]

print(l_list1)

v_sum_list1 = sum(l_list1)

print("Sum of l_list1 is : " ,  v_sum_list1)

print()

v_cum_list1 = 0

v_loopindex = 0

for v_current in l_list1:

    v_cum_list1 = v_cum_list1 + v_current

    print(v_loopindex , " nd value is : " , v_current)

    print("Cumulative is : " , v_cum_list1)

    v_loopindex = v_loopindex + 1

    print("------")
i = 1

while(i < 17):

    print(i,"yıl daha yaşlandınız")

    i = i+1
print(l_list1)

print()



i = 0

k = len(l_list1)



while(i<k):

    print(l_list1[i])

    i=i+1
l_list2 = [14,15,16,17,-1]

print("kursa gelen kişilerin yaşları")

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



print ("En büyük öğrenci: " , v_max,"yaşındadır")

print ("En küçük öğrenci: " , v_min,"yaşındadır")
import numpy as np
v_array1 = [3,5,7,10,13,16,19,22,25,28,30]

v_array2_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print("v_array1 : " , v_array1)

print("Type of v_array1 : " , type(v_array1))
v_shape1 = v_array2_np.shape

print("v_shape1 : " , v_shape1 , " and type is : " , type(v_shape1))
v_array3_np = v_array2_np.reshape(3,5)

print(v_array3_np)
v_shape2 = v_array3_np.shape

print("v_shape2 : " , v_shape2 , " and type is : " , type(v_shape2))
v_dimen1 = v_array3_np.ndim

print("v_dimen1 : " , v_dimen1 , " type is : " , type(v_dimen1))
v_dtype1 = v_array3_np.dtype.name

print("v_dtype1 : " , v_dtype1 , " and type is : " , type(v_dtype1))
v_size1 = v_array3_np.size

print("v_size1 : " , v_size1 , " and type : " , type(v_size1))
v_array4_np = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(v_array4_np)

print("---------------")

print("Shape is : " , v_array4_np.shape)
v_array5_np = np.zeros((3,4))

print(v_array5_np)
v_array5_np[0,3] = 27

print(v_array5_np)
v_array6_np = np.ones((3,4))

print(v_array6_np)
v_array7_np = np.empty((4,4))

print(v_array7_np)
v_array8_np = np.arange(10,45,5)

print(v_array8_np)

print(v_array8_np.shape)
v_array9_np = np.linspace(5,20,7)

v_array10_np = np.linspace(10,20,7)



print(v_array9_np)

print(v_array9_np.shape)

print("-----------------------")

print(v_array10_np)

print(v_array10_np.shape)
print(np.sin(v_np2))
v_np1 = np.array([15,7,3])

v_np2 = np.array([2,4,1])

print(v_np1 * v_np2)
v_np5 = np.array([[5,7,11],[1,6,4]])

v_np5Transpose = v_np5.T

print(v_np5)

print(v_np5.shape)

print()

print(v_np5Transpose)

print(v_np5Transpose.shape)
v_np6 = v_np5.dot(v_np5Transpose)

print(v_np6)
v_np6 = v_np5.dot(v_np5Transpose)

print(v_np6)
v_np5Exp = np.exp(v_np5)



print(v_np5)

print(v_np5Exp)
v_np8 = np.random.random((6,6))

print(v_np8)
v_np8Sum = v_np8.sum()

print("Sum of array : ", v_np8Sum) 

print("Max of array : ", v_np8.max()) 

print("Min of array : ", v_np8.min())
print("Sum of Columns :")

print(v_np8.sum(axis=0)) 

print()

print("Sum of Rows :")

print(v_np8.sum(axis=1)) 
print(np.sqrt(v_np8))

print()

print(np.square(v_np8))
v_np10 = np.array([1,4,9,2])

v_np11 = np.array([1,2,3,4])



print(np.add(v_np10,v_np11))
v_np12 = np.array([5,4,9,7,1,6,8])



print("First item is : " , v_np12[0])

print("Third item is : " , v_np12[5])
print(v_np12[0:4])
v_np12_Rev = v_np12[::-1]

print(v_np12_Rev)
v_np13 = np.array([[1,2,3,4,5],[11,12,13,14,15]])

print(v_np13)

print()

print(v_np13[1,3]) 



print()

v_np13[1,3] = 314

print(v_np13)
print(v_np13[:,2])
print(v_np13[1,1:4])
print(v_np13[-1,:])
print(v_np13[:,-1])
#Flatten

v_np14 = np.array([[5,4,3],[2,1,4],[9,8,7],[14,15,16]])

v_np15 = v_np14.ravel()



print(v_np14)

print("Shape of v_np14 is : " ,v_np14.shape)

print()

print(v_np15)

print("Shape of v_np15 is : " ,v_np15.shape)

print()
v_np16 = v_np15.reshape(3,4)

print(v_np16)

print("Shape of v_np16 is : " ,v_np16.shape)
v_np17 = v_np16.T

print(v_np17)

print("Shape of v_np17 is : " ,v_np17.shape)
v_np20 = np.array([[2,4],[6,8],[10,12]])



print(v_np20)

print()

print(v_np20.reshape(2,3))



print()

print(v_np20) 
v_np20.resize((2,3))

print(v_np20)
v_np21 = np.array([[5,4],[3,2]])

v_np22 = np.array([[8,9],[6,5]])



print(v_np21)

print()

print(v_np22)
v_np23 = np.vstack((v_np21,v_np22))

v_np24 = np.vstack((v_np22,v_np21))



print(v_np23)

print()

print(v_np24)
# Horizontal Stack

v_np25 = np.hstack((v_np21,v_np22))

v_np26 = np.hstack((v_np22,v_np21))



print(v_np25)

print()

print(v_np26)
v_list1 = [7,8,9,10]

v_np30 = np.array(v_list1)



print(v_list1)

print("Type of list : " , type(v_list1))

print()

print(v_np30)

print("Type of v_np30 : " , type(v_np30))
v_list2 = list(v_np30)

print(v_list2)

print("Type of list2 : " , type(v_list2))
v_list3 = v_list2

v_list4 = v_list2



print(v_list2)

print()

print(v_list3)

print()

print(v_list4)
v_list2[0] = 27

print(v_list2)

print()

print(v_list3) 

print()

print(v_list4)
v_list5 = v_list2.copy()

v_list6 = v_list2.copy()



print(v_list5)

print()

print(v_list6)
v_list2[0] = 15



print(v_list2)

print()

print(v_list5) 

print()

print(v_list6)
import pandas as pd
v_dict1 = { "ÜLKE" : ["TÜRKİYE","TÜRKİYE","TÜRKİYE","TÜRKİYE","TÜRKİYE"],

            "ŞEHİR":["ISTANBUL","KAHRAMANMARAŞ","ADANA","İZMİR","MANİSA"],

            "PLAKA":[34,46,1,35,45]}



v_dataFrame1 = pd.DataFrame(v_dict1)



print(v_dataFrame1)

print()

print("Type of v_dataFrame1 is : " , type(v_dataFrame1))
v_head1 = v_dataFrame1.head()

print(v_head1)

print()

print("Type of v_head1 is :" ,type(v_head1))
print(v_dataFrame1.head(100))
v_tail1 = v_dataFrame1.tail()

print(v_tail1)

print()

print("Type of v_tail1 is :" ,type(v_tail1))
v_columns1 = v_dataFrame1.columns

print(v_columns1)

print()

print("Type of v_columns is : " , type(v_columns1))
v_info1 = v_dataFrame1.info()

print(v_info1)

print()

print("Type of v_info1 is : " , type(v_info1))
v_dtypes1 = v_dataFrame1.dtypes

print(v_dtypes1)

print()

print("Type of v_dtypes1 is : " , type(v_dtypes1))
v_descr1 = v_dataFrame1.describe()

print(v_descr1)

print()

print("Type of v_descr1 is : " , type(v_descr1))
v_country1 = v_dataFrame1["ŞEHİR"]

print(v_country1)

print()

print("Type of v_country1 is : " , type(v_country1))
v_currenyList1 = ["TRY","TRY","TRY","TYR","TYR"]

v_dataFrame1["ŞEHİR"] = v_currenyList1



print(v_dataFrame1.head())
v_AllCapital = v_dataFrame1.loc[:,"ÜLKE"]

print(v_AllCapital)

print()

print("Type of v_AllCapital is : " , type(v_AllCapital))
v_top3Currency = v_dataFrame1.loc[0:3,"PLAKA"]

print(v_top3Currency)
v_CityCountry = v_dataFrame1.loc[:,["ÜLKE","ŞEHİR","PLAKA"]] 

print(v_CityCountry)
v_Reverse1 = v_dataFrame1.loc[::-1,:]

print(v_Reverse1)
print(v_dataFrame1.loc[:,:"ŞEHİR"])

print()

print(v_dataFrame1.loc[:,"ŞEHİR":])
print(v_dataFrame1.iloc[:,2])
v_filter1 = v_dataFrame1.PLAKA > 5

print(v_filter1)
v_filter2 = v_dataFrame1["PLAKA"] < 9

print(v_filter2)
print(v_dataFrame1[v_filter1 & v_filter2])
print(v_dataFrame1[v_dataFrame1["ŞEHİR"] == "TYR"])
v_meanPop =v_dataFrame1["PLAKA"].mean()

print(v_meanPop)



v_meanPopNP = np.mean(v_dataFrame1["PLAKA"])

print(v_meanPopNP)
for a in v_dataFrame1["PLAKA"]:

    print(a)
v_dataFrame1["POP LEVEL"] = ["Low" if v_meanPop > a else "HIGH" for a in v_dataFrame1["PLAKA"]]

print(v_dataFrame1)
print(v_dataFrame1.columns)



v_dataFrame1.columns = [a.lower() for a in v_dataFrame1.columns]



print(v_dataFrame1.columns)
v_dataFrame1.columns = [a.split()[0]+"_"+a.split()[1] if (len(a.split())>1) else a for a in v_dataFrame1.columns]

print(v_dataFrame1.columns)