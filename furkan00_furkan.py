v_message = "hi"



v_name = "furkan ahmet"

v_surname = "ergün"

v_fullname = v_name + " " + v_surname



v_var1 = "100"

v_var2 = "200"

v_VarSum = v_var1 + v_var2



print(v_message)

print()



print(v_fullname)

print()



print("v_VarSum" , v_VarSum)
print(len(v_fullname))

print( )



#.title komutu seçilen parametrenin içeriğindeki kelimelerin ilk elemanlarını büyültür

print(v_fullname.title())

print( )



#.upper komutu seçilen parametrenin içeriğindeki kelimelerin tüm elemanlarını büyültür

print(v_fullname.upper())

print( )

 

#.lower komutu seçilen parametrenin içeriğindeki kelimelerin tüm elemanlarını küçültür 

print(v_fullname.lower())

print( )



print("type of v_fullname is" , type (v_fullname))
v_char0 = v_fullname[0]

v_cahr1 = v_fullname[1]

v_cahr2 = v_fullname[2]

v_cahr3 = v_fullname[3]

v_cahr4 = v_fullname[4]

v_cahr5 = v_fullname[5]

v_cahr6 = v_fullname[6]

v_cahr7 = v_fullname[7]

v_cahr8 = v_fullname[8]

v_cahr9 = v_fullname[9]



print(v_char0 , v_cahr1 , v_cahr2 , v_cahr3 , v_cahr4 , v_cahr5 , v_cahr7 , v_cahr8 , v_cahr9)
v_num1 = 10

v_num2 = 20

v_Sumnum = v_num1 + v_num2



print("v_num1 : " , v_num1 , " and type : " , type(v_num1))



print()

print("Sum of Num1 and Num2 is : " , v_Sumnum , " and type : " , type(v_Sumnum))

#def komutunu kullanırden , komuttan sonra kendiuydurduğumuz kodu ,girer parantez aç  kapa yapar ve ikili (:) nokta koyarız

#ardından alt satıra geçtiğimizde bi boşlu olur bu boşlu "def" komutu yüzündendir "print" yazıp komutun içine istediğimiz veriyi yükleriz 
#---ÖRNEK---

def v_selam():

    print("merhaba efendim")

    

v_selam()
v_name = "furkan ahmet"

v_surname = "ergün"

v_age="16"

v_fullname = v_name +" "+ v_surname

print(v_fullname)

print(v_age)
#def print (yani yazdırma ) komutunu ortadan kaldıran komut



def v_konuşma():

    print("bugün hava cok güzel")

    

def v_konuşma2():

    print("cumartesi tarih ödevi var")

    print("iste bişeyler")
v_konuşma()

v_konuşma2()
#type komutu bize kodun türünü gösterir

print("v_hızlımesaj type is :" , type(v_hızlımesaj))
def v_hızlımesaj(v_mesaj1):

    print (v_mesaj1 , " merhaba sizi geri arayacagım :  'v_hızlımesaj' " )

    

v_hızlımesaj("kusura bakmayın")

v_hızlımesaj("suan müsaıt deyilim")
def f_sayMessage(v_Message1):

    print(v_Message1 , " where are you coming from"" 'f_sayMessage'")
f_sayMessage("How are you ?")
def v_kısatanıtım(v_FirstName , v_Surname , v_lv):

    print("selam" , v_FirstName , " " , v_Surname , " levelin : " , v_lv)
v_kısatanıtım("furkan ahmet" , "ergün" , 1 )
def v_clac1(v_num1 , v_num2 , v_num3):

    v_sonuc=v_num1+v_num2+v_num3

    print("sonuç = " , " " , v_sonuc)



v_clac1 (60,60,60)
#return komutu fonksyona değer verebilmemizi sağlar
def v_clac2(f_num1 , f_num2 , f_num3):

    v_out = f_num1+f_num2+f_num3*2

    print("selam adamım v_clac2")

    return v_out

    
v_gelen = v_clac2(1,2,5)

print ("puanın:" , v_gelen)
def v_telesekreter(v_numara,v_arayan,v_hızlımesaj2="merhaba sizi geri arayacagım"):

    print("numara : " , v_numara , "arayan : " , v_arayan , "hızlımesaj : " , v_hızlımesaj2)
v_telesekreter("xxx xxx xx xx   " , "ahmet bey   ")

v_telesekreter("xxx xxx xx xx   " , "ahmet bey   " , "çok üzgünüm müsait değilim ahmet bey. Ben size dönerim")
def v_gönderi1(v_Name , *v_mesaj):

    print("selam " , v_Name , " oto ilk mesaj : " , v_mesaj[1])
v_gönderi1("furkan" , "Selam" , "Naber" , "İyisindir İnşallah")
v_result1 = lambda x : x*3

print("Result is : " , v_result1(10))
def v_alan(kenar1,kenar2):

    print(kenar1*kenar2 )
v_alan(15,15)
l_list1 =  [1,2,3,4,5,6,7,8,9]

print (l_list1)

print ("l_list1 type is" , type(l_list1))
l_list1_4 = l_list1[3]

print(l_list1_4)

print("l_list1_4 type is :" , type(l_list1_4))
l_list2 = ["türkiye" , "almanya" , "meksika" , "alasya"]

print(l_list2)

print("l_list2 type is :" , type(l_list2))
l_list2_4 = l_list2[3]

print (l_list2_4)

print ("l_list2_4 type is :" , type (l_list2_4))
l_list2_x2 = l_list2[-2]

print(l_list2_x2)
l_list2_1x3 = l_list2[1:3]

print(l_list2_1x3)
v_len_l_list2_1x3 = len(l_list2_1x3)

print("size of l_list2_1x3 :" , v_len_l_list2_1x3)

print(l_list2_1x3)
#ekleme

l_list2_1x3.append("türkiye")

print(l_list2_1x3)

l_list2_1x3.append("suriye")

print(l_list2_1x3)
print(l_list2_1x3)
#tersten sarma

l_list2_1x3.reverse()

print(l_list2_1x3)
#alfabetik sıra

l_list2_1x3.sort()

print(l_list2_1x3)
l_list2_1x3.append("suriye")

print(l_list2_1x3)
#slime

l_list2_1x3.remove("suriye")

print(l_list2_1x3)
d_dict1 = {"TRY":"TL" , "USD":"$" , "EUR":"€" , "GBP":"	£"}

print(d_dict1)

print("d_dict1 type is:" , type(d_dict1))
v_türkiye = d_dict1["TRY"]

print(v_türkiye)

print(type(v_türkiye))
#keys & values



v_keys = d_dict1.keys()

v_values = d_dict1.values()



print(v_keys)

print(type(v_keys))



print(" ")



print(v_values)

print(type(v_values))