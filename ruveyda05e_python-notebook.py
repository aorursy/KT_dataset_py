_nm="rUvEyDa"

_surnm="eYİmaya"

_fullnm=_nm+" "+_surnm

print("Full Name:",_fullnm)

print()

print("lower of full name:",_fullnm.lower())

print()

print("title of full name:",_fullnm.title())

print()

_m1="hallo alle zusammen"

print(_m1.upper())
_s1="20"

_s2="04"

_sums1=_s1+_s2

print("Year of birth:",_sums1)

print("Age:",len(_fullnm))

print("_s1:",_s1,"and type",type(_s1))

      
_s3=0.65

_s4=0.95

_sums2=_s3+_s4

_chr1=_m1[15]

print("My height is ",_sums2,_chr1)

print("_s3 :",_s3,"and type :",type(_s3))
def f_thank(_nm):

    print(_nm,"Thank you for your help")

    print("EYVALLAH")
f_thank("tugce")
def f_kayıt(_nm,_tesis,_dal,_tarih):

    print("Sn ",_nm," basvuru yaptığınız",_tesis,"-",_dal,

         " üye kaydınız",_tarih,"a kadar gerekli belgeleri getirdiğiniz taktirde yapılacaktır.")
f_kayıt("Rüveyda EYİMAYA","Hamza Yerlikaya Spor Kompleksi","Yüzme","09.09.2019")
def f_kayıt(_nm,_tesis,_dal,_tarih="09.09.2019"):

    print("Sn ",_nm," basvuru yaptığınız",_tesis,"-",_dal,

         "üye kaydınız",_tarih ,"a kadar gerekli belgeleri getirdiğiniz takdirde yapılacaktır")
f_kayıt("Rüveyda EYİMAYA ","Hamza Yerlikaya Spor Kompleksi","yüzme")
def f_krktr(k1,k2,k3):

    _formul=k1+k2+k3

    print("hesap1'den selam!")

    return _formul
_aland=f_krktr(3,5,7)

print("üçgenin çevresi :",_aland)
def f_message1(_name,*_messages):

    print("Selam",_name,"Lütfen",_messages[2])

    
f_message1("Ayse","eve gel","okula git","gelirken ekmek al")

_sonuç=lambda x:x*7

print("sonuç",_sonuç(5))
def f_alan(k1,k2):

    print(k1*k2)
f_alan(8,9)
list_1=["matematik ","fizik ","kimya ","biyoloji "]

print("Sayısal dersler :",list_1)
list_2=["edebiyat ","tarih ","cografya ","din "]

print("Sözel dersler :",list_2)
print("type of list_1 is :",type(list_1))
print("    sayısal ders sayısı :",len(list_1))

print(list_1)

print("    sözel ders sayısı :",len(list_2))

print(list_2)
list_1.append("tarih ")

print("en sevdiğim dersler :",list_1)
list_2.append("tarih ")

print(list_2)
list_2.reverse()

print(list_2)
list_2.remove("tarih ")

print(list_2)
list_2.sort()

print(list_2)

d_telno = {"ahmet":5506769787,"mehmet":5405658676,

          "ayşe":5304547565,"fatma":5203436454}

print(d_telno)

print("type of d_telno is :", type (d_telno))

print()

print("lenght of d_telno is",len(d_telno),

      "und type of lenght is",type(len(d_telno)))
_nmsK=d_telno.keys()

_numsV=d_telno.values()



print(_nmsK)

print()

print(_numsV)
_yas1=12

_yas2=18

if _yas1 > _yas2 :

        print("kişi reşittir")

elif _yas1 < _yas2:

        print("kişi reşit değildir")

else:

        print("kişi reşit olmuşur ")
def f_yas(_yas1,_yas2):

    if _yas1 > _yas2:

        print("kişi reşittir")

    elif _yas2 > _yas1:

        print("kişi reşit degildir")

    else:

        print("kişi reşit olmuştur")

        

f_yas=(40,18)

f_yas=(10,18)
def f_ıncluderornot(_search,_searchlist):

    if _search in _searchlist :

        print(_search ," is  in tel.list .")

    else:

        print(_search , "is not in tel.list , sorry .")

        

l_telnolist = list(_nmsK)

print(l_telnolist)

print()

print(type(l_telnolist))



f_ıncluderornot("ahmet" , l_telnolist)

f_ıncluderornot("ayse" , l_telnolist)
for n_numbers in range(1,10):

    print(n_numbers , "bir rakamdır")

    
_msg ="ich liebe dich mama"

print(_msg)
for _m in _msg :

    print(_m)

    print("  <3 ")
for _m in _msg.split():

    print(_m)
l_list1=[5,4,3,2,1,9,8,7,6,15]

print(l_list1)

print()

print()

_sum_num =sum(l_list1)

print("listedeki sayıların toplamı", _sum_num )

print()









_cumlist1=0

_loopindex=0



for _current in l_list1:

    _cumlist=_cumlist+ _current

    print( _loopindex , ". degisken",

          _current , "dir")

    print( "toplam : " , _cumlist)

    _loopindex= _loopindex +1 
g=0

while(g<4):

    if g==0:

        print("bu derse girmedin")

    else:

        print(g, ". derse geç kaldın")

    g=g+1
print(l_list1)

print()



g=0

k=len(l_list1)



while (g<k):

    print(l_list1[g])

    g=g+1