# ODEV 1
print("Programming", "Essentials", "in", end="...", sep="*")
print("Python")

print()

#ODEV 2

#print(" "*4,"","\n"," "*2,"*"," ","*","\n", " "*7,"*","\n", " "*9,"*")

print(" "*7,"*","\n"," "*4,"*"," "*1,"*","\n"," "*2,"*"," "*5,"*","\n"," * * "," "*3," * * ","\n"," "*4,"*"," ","*" ,"\n"," "*4,"*"," ","*","\n"," "*4,"*","*","*")
# Parantezlerden birini silersem unexpected EOF while parsing error'unu döndürür.
# Çift tırnak yerine kesme işareti kullanırsam  EOL while scanning string literal error'unu döndürür. 
print()

print("I'm","learning","Python",end="\n")
print("\"i'm\"\n\"\"learning\"\"\n\"\"\"python\"\"\"")
      
      
      
print(' "ım" ' , ' ""learning"" ' ,  ' """python""" ' ,sep="\n" )
print(' "ım" ')
print(' ""learning"" ')
print(' """python"" ')

#ödev 1 
john= 3
mary= 5 
adem= 6
toplamelma= john+mary+adem

print("john elma sayısı:" , john, "mary elma sayısı:", mary, "adem elma sayısı:", adem)
print("toplam elma sayısı:" , toplamelma)
print("toplam elma sayısının mary'nin elma sayısına oranı:",toplamelma/mary)
print(john>mary)
print(john**mary**adem)

#ödev2
#1 milin yaklaşık 1,61 kilometreye eşit

kilometre=12.25
mil=7.38
mil_kilometre=7.38*1.61
kilometre_mil= 12.25*1.61
print(mil, "mil", round(mil_kilometre,2), "kilometredir.")
print(kilometre, "kilometre", round(kilometre_mil, 2 ), "mildir.")


#ödev3
tl=23
dolar=14
tl_dolar=23/6.90
dolar_tl=14*6.90

print(tl, "tl", round(tl_dolar, 2 ), "dolardır.")
print(dolar, "dolar", round(dolar_tl, 2 ), "tl dir.")

#ödev4
x= 1
x=float(1)
y= 3*(x**3)- 2*(x**2)+ 3*x-1
print("y=",y)
#ödev5
#bu program belirli bir saatteki saniye sayısını hesaplar
saatsayısı=2 # saat sayısı
saniye=3600 # 1 saatteki saniye sayısı
print("saat:" ,saatsayısı )
print("saatteki saniye:" ,saatsayısı*saniye ) #saatin içindeki saniye sayısının hesaplanması 