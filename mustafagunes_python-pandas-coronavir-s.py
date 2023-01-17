# data feıme kullanımın da hızlı ve etkilidir
#excell dosyalarında
#dosyalar arasında gecis kolaydır text vb
#missing datalarda işimizi kolaylastırır
#hızlı bir kutphanedır,

import pandas as pd
dictionary={"name":["ali","kenan","veli","ayse","memet","bulut"],"age":[10,25,45,35,74,36],
            "maas":[200,300,150,550,1250,450]}
#print(dictionary)
datafrıme=pd.DataFrame(dictionary)
print(datafrıme)

# head metotu
import pandas as pd
dictionary={"name":["ali","kenan","veli","ayse","memet","bulut"],"age":[10,25,45,35,74,36],
            "maas":[200,300,150,550,1250,450]}
datafrıme=pd.DataFrame(dictionary)
#print(datafrıme)
head=datafrıme.head()
print(head)
import pandas as pd
dictionary={"name":["ali","kenan","veli","ayse","memet","bulut"],"age":[10,25,45,35,74,36],
            "maas":[200,300,150,550,1250,450]}
datafrıme=pd.DataFrame(dictionary)
tail=datafrıme.tail() #burada parantez içerisine hangi degeri girersek o kadar elamanı getırır
print(tail)
tail2=datafrıme.tail(100) #burada parantez içerisine hangi degeri girersek o kadar elamanı getırır
print(tail2)
import pandas as pd
dictionary={"name":["ali","kenan","veli","ayse","memet","bulut"],"age":[10,25,45,35,74,36],
            "maas":[200,300,150,550,1250,450]}
datafrıme=pd.DataFrame(dictionary)
sutun_isimleri=datafrıme.columns
print(sutun_isimleri)
print(datafrıme.info())
#describe() metodu
import pandas as pd
dictionary={"name":["ali","kenan","veli","ayse","memet","bulut"],"age":[10,25,45,35,74,36],
            "maas":[200,300,150,550,1250,450]}
datafrıme=pd.DataFrame(dictionary)
print(datafrıme.describe())#ortalam min max v degerleri verir örneğin listede min maas 150 gibi
import pandas as pd
dictionary={"name":["ali","kenan","veli","ayse","memet","bulut"],"age":[10,25,45,35,74,36],
            "maas":[200,300,150,550,1250,450]}
datafrıme=pd.DataFrame(dictionary)
print(datafrıme["name"]) #farklı bir metotd asagıda
print(datafrıme.age)

import pandas as pd
dictionary={"name":["ali","kenan","veli","ayse","memet","bulut"],"age":[10,25,45,35,74,36],
            "maas":[200,300,150,550,1250,450]}
datafrıme=pd.DataFrame(dictionary)
datafrıme["yeni_alan"]=[-1,-2,-3,-4,-5,-6]
print(datafrıme)
print(datafrıme.yeni_alan)

#loc metodu
import pandas as pd
#0 dan 2 inci satıra kadar 2 dahıl ve age den baslayıp yenı_alana kadar olan verileri yazdır
dictionary={"name":["ali","kenan","veli","ayse","memet","bulut"],"age":[10,25,45,35,74,36],
            "maas":[200,300,150,550,1250,450]}
datafrıme=pd.DataFrame(dictionary)
datafrıme["yeni_alan"]=[-1,-2,-3,-4,-5,-6]
print(datafrıme.loc[:2,"age":"yeni_alan"])

#sadece age ve maas bılgılerıne getırme istenilen satır kadar
print(datafrıme.loc[:3,["age","maas"]])
#tersten yazdıema
print(datafrıme.loc[::-1,:])

#maas akadar olan alanları yazdır
print(datafrıme.loc[:,:"maas"])
#örnegın maası 300 un ustundekıleri bulma FALSE veya TRUE yazdıerı

import pandas as pd
#0 dan 2 inci satıra kadar 2 dahıl ve age den baslayıp yenı_alana kadar olan verileri yazdır
dictionary={"name":["ali","kenan","veli","ayse","memet","bulut"],"age":[15,16,17,35,74,36],
            "maas":[100,150,240,350,110,220]}
datafrıme=pd.DataFrame(dictionary)
datafrıme["yeni_alan"]=[-1,-2,-3,-4,-5,-6]
filtre1=datafrıme.maas>200
print(filtre1)

#fitrelenen data ile ilgili tüm bilgiler
filrelenen_data=datafrıme[filtre1]
print(filrelenen_data)
filtre2=datafrıme.age<20
print(filtre2)
#iki filtreyi birleştime
fitrelene_data2=datafrıme[filtre1 & filtre2]
print(fitrelene_data2)
#yaşı 60 tan buyuk olanları bulam
print("60 ustu:  ",datafrıme[datafrıme.age>60])


#ortalama maaasssss
import pandas as pd
import numpy as np
#0 dan 2 inci satıra kadar 2 dahıl ve age den baslayıp yenı_alana kadar olan verileri yazdır
dictionary={"name":["ali","kenan","veli","ayse","memet","bulut"],"age":[15,16,17,35,74,36],
            "maas":[100,150,240,350,110,220]}
datafrıme=pd.DataFrame(dictionary)
datafrıme["yeni_alan"]=[-1,-2,-3,-4,-5,-6]
ortalama=datafrıme.maas.mean()
print(ortalama)

#numpy ile bulma
ort2=np.mean(datafrıme.maas)
print(ort2)

import pandas as pd
import numpy as np
#0 dan 2 inci satıra kadar 2 dahıl ve age den baslayıp yenı_alana kadar olan verileri yazdır
dictionary={"name":["ali","kenan","veli","ayse","memet","bulut"],"age":[15,16,17,35,74,36],
            "maas":[100,150,240,350,110,220]}
datafrıme=pd.DataFrame(dictionary)
datafrıme["yeni_alan"]=[-1,-2,-3,-4,-5,-6]
ortalama=datafrıme.maas.mean()
datafrıme["maas_seviyesi"]=["düsük" if each<ortalama else "yüksek" for each in datafrıme.maas]
print(datafrıme)
import pandas as pd
import numpy as np
#0 dan 2 inci satıra kadar 2 dahıl ve age den baslayıp yenı_alana kadar olan verileri yazdır
dictionary={"NAME":["ali","kenan","veli","ayse","memet","bulut"],"age":[15,16,17,35,74,36],
            "MAAS":[100,150,240,350,110,220]}
datafrıme=pd.DataFrame(dictionary)
datafrıme["yeni_alan"]=[-1,-2,-3,-4,-5,-6]
ortalama=datafrıme.MAAS.mean()
datafrıme["maas_seviyesi"]=["düsük" if each<ortalama else "yüksek" for each in datafrıme.MAAS]

datafrıme.columns=[each.lower() for each in datafrıme.columns]
print(datafrıme)





