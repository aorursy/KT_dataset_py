#üyelik açan üç kullanıcının girdiği bilgilerden doğum tarihini

#alıp 8 haneli sayıya dönüştürdükten sonra listeye 8 haneli

#sayıyı ekleyen program

Kullanıcı1=["Spam",4,11,1983,"Mühendis"]

Kullanıcı2=["Eggs",25,12,2001,"Öğrenci"]

Kullanıcı3=["Bacon",14,3,1995,"Doktor"]

gün1,gün2,gün3=Kullanıcı1[1],Kullanıcı2[1],Kullanıcı3[1]

ay1,ay2,ay3=Kullanıcı1[2],Kullanıcı2[2],Kullanıcı3[2]

yıl1,yıl2,yıl3=Kullanıcı1[3],Kullanıcı2[3],Kullanıcı3[3]

gün1=str(gün1)

ay1=str(ay1)

yıl1=str(yıl1)

gün2=str(gün2)

ay2=str(ay2)

yıl2=str(yıl2)

gün3=str(gün3)

ay3=str(ay3)

yıl3=str(yıl3)

a=0

a=str(a)

if(len(gün1)==1):

    gün1=a+gün1

if(len(ay1)==1):

    ay1=a+ay1

if(len(gün2)==1):

    gün2=a+gün2

if(len(ay2)==1):

    ay2=a+ay2

if(len(gün3)==1):

    gün3=a+gün3

if(len(ay3)==1):

    ay3=a+ay3

Kullanıcı1.append(yıl1+ay1+gün1)

Kullanıcı2.append(yıl2+ay2+gün2)

Kullanıcı3.append(yıl3+ay3+gün3)

print(Kullanıcı1,Kullanıcı2,Kullanıcı3) 