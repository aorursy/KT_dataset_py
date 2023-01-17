import math
yA = float(input("A noktasının bilinen Y koordinatını giriniz: "))
xA = float(input("A noktasının bilinen X koordinatını giriniz: "))
mAB = float(input("A noktasından B noktasına ölçülen semt açısını giriniz: "))
sAB = float(input("A noktasından B noktasına ölçülen yatay mesafeyi giriniz: "))
decimal = int(input("Çıktı hesapların arzu ettiğiniz ondalık hassasiyetini  giriniz: "))

def temel_odev1(yA, xA, mAB, sAB, decimal): # decimal argümanı, sonuçların virgülden sonra kaç hane olmasını istediğimizi belirler.
  #1. temel ödev formüllerini uyguluyoruz. Kırılma açılarını radyan cinsine çevirmeyi unutmayalım. 
  yB=yA+(sAB*math.sin((mAB)/(200/math.pi)))
  xB=xA+(sAB*math.cos((mAB)/(200/math.pi)))
  yB = round(yB,decimal)
  xB = round(xB,decimal)
  # Hesaplanan B noktasının koordinatlarını ekrana verelim.
  print("                              ")
  print("1. TEMEL ÖDEV HESABI SONUÇLARI")
  print("------------------------------")
  print("B noktasının hesaplanan Y koordinatı: " + str(yB) + " metre")
  print("B noktasının hesaplanan X koordinatı: " + str(xB)+ " metre")

temel_odev1(yA, xA, mAB, sAB, decimal)  
yA = float(input("A noktasının bilinen Y koordinatını giriniz: "))
xA = float(input("A noktasının bilinen X koordinatını giriniz: "))
yB = float(input("B noktasının bilinen Y koordinatını giriniz: "))
xB = float(input("B noktasının bilinen Y koordinatını giriniz: "))
decimal = int(input("Çıktı hesapların arzu ettiğiniz ondalık hassasiyetini  giriniz: "))

def temel_odev2(yA, xA, yB, xB, decimal): # decimal argümanı, sonuçların virgülden sonra kaç hane olmasını istediğimizi belirler.
  deltaY = yB-yA
  deltaX = xB-xA
  tanAB = (yB-yA)/(xB-xA)
  semt = math.atan(tanAB)*(200/math.pi)
  semt = round(semt,decimal)
  print("                                   ")
  print("2. TEMEL ÖDEV HESABI SONUÇLARI")
  print("-----------------------------------")
  if deltaX < 0 or deltaY > 0 and deltaX < 0:
      print("A ve B noktaları arasındaki semt açısı: "+str(semt+200) + " grad")
  elif deltaY < 0 and deltaX > 0:
      print("A ve B noktaları arasındaki semt açısı: "+str(semt + 400)+ " grad")
  else:
      print("A ve B noktaları arasındaki semt açısı: "+str(semt)+ " grad")

  mesafe = math.sqrt((deltaY**2)+(deltaX**2))
  mesafe = round(mesafe,decimal)

  print("A ve B noktaları arasındaki yatay mesafe: " +str(mesafe)+ " metre")

temel_odev2(yA, xA, yB, xB, decimal)
# 3. temel ödev fonksiyonu için kullanacağımız B ve C doğruları arasındaki kırılma açısını girelim.
beta = float(input("B ve C doğruları arasındaki kırılma açısını giriniz: "))
ABsemt = float(input("A ve B noktalarından geçen semt açısını giriniz: "))

def temel_odev3(beta, ABsemt):
 
  print("                              ")
  print("3. TEMEL ÖDEV HESABI SONUÇLARI")
  print("------------------------------")
  n = ABsemt + beta # 3. temel ödevin ana formülünü koşullara göre uygulayalım.
  if n < 200:
      n + 200
      print("B ve C noktaları arasındaki semt açısı: "+str(n+200))
  elif 200 < n < 400 or 400 < n < 600:
      print("B ve C noktaları arasındaki semt açısı: "+str(n-200))
  else:
      print("B ve C noktaları arasındaki semt açısı: "+str(n-600))

temel_odev3(beta,ABsemt) 
yA = float(input("A noktasının bilinen Y koordinatını giriniz: "))
xA = float(input("A noktasının bilinen X koordinatını giriniz: "))
yB = float(input("B noktasının bilinen Y koordinatını giriniz: "))
xB = float(input("B noktasının bilinen X koordinatını giriniz: "))
yC = float(input("C noktasının bilinen Y koordinatını giriniz: "))
xC = float(input("C noktasının bilinen X koordinatını giriniz: "))
decimal = int(input("Çıktı hesapların arzu ettiğiniz ondalık hassasiyetini  giriniz: "))


def temel_odev4(yA, xA, yB, xB, yC, xC, decimal): # decimal argümanı, sonuçların virgülden sonra kaç hane olmasını istediğimizi belirler.
  deltaY_AB = yB-yA
  deltaX_AB = xB-xA
  tanAB = (yB-yA)/(xB-xA)
  semt_AB = math.atan(tanAB)*(200/math.pi)
  semt_AB = round(semt_AB,decimal)
  print("                                   ")
  print("2. TEMEL ÖDEVDEN AB SEMT AÇISI HESABI")
  print("-----------------------------------")
  if deltaX_AB < 0 or deltaY_AB > 0 and deltaX_AB < 0:
      print("A ve B noktaları arasındaki semt açısı: "+str(semt_AB+200) + " grad")
  elif deltaY_AB < 0 and deltaX_AB > 0:
      print("A ve B noktaları arasındaki semt açısı: "+str(semt_AB + 400)+ " grad")
  else:
      print("A ve B noktaları arasındaki semt açısı: "+str(semt_AB)+ " grad")

  deltaY_AC = yC-yA
  deltaX_AC = xC-xA
  tanAC = (yC-yA)/(xC-xA)
  semt_AC = math.atan(tanAC)*(200/math.pi)
  semt_AC = round(semt_AC,decimal)
  print("                                   ")
  print("2. TEMEL ÖDEVDEN AC SEMT AÇISI HESABI")
  print("-----------------------------------")
  if deltaX_AC < 0 or deltaY_AC > 0 and deltaX_AC < 0:
      print("A ve C noktaları arasındaki semt açısı: "+str(semt_AC+200) + " grad")
  elif deltaY_AC < 0 and deltaX_AC > 0:
      print("A ve C noktaları arasındaki semt açısı: "+str(semt_AC + 400)+ " grad")
  else:
      print("A ve C noktaları arasındaki semt açısı: "+str(semt_AC)+ " grad")

  print("                                   ")
  print("A-B-C DOĞRULARI ARASINDAKİ KIRILMA AÇISI HESABI")
  print("-----------------------------------")
  kirilma = (semt_AC - semt_AB)
  kirilma = round(kirilma,decimal)
  if (kirilma) >= 0: 
    print("Kırılma açısı: " + str(kirilma) + " grad")
  else:  # Fark işlemiyle negatif çıkan sonuçlara 200 grad eklemeliyiz.
    print("Kırılma açısı: " + str(kirilma + 200) + " grad")

temel_odev4(yA, xA, yB, xB, yC, xC, decimal)