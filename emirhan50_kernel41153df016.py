import numpy as np
v_milliyazilimlar =  (['Kuzgun Mobil Canlı Yayın Yazılımı','Aselsan Entegre Komuta Kontrol Sistemi','GAG-Geniş Alan Gözetleme Sistemi'

                       ,'Aselsan AcroSAT Gemi Uydu Yazılımı', 'Kement Projesi','Adop Ateş DestekOtomasyon Sistemi'

                       ,'Denizgözü Martı Elektroptik Gözetleme Sistemi','Tubitak Gerçek Zamanlı İletişim Sistemi','Denizgözü Martı Elektroptik Gözetleme Sistemi'])





print(v_milliyazilimlar)

print()

print('type:',type(v_milliyazilimlar))
# shape

v_shape = v_milliyazilimlar.shape

print("v_shape : " , v_shape , " and type is : " , type(v_shape))
# Reshape



v_array = v_milliyazilimlar.reshape(8,1)

print(v_array)
v_shape2 = v_milliyazilimlar.shape

print("v_shape2 : " , v_shape2 , "and type is : " , type(v_shape2))