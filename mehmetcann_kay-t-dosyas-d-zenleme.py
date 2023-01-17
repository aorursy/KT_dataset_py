#kayıt dosyasında ilk 3 satırda yazı formatında olan verileri

#okuyup bu süreyi saniye cinsinden sayısal olarak

#yazdıran program (gün:saat:dakika:saniye)

satır1="00:01:34:23"

satır2="43:14:06:46"

satır3="00:16:06:48"

saniye_1=satır1[9:11]

saniye_1=int(saniye_1)

dakika_1=satır1[6:8]

dakika_1=int(dakika_1)

dakika_1=dakika_1*60    

saat_1=satır1[3:5]

saat_1=int(saat_1)

saat_1=saat_1*(60**2)

gün_1=satır1[0:2]

gün_1=int(gün_1)

gün_1=gün_1*(60**2)*24

ToplamSaniye1=gün_1+saat_1+dakika_1+saniye_1



saniye_2=satır2[9:11]

saniye_2=int(saniye_2)

dakika_2=satır2[6:8]

dakika_2=int(dakika_2)

dakika_2=dakika_2*60

saat_2=satır2[3:5]

saat_2=int(saat_2)

saat_2=saat_2*(60**2)

gün_2=satır2[0:2]

gün_2=int(gün_2)

gün_2=gün_2*(60**2)*24

ToplamSaniye2=gün_2+saat_2+dakika_2+saniye_2



saniye_3=satır3[9:11]

saniye_3=int(saniye_3)

dakika_3=satır3[6:8]

dakika_3=int(dakika_3)

dakika_3=dakika_3*60

saat_3=satır3[3:5]

saat_3=int(saat_3)

saat_3=saat_3*(60**2)

gün_3=satır3[0:2]

gün_3=int(gün_3)

gün_3=gün_3*(60**2)*24

ToplamSaniye3=gün_3+saat_3+dakika_3+saniye_3



print(ToplamSaniye1,ToplamSaniye2,ToplamSaniye3)