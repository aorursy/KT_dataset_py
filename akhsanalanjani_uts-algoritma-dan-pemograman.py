#Nama      :NUR AKHSAN AL ANJANI
#NIM       :20.01.53.3024
#Prodi     :TEKNIK INFORMATIKA
#Makul     :UTS ALGORITMA DAN PEMOGRAMAN

nilai = [60,70,90,65,75,55,70,55,85,990,660]
nilai = [0] * (11)

jml = 0
ke = 0
for x in range(0, 10 + 1, 1):
    ke = ke + 1
    print("masukan nilai ke" + str(ke))
    nilai[x] = int(input())
    jml = nilai[x] + ke
for x in range(0, 10 + 1, 1):
    print("nilai yang di masukan:" , str(nilai[x]))
print("=======================================")
print("banyaknya data yang di masukan " , len(nilai))
print("jumlah dari angka yang di masukan " , str(jml))
r = float(jml) / 11
print("rata rata nya adalah " + str(r))
jmldev = 0
for x in range(0, 10 + 1, 1):
    hitung = (nilai[x] - r) ** 2
    jmldev = jmldev + hitung
bagi = jmldev / (11 - 1)
jmldev = bagi ** 0.5
print("jml dev " + str(jmldev))
