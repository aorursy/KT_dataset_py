#UTS Algoritma dan Pemrograman
#Nama : Fajar Kurniawan

nilai=(60,70,90,65,75,55,70,55,85,99,66)
jumlah=0
for i in range(len(nilai)):
    jumlah +=nilai[i]
rata=jumlah/len(nilai)
sigma = 0
for i in range(len(nilai)):
    hitung =(nilai[i]-rata)**2
    sigma += hitung
varian=sigma/len(nilai)
deviasi=varian ** 0.5
print("Nilai murid = ",nilai)
print("Jumlah nilai = ", jumlah)
print("Rata-rata nilai = ", rata)
print("Varian = ", varian)
print("Standar Deviasi = ", deviasi) 