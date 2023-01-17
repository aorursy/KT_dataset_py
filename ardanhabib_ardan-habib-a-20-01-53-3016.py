#Tugas Algoritma dan Pemrograman
a = str("Ardan Habib A")
b = str("20.01.53.3016")
c = str("Teknik Informatika(DLC)")

print("nama = ", a)
print("nim = ", b)
print("prodi = ", c)
print()
nama = ["a","b","c","d","e","f","g","h","i","j"]
uts = [60,70,50,70,80,70,90,80,40,75]
uas = [70,80,60,90,70,75,90,70,60,85]
hasil = [0] * 10
x = 0
y = 0
print("--------------------------------------")
print("Batas Nilai Kelulusan adalah 70")
print("--------------------------------------")
print()
for a in range(len(uts)):
    hasil[a] = uts[a]*40/100 + uas[a]*60/100
    
    if hasil[a] > 70:
        x = x+1
        print("Nama Siswa : " , nama[a]," nilainya ", float(hasil[a]), " Lulus")
    else:
        y = y+1
#         print("Nama Siswa : " , nama[a]," nilainya ", float(hasil[a]),  "Tidak lulus")
        
print()
print("Jumlah Total Siswa Yang Lulus : ", x)
print("Jumlah Total Siswa Yang Tidak Lulus : ",y)
# Deret Fibonacci

bil1 = 0
bil2 = 1
j = 0
b = 0
n = int(input("masukan N fibonacci : "))
while b < n:
    j = bil1+bil2
    bil1=bil2
    bil2=j
    print(bil1)
    b=b+1
    
# Tugas Factorial

print("N factorial, Masukan nilai N : ")
n = int(input())
a = 1
for c in range(1,n+1,1):
    a = a * c
print("Factorial dari " + str(n) + " adalah : " + str(a))
# Memanggil Array

i = 0
larik = [0]*(4)
for i in range(0, 3+1 , 1):
    larik[i] = int(input("masukan angka : "))
for i in range(0, 3+1 , 1):
    print("Angka yang dimasukkan = ", larik[i], flush=True)
# Menghitung nilai dengan array

i = 0
nama = [""] * (3)
uts = [0] * (3)
uas = [0] * (3)
hasil = [0] * (3)
for i in range(0, 2+1, 1):
    nama[i] = input("masukkan nama = ")
    uts[i] = float(input("masukkan uts = "))
    uas[i] = float(input("masukkan uas = "))
    hasil[i] = 0.4 * uts[i] + 0.6 * uas[i]
    print("Hasil akhir = ", hasil[i] ,flush=True)