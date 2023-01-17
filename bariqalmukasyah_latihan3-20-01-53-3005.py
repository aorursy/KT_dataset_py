# Menghitung Deret Fibonacci dengan perulangan While
totalprev = 0
bil1 = 0
bil2 = 1
billoop = 1
batas = int(input('Masukkan Batas = '))
print(totalprev)
while billoop < batas:
    totalprev = bil1 + bil2
    bil2 = bil1
    bil1 = totalprev
    print(totalprev)
    billoop = billoop + 1
# Menghitung Faktorial dari sebuah bilangan dengan perulangan While
bilinput = int(input('Faktorial dari bilangan = '))
hasil = 1
bilulang = 1
while bilulang < bilinput+1:
    hasil = bilulang*hasil
    bilulang = bilulang+1
print(hasil)
