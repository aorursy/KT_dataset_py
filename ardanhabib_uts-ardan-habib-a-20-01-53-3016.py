#Nama : Ardan Habib A
#NIM : 20.01.53.3016
#Kelas : Teknik Informatika (DLC)

n = [60,70,90,65,75,55,70,55,85,99,66]

l = 0
k = 0
for a in range(0, 11, 1):
    k = k + 1
    print("nilai ke-", int(k),":",  n[a])
    l = n[a] + l
print("==============================")
print("banyaknya data : " , len(n))
print("jumlah : " , int(l))
r = l / len(n)
print("rata rata : " , float(r))
jmldev = 0
for a in range(0, 11, 1):
    hitung = (n[a] - r) ** 2
    jmldev = jmldev + hitung
bagi = jmldev / 11
stndev = bagi ** 0.5
print("standar deviasinya : " , float(stndev))