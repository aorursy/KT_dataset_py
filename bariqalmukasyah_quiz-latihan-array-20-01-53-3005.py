i = 0
UTS = [0] * (10)
UAS = [0] * (10)
nilaiakhir = [0] * (10)

lulus = 0
tdklulus = 0
for i in range(0, 10, 1):
    print("Masukkan nilai UTS")
    UTS[i] = float(input())
    print("Masukkan nilai UAS")
    UAS[i] = float(input())
    nilaiakhir[i] = UTS[i] * 0.4 + UAS[i] * 0.6
    print(nilaiakhir[i])
    if nilaiakhir[i] >= 70:
        lulus = lulus + 1
        print("Selamat Anda LULUS")
    else:
        tdklulus = tdklulus + 1
        print("Maaf Anda TIDAK LULUS")
print("Banyaknya siswa lulus = ")
print(lulus)
print("Banyaknya siswa tidak lulus = ")
print(tdklulus)
