i = 0
nama = [""] * (10)
uts = [0] * (10)
uas = [0] * (10)
nilaiakhir = [0] * (10)

lulus = 0
tidaklulus = 0
for i in range(0, 9 + 1, 1):
    print("masukan nama=")
    nama[i] = input()
    print("masukan nilai uts=")
    uts[i] = float(input())
    print("masukan nilai uas=")
    uas[i] = float(input())
    nilaiakhir[i] = 0.4 * uts[i] + 0.6 * uas[i]
    print(nilaiakhir[i])
    if nilaiakhir[i] >= 70:
        lulus = lulus + 1
        print("mahasiswa lulus")
    else:
        tidaklulus = tidaklulus + 1
        print("mahasiswa tidak lulus")
print("banyaknya mahasiswa lulu=")
print(lulus)
print("banyaknya mahasiswa tidak lulus=")
print(tidaklulus)



