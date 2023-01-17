i = 0
jmlhlulus = 0
jmlhtdklulus = 0
nama = [""] * (10)
uts = [0] * (10)
uas = [0] * (10)
nilaiakhir = [0] * (10)

for i in range(0, 9 + 1, 1):
    print("NAMA")
    nama[i] = input()
    print("Nilai UTS")
    uts[i] = float(input())
    print("Nilai UAS")
    uas[i] = float(input())
    nilaiakhir[i] = uts[i] * 0.4 + uas[i] * 0.6
    print("Nilai Akhir " + str(nilaiakhir[i]))
    if nilaiakhir[i] > 70:
        print("LULUS")
        jmlhlulus = jmlhlulus + 1
        jmlhsiswa = jmlhlulus + jmlhtdklulus
    else:
        print("TIDAK LULUS")
        jmlhtdklulus = jmlhtdklulus + 1
        jmlhsiswa = jmlhlulus + jmlhtdklulus
print("Jumlah Mahasiswa Adalah " + str(jmlhsiswa))
print("Banyaknya Siswa LULUS Adalah " + str(jmlhlulus))
print("Banyaknya Siswa TIDAK LULUS Adalah " + str(jmlhtdklulus))