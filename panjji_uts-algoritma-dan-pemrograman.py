#("Nama : Panji suci sugiyanto")
#("Nim : 20.01.53.3019"


Nilai=[60,70,90,65,75,55,70,55,85,990,660]
n = [0] * (11)

jumlah = 0
urutan = 0
for data in range(0, 10 + 1, 1):
    urutan = urutan + 1
    print("Masukan Nilai " + str(urutan))
    n[data] = int(input())
    jumlah = n[data] + jumlah
print("======================================")
for data in range(0, 10 + 1, 1):
    print("Nilai yang dimasukan : " + str(n[data]))
print("======================================")
print("Banyak nya data yang telah masuk" , len(Nilai))
print("Total angka yang masuk " + str(jumlah))
rata = jumlah / 11
print("rata-rata nya adalah " + str(rata))
totalDev = 0
for data in range(0, 10 + 1, 1):
    rumus = (n[data] - rata) ** 2
    totalDev = totalDev + rumus
bagi = totalDev / (11 - 1)
deviasi = bagi ** 0.5
print("standar daviasi adalah" + str(deviasi))
