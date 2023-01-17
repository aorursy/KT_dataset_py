nilai = [0] * (11)

i = 0
y = int(input())
for i in range(0, y + 1, 1):
    nilai[i] = int(input())

jumlah = nilai[0] + nilai[1] + nilai[2] + nilai[3] + nilai[4] + nilai[5] + nilai[6] + nilai[7] + nilai[8] + nilai[9] + nilai[10]
print("Jumlah Nilai =")
print(jumlah)
ratarata = float(jumlah) / 11
print("Rata Rata =")
print(ratarata)
for i in range(0, y + 1, 1):
    deviasi = float(1) / (y - 1) * (nilai[i] - ratarata) ** 2
    print("Deviasi =")
    print(deviasi)
