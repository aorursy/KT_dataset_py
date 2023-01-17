#Nilai=[60,70,90,65,75,55,70,55,85,99,66]
nilai = [0] * (11)

i = 0
y = int(input())
for i in range(0, y + 1, 1):
    nilai[i] = float(input())
    print(nilai[i])
jml = nilai[0] + nilai[1] + nilai[2] + nilai[3] + nilai[4] + nilai[5] + nilai[6] + nilai[7] + nilai[8] + nilai[9] + nilai[10]
print("Jumlah Nilai = ")
print(jml)
rerata = float(jml) / 11
print("Rata-rata")
print(rerata)
for i in range(0, y + 1, 1):
    deviasi = float(1) / (y - 1) * (nilai[i] - rerata) ** 2
    print(deviasi)



