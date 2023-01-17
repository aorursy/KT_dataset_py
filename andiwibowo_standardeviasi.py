nilai = [0]*(11)



i = 0

jumlah = 0

n = 11

for i in range(0, 10 + 1, 1):

    print("Masukkan Nilai ke-" + str(i+0))

    nilai[i] = float(input())

    jumlah = jumlah+nilai[i]

print("Jumlah Nilai = " + str(jumlah))

ratarata = float(jumlah/n)

print("Rata-rata = " + str(ratarata))

for i in range(0, 10 + 1, 1):

    deviasi = float((1)/(n-1)) * (nilai[i]-ratarata)**2

    print("deviasi ke-" + str(i+0) + "=" + str(deviasi))
sigma = 0

i = 0

jumlah = 0

data = ([60,70,90,65,75,55,70,55,85,99,66])

for i in range(len(data)):

    jumlah += data[i]

    ratarata = jumlah/len(data)

for i in range(len(data)):

    dev=(data[i]-ratarata)**2

n = sigma/len(data)

deviasi = n**0.5

print("Jumlah = ",jumlah)

print("Rata-rata = ",ratarata)

print(dev)