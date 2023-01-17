n = [0] * (11)

jumlahdata = int(input())
jumlahnilai = 0
for i in range(0, jumlahdata - 1 + 1, 1):
    n[i] = int(input())
    jumlahnilai = jumlahnilai + n[i]
rata = float(jumlahnilai) / jumlahdata

# rata dari sebuah data
print(jumlahnilai)
print(rata)
sigma = 0
for i in range(0, jumlahdata - 1 + 1, 1):
    perhitungan = (n[i] - rata) ** 2
    sigma = sigma + perhitungan
    
    # standar deviasi
pembagian = float(sigma) / jumlahdata
stdev = pembagian ** 0.5
print(stdev)
