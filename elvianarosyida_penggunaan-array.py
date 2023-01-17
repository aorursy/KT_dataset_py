# Latihan penggunaan Array
i = 0

uts=[60,70,50,70,80,70,90,80,40,75]
uas=[70,80,60,90,70,75,90,70,60,85]
totaluts=0
totaluas=0
lulus=0
tidaklulus=0
print("Nilai minimum kelulusan 70")
# Untuk mencari nilai Akhir= 0.4 * uts[i] + 0.6 * uas[i].

for i in range (10):
    if 0.4 * uts[i] + 0.6 * uas[i] >= 70 :
      print("Mahasiswa ke",i,", Nilai uts =",uts[i] ,"dan Nilai Uas =",uas[i]," Maka Nilai Akhir Mahasiswa =", 0.4 * uts[i] + 0.6 * uas[i],"dinyatakan Lulus")
      lulus+=1
    else:
         print("Mahasiswa ke",i,", Nilai uts =",uts[i] ,"dan Nilai Uas =",uas[i]," Maka Nilai Akhir Mahasiswa =", 0.4 * uts[i] + 0.6 * uas[i],"dinyatakan Tidak Lulus")
         tidaklulus+=1

# Menampilkan jumlah mahasiswa yang lulus.           
print("Jumlah Mahasiswa yang lulus adalah",lulus) 

# Menampilkan mahasiswa yang tidak lulus.
print("Jumlah Mahasiswa yang tidak lulus adalah",tidaklulus)
    
    
