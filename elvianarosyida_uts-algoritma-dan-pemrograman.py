# Tugas UTS

nilai=[60,70,90,65,75,55,70,55,85,99,66.0]

jumlah=0

for i in nilai:

    jumlah+=i

rerata=jumlah/len(nilai)

print("Rerata Nilai = ",rerata)



X=0

for i in range(len(nilai)):

    hitung=(nilai[i]-rerata)**2

    X+=hitung

varian=X/(len(nilai)-1)

standardeviasi=varian**0.5

print("Standar Deviasi = ",standardeviasi)
