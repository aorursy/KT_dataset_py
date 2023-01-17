nilai = [60,70,90,65,75,55,70,55,85,99,66]

rerata = 0
for x in nilai:
    rerata+=x
reratanilai=rerata/len(nilai)
print("Rerata = ", reratanilai)

var = 0
for i in range(0,len(nilai)):
    var+=((nilai[i]-reratanilai)**2) / len(nilai)

stdDvs = var**(1/2)

print("Standart Deviasi = ", stdDvs)