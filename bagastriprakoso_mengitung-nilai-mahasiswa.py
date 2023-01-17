# NILAI ARRAY

uts = [60, 70, 50, 70, 80, 70, 90, 80, 40, 75]

uas = [70, 80, 60, 90, 70, 75, 90, 70, 60, 85]

nilaimin = 70

lulus = 0







for i in range(len(uas)):

    ratarata = 0.4*uts[i]+0.6*uas[i]

    if ratarata >= nilaimin:

        lulus += 1



        print("Mahasiswa ke " , i , " UTS = " , uts[i] , " dan UAS = " , uas[i] , " Nilai Akhir =" , ratarata , " Lulus ")

    else :

        print("Mahasiswa ke " , i , " UTS = " , uts[i] , " dan UAS = " , uas[i] , " Nilai Akhir =" , ratarata , " Tidak Lulus :( ")



print("Jumlah Mahasiswa Yang Lulus: " , lulus)

print("Jumlah Mahasiswa Yang Tidak Lulus: " , len(uas) - lulus)