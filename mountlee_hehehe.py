file = open("../input/train.csv")
fout = open('subset_datatest.csv','w')
n = 0
for line in file:
    if n == 0:
        fout.write(line)
    if n <200000*5:
        n +=1
    elif 200000*5<=n <200000*10:
        n +=1
        fout.write(line)
    else:
        break
fout.close()
file.close()