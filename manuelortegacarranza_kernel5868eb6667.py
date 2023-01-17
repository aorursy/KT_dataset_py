import csv
total = 0
lineaCompleta = []
archivo = open("../input/librosPOO.csv")
reader = csv.reader(archivo,delimiter=',')
for fila in reader:
    print(fila)
    
archivo.close()


