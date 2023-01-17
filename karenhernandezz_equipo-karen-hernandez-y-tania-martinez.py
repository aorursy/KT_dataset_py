num=8
if num > 1:
    cont=0
    for i in range(2,num):
        resto=num%i
        if resto==0:
            cont+=1
    if cont==0:
        print("El numero",num,"es primo")
    else:
        print("El numero",num,"no es primo")
numeros = [5,3,7,2,6,3]
print("El numero mayor es:", max(numeros))
print("El numero menor es:", min(numeros))
sorted(numeros)
numeros = [5,4,7,2,8,4,6]
def media(numeros):
    s = 0
    for elemento in numeros:
        s += elemento
    return s / float(len(numeros))

def varianza(numeros):
    s = 0
    m = media(numeros)
    for elemento in numeros:
        s += (elemento - m) ** 2
    return s / float(len(numeros))

def desviacion_tipica(numeros):
    x = varianza(numeros)
    return (x**(.5))

print("Media:", media(numeros))
print("Varianza:", varianza(numeros))
print("Desviacion:", desviacion_tipica(numeros))

nombres = ["Juan", "Ana", "Alejandra", "Raul"]
nombres.sort(key = str.lower)
print(nombres)


Billetes500 ="10 billetes de 500"
Billetes200 = "1 billete de 200" 
Billetes100 = "2 billetes de 100"
Billetes50 = "2 billetes de 50"
Billetes20 = "3 billetes de 20"

print("Se deben pagar", Billetes500 ,"," , Billetes200 ,",", Billetes100 ,",", Billetes50,"y", Billetes20 ,"para tener un total de $ 5,560" )
Billetes200 = "25 billetes de 200"
Billetes50 = "2 billetes de 50"
Billetes20 = "3 billetes de 20"

print("Se deben pagar", Billetes200 ,",", Billetes50 ,"y", Billetes20 , "para tener un total de $ 5,560")

import random as rn
cantidad = 300

for i in range(1, cantidad + 1):
    aleatorio=rn.randint(500,1000)
    print(aleatorio) 
numeros = [555,736,594,640,644,806,695,511,665,716,584,668,924,730,848,958,801,638,946,855,788,845,844,806,714,688,561,963,683,581,876,519,731,863,615,524,611,794,894,909,670,737,541,585,516,711,796,973,793,716,937,728,973,762,906,826,982,827,575,820,627,555,757,757,852,509,782,808,842,757,916,511,788,669,768,962,623,726,996,525,663,937,544,623,911,973,792,656,735,703,682,700,979,564,769,518,833,510,852,983,607,917,657,542,880,993,902,595,858,947,802,837,982,661,956,871,737,583,736,779,509,616,715,558,907,703,789,630,905,560,586,695,918,755,669,572,671,854,974,505,574,513,919,882,513,768,857,842,675,989,737,835,924,964,609,526,713,554,589,610,952,877,571,964,614,966,668,838,662,615,911,932,658,790,797,517,636,826,849,926,639,547,737,591,566,815,856,727,686,997,747,716,799,579,571,567,734,686,986,730,765,967,964,725,586,976,649,954,988,849,513,551,530,752,760,843,609,598,672,575,912,605,671,505,689,548,888,726,769,672,936,994,739,698,572,820,522,924,892,988,742,683,680,940,775,852,871,724,647,502,990,593,761,655,813,574,565,853,924,951,900,627,928,731,604,655,821,678,798,820,713,635,965,990,644,562,929,982,982,976,760,933,591,536,709,675,574,966,679,565,812,833,576,818,845,902,584,613,683,689]
def media(numeros):
    s = 0
    for elemento in numeros:
        s += elemento
    return s / float(len(numeros))

def varianza(numeros):
    s = 0
    m = media(numeros)
    for elemento in numeros:
        s += (elemento - m) ** 2
    return s / float(len(numeros))

def desviacion_tipica(numeros):
    x = varianza(numeros)
    return (x**(.5))

print("El numero mayor es:", max(numeros))
print("El numero menor es:", min(numeros))
print("Media:", media(numeros))
print("Varianza:", varianza(numeros))
print("Desviacion:", desviacion_tipica(numeros))
