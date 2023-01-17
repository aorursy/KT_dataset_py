edad=22

if edad>18:

    print("Eres mayor de edad")

else:

    print("Eres menor de edad")
minutos=90

horas=minutos/60

print(horas)

if minutos%60 > 0:

    horas=minutos//60+1

else:

    horas=minutos/60

pago=horas*15

print("Estuviste," , minutos, "minutos, paga la cantidad de" ,pago, "pesos por" , horas, "horas")
numero=18

if numero%2==0:

    print("El numero" , numero , "es par")

    print("El numero" , numero , " es impar")



precio=100

producto=240

docenas=precio//12

monto=precio+docenas

if docenas >=3:

    descuento=monto*.15

    montoF=monto-descuento

    regalos=docenas//3

    print("El monto de la compra es", monto , "pesos, con un descuento de" , descuento , "pesos,lo cual da un total de" , montoF , "pesos a pagar, por la compra de" , docenas , "docenas se regalan" , regalos , "productos")

    

else:

    descuento=monto*.10

    montoF=montoF-descuento

    print("El monto de la compra es" , monto ,  "pesos, con un descuento de" , descuento , "pesos, lo cual da un total de" , montoF , "pesos a pagar.") 
numero=745

unidades = numero%10

centenas = int(numero/100)



if(numero)<1000 and numero>99:

    if(unidades==centenas):

        print("El numero" , numero , "es igual al inverso")

    else:

        print("El numero" , numero , "no es igual al inverso")

        

else:

    print("ingresa numero de tres cifras")
