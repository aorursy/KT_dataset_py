edad=17
if edad>=18:
    print("Eres mayor de edad")
else:
    print("Eres menor de edad")
minutos=121
horas=minutos/60
if minutos%60>0:
    horas=minutos//60+1
else:
    horas=minutos/60
pago=horas*15
print("Estuviste",minutos, "minutos, paga la cantidad de",horas)


numero=18
if numero%2==0:
    print("el numero", numero, "es par")
else:
    print("el numero", numero, "es impar")


Docenas=6
Costo=20
Monto=Docenas*Costo
print("El monto de compra es de", Monto, "pesos")

if Docenas<=3:
    print("Obtienes el 10% de descuento de tu compra de", Docenas, "docenas")
else:
    print("Obtienes el 15% de descuento de tu compra de", Docenas, "docenas")

if Docenas<=3:
    Descuento=Monto*.90
    print("El monto a pagar es de",Descuento,"pesos")
if Docenas>4:
    Descuento=Monto*.85
    print("El monto a pagar es de",Descuento,"pesos")

if Docenas>=4:
    print("Obtienes ",Docenas, "unidades gratis")
numero= input ("digite un valor : ");

invertido = numero[ ::-1]

if numero==invertido:
    print("si es igual al reves")

else:
    print("no es igual al reves")