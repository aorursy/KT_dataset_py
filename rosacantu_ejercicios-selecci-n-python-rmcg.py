edad=17
if edad>18:
    print("Eres mayor de edad")
else:
    print("Eres menor de edad")
minutos=90

if minutos%60>0:
    horas=minutos//60+1
else:
    horas=minutos/60
pago=horas*15
print("Estuviste",minutos,"minutos, paga la cantidad de", pago, "pesos por",horas,"horas")
numero=188
if numero%2==0:
    print("El numero", numero, "es par")
else:
    print("el numero", numero, "es impar")
print(numero/2)
print(numero//2)
print(numero%2)
print(numero/2-numero//2)
print(6%2)
cantidad= 200
precioUnit=10
descuento=0.10
obsequio=0

if cantidad >= 36:
    descuento = 0.15
    obsequio = cantidad//12-3

total=cantidad*precioUnit
descuento_total = total*descuento
total_pagar=total-descuento_total
print("Producto","Cantidad","Subtotal")
print("Producto1", cantidad," ", total)
print("Descuento aplicable",descuento*100,"%")
print("Total descuento",descuento_total)
print("Total a pagar",total_pagar)
print("unidades de obsequio",obsequio)
    
    

docena=2
costodocena=250
montocompra1= (costodocena*0.85)*(docena)
montocompra2= (costodocena*0.90)*(docena)
obsequio=(docena//3)

if docena>=3:
    print("Se tiene un 15% de descuento, tu monto de compra es",montocompra1," y ademas te llevas como obsequio", obsequio,"unidades de regalo por la compra de", docena,"docenas")
if docena<3:
    print("Se tiene un 10 % de descuento, tu monto de compra es", montocompra2)
numero=686

centenas=(numero-numero%100)/100
unidades=numero%10


if centenas==unidades:
    print("El numero ", numero," es igual al derecho y al revés")
else:
    print("El numero ", numero,"no es igual al derecho y al reves")


numero = 808

if numero > 99 and numero < 1000:
    print("El numero tiene tres cifras")
else:
    print("El numero NO tiene tres cifras")

if numero//100==numero%100:
    print("Y el numero es igual, al derecho y al reves")
else:
    print("Y el numero no es igual, al derecho y al reves")