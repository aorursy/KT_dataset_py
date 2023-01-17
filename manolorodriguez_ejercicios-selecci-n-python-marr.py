edad=23

if edad>=18:

    print("Eres mayor de edad")

else:

    print("Eres menor de edad")
minutos=121

horas=minutos/60

print(horas)

if minutos%60>0:

    horas=minutos//60+1

else:

    horas=minutos/60

pago=horas*15

print("Estuviste", minutos, "minutos, paga la cantidad de ", pago, "pesos por", horas,"horas")
numero= 187

if numero%2==0:

    print("el numero", numero, "es par")

else:

    print("el numero", numero, "es impar")
cantidad=200

precioUnit=10

descuento=.10

obsequio=0

if cantidad >= 36:

    descuento = .15

    obsequio = cantidad//12-3

    

total=cantidad*precioUnit

descuento_total = total*descuento

total_pagar=total-descuento_total

print("producto1", "cantidad","Subtotal")

print("producto1", cantidad,"   ", total)

print("descuento aplicable", descuento*100,"%")

print("total descuento", descuento_total)

print("total a pagar", total_pagar)

print("unidades de obsequio",obsequio)
numero=575



centenas=(numero-numero%100)/100

unidades=numero%10



if centenas==unidades:

    print("El numero",numero,"es igual al derecho y al reves")

else:

    print("El numero",numero,"no es igual al derecho y al reves")