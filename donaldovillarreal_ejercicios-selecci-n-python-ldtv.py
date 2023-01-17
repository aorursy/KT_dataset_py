edad=17

if edad >=18:

    print("Eres mayor de edad")

else:

    print("Eres menor de edad")
minutos=90

horas=minutos/60

if horas%60>0:

    horas=minutos//60+1

else:

    horas=minutos/60

pago=horas*15

print("Estuviste",minutos, "minutos,paga la cantidad de",pago, "pasos por", horas, "horas")
numero=18

if numero %2==0:

    print("el numero", numero ,"es par")

else:

    print("el numero", numero , "el impar")

cantidad=20

precioUnit=10

descuento=.10

obsequio=0

if cantidad >= 36:

    descuento=.15

    obsequio= cantidad//12-3

    

total=cantidad*precioUnit

descuento_total = total*descuento

total_pagar=total-descuento_total



print("Producto ","cantidad","Subtotal")

print("Producto1 ",cantidad,"     ",total)

print("Dewcuento aplicable",descuento*100,"%")

print("Total descuento",descuento_total)

print("Total a pagar",total_pagar)

print("Unidades de obsequio",obsequio)
numero=646



centenas=(numero-numero%100)/100

print(centenas)

unidades=numero%10

print(unidades)



if centenas==unidades:

    print("el numero",numero,"es igual al derecho y al reves")

else:

    print("el numero",numero,"NO ess igual al derecho y al reves")