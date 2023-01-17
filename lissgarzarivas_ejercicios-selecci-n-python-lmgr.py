edad=22

if edad>=18:

    print("eres mayor de edad")

else:

    prin("eres menor de edad")
minutos=90

horas=minutos/60

print(horas)

if minutos%60>0:

    horas=minutos//60+1

else:

    horas=minutos/60

pago=horas*15

print("Estuviste", minutos, "para la cantidad de",pago, "pesos por", horas, "horas")
numero=18

print(numero/2)

print(numero//2)

print(numero%2)

print(numero/2-numero//2)



if numero%2==0:

    print("el numro", numero, "es par")

else:

    print("el numero", numero, "es impar")

docenas=6

obsequio=docenas-3

if docenas>3:

    print("el descuento es de 15% al comprar",docenas ,"docenas y optienes un",obsequio ,"obsequio extra")

else:

    print("el descuento es de 10% al comprar")



#opcion hecha en clase



cantidad=40

precioUnit=10

descuento=.10

regalo=0



if cantidad>=36:

    descuento=.15

    regalo=cantidad//12-3



total=cantidad*precioUnit

descuento_total=total*descuento

total_pagar=total-descuento_total



print("Producto1","Cantidad", "Subtotal")

print("Producto 1" ,cantidad,"     ",total)

print("Descuento aplicable",descuento*100,"%")

print("Total descuento",descuento_total)

print("Total a pagar",total_pagar)

print("Unidades de obsequio",regalo)

numero=285

centenas=(numero-numero%100)/100

print(centenas)



unidades=numero%10

print(unidades)



if centenas==unidades:

    print("El numero",numero,"es igual al derecho y al reves")

else:

    print("El numero",numero,"no es igual al derecho y al reves")