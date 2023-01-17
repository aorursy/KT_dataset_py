edad=21

if edad>=18:

    print("Eres mayor de edad")

else:

    print("Eres menor de edad")
minutos=121



if minutos%60>0:

    horas=minutos//60+1

else:

    horas=minutos/60

pago=horas*15

print("Estuviste",minutos, "minutos, paga la cantidad de",pago,"pesos por", horas,"horas")

numero= 187

if numero%2==0:

    print("el numero", numero, "es par")

else:

    print("el numero", numero, "es impar")

cantidad= 300

precio= 10

descuento= .10

obsequio=0

if cantidad >= 36:

    descuento= .15

    obsequio=cantidad//(12-3)



total= cantidad * precio

descuento_total= total * descuento

total_pagar= total - descuento_total



print("Producto","Cantidad", "Subtotal")

print("Producto1", cantidad, total)

print(obsequio)
numero=123



centenas=(numero-numero%100)/100

print(centenas)

unidades=numero%10

print(unidades)



if centenas==unidades:

    print("el numero", numero, "es igual al derecho y al reves")

else:

    print("El numero", numero, "no es igual al derecho y al reves")