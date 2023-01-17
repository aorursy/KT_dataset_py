Edad=778
if Edad>=18:
    print("eres mayor de edad")
else:
    print("eres menor de edad")
minutos=270

if minutos%60>0:
    horas=minutos//60+1
else:
    horas=minutos/60
pago=horas*15
print("Estuviste",minutos, "minutos, paga la cantidad de",pago,"pesos por", horas,"horas")
numero=6
if numero%2==0:
    print("el numero", numero, "es par")
else:
    print("el numero", numero, "es par")





#Ejercicio Hecho por mi

docenas =4
descuento1 = .85
descuento2 = .90
precio = 10

if docenas == 0:
    print("No compraste ninguna docena")

elif docenas == 1:
    print("Debido a que compraste", docenas, "docena el monto de compra es", precio*docenas,", el descuento del 10% seria de", precio*docenas*.1, ", y el monto final a pagar es de" , precio*docenas*descuento2, "dolares.")
    
elif docenas == 2 or docenas == 3:
    print("Debido a que compraste", docenas, "docenas el monto de compra es", precio*docenas,", el descuento del 15% seria de", precio*docenas*.15, ", y el monto final a pagar es de" , precio*docenas*descuento1, "dolares.")
    
else:
    print("Debido a que compraste", docenas, "docenas el monto de compra es", precio*docenas,", el descuento del 15% seria de", precio*docenas*.15, ", y el monto final a pagar es de" , precio*docenas*descuento1, "dolares. Productos de regalo: ", docenas-3)

    
#Ejercicio hecho por el profe
cantidad=200
precioUnit=10
descuento=.10
obsequio=0
if cantidad >= 36:
    descuento =.15
    obsequio= cantidad//12-3

total=cantidad*precioUnit
descuento_total = total*descuento
total_pagar=total-descuento_total
print("Producto ","Cantidad","Subotal")
print("Producto1 ",cantidad,"     ",total)
print("Descuento aplicable",descuento*100,"%")
print("Total descuento",descuento_total)
print("Total a pagar",total_pagar)
print("unidades de obsequio",obsequio)
numero=777

centenas=(numero-numero%100)/100
print(centenas)

unidades=numero%10
print(unidades)

if centenas==unidades:
    print("el numero ",numero, "es un palindromo")
else:
    print("el numero ",numero, "no es un palindromo")