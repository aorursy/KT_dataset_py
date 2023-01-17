edad=17
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
print("Estuviste", minutos, "minutos, paga una cantidad de",pago,"pesos por", horas,"horas")

#Resolver 4 y 5 de tarea

numero= 5
print(numero/2)
print(numero//2)
print(numero/2-numero//2) #modulo es el residuo de una division
numero= 187
if numero%2==0:
    print("el numero", numero, "es par")
else:
    print("el numero", numero, "es impar")
docenas=5
costo=15
monto=docenas*costo
print("Monto de compra es de", monto, "pesos")

if docenas<=3:
    print("Obtienes un 10% de descuento en tu compra", docenas, "docenas")
else:
    print("Obtiene sun 15 de descuento en tu compra", docenas, "docenas")
    
if docenas<=3:
    descuento=monto*.90
    print("La canitidad a pagar es de", descuento, "pesos")
    
if docenas>=4:
    descuento=monto*.85
    print("La cantidad a pagar es de", descuento, "pesos")
    
if docenas>=4:
    print("Obtienes", docenas, "unidades gratis")
cantidad=200
precioUnit=10
descuento=.10
obsequio=0
if cantidad >=36:
    descuento=.15
    obsequio= cantidad//12-3
    
total=cantidad*precioUnit   
descuento_total= total*descuento
total_pagar= total-descuento_total

print("Producto 1", "Cantidad", "Subtotal")
print("Producto1", cantidad,"    " ,total)
print("Descuento aplicable", descuento*100,"%")
print("Total descuento",descuento_total )
print("Total a pagar", total_pagar)
print("unidades de obsequio", obsequio)
numero=333
#Esta funcion es palindromo

if numero==333:
    print("El numero", numero, "es palindromo")
else:
    print("El numero", numero, "no es palindromo")
    
numero=666

centenas=(numero-numero%100)/100
print(centenas)
unidades=numero%10
print(unidades)

if centenas==unidades:
    print("el numero",numero,"es igual al derecho y al reves")
else:
    print("el numero", numero,"no es igual al derecho y al reves")