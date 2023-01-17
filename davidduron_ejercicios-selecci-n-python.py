edad=23

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

Cantidad=200

Precio=10

Descuento=.10

Obsequio=0

if Cantidad>=36:

    Descuento=.15

    Obsequio=Cantidad//12-3

    

Total=Cantidad*Precio

Descuento_Total= Total*Descuento

Total_Pagar=Total-Descuento_Total

print("Producto ","Cantidad","Subtotal")

print("Producto1 ",Cantidad,Total)

print("Descuento ",Descuento*100,"%")

print("Total Descuento",Descuento_Total)

print("Total a Pagar",Total_Pagar)

print("Unidades de Obsequio",Obsequio)
número=888

print(número)

if número<99 or número>1000:

    print("Ingresar número de 3 cifras")

if número>99 or número<1000:

    unidades=número%10

    centenas=int(número/100)

    if (unidades==centenas):

        print("El número es igual al inverso")

    else:

        print("El número no es igual al inverso")

   

        


