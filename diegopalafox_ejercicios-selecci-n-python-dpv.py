edad=15

if edad>=18:

    print("Esta persona si es mayor de edad")

else:

    print("Esta persona no es mayor de edad")
minutos=121

horas= minutos//60

if minutos%60>0:

    horas=minutos//60+1



pago=horas*15

print("La persona va a pagar",pago,"pesos porque estuvo",minutos,"minutos")

numero=6

if(numero%2)==0:

    print("El numero es par")

else:

   print("El numero es impar")

precio=10

productos=60

docenas=(productos//12)

monto=precio*productos



if(docenas<=2):

    descuento=monto*(.1)

else:descuento=monto*(.15)



if(docenas>=4):

    obsequio=docenas-3

else:obsequio=0

    

montofinal=precio*productos-descuento



print("La persona compró",productos,"artículos, equivalente a",docenas,"docenas, por lo que el monto inicial a pagar sería de",monto,"$,por la cantidad de docenas que completó va a tener un descuento de",descuento,"$ entonces el subtotal es de",montofinal,"$. Ademas, por la cantidad de docenas compradas obtiene",obsequio,"articulo de regalo")

numero=263



if(numero>99 and numero<1000):

    primerdigit=numero%10

    tercerdigit=int(numero/100)



    if(primerdigit==tercerdigit):

        print("El numero",numero,"es igual al reves")

    else:print("el numero",numero,"no es igual al reves")

        

else:print("El numero debe ser de tres digitos")


