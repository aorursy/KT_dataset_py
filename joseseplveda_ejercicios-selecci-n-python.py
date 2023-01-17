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

precio=10

productos=120

docenas=(productos//12)

monto=precio*productos



if(docenas<=2):

    descuento=monto*(.1)

else:descuento=monto*(.15)



if(docenas>=4):

    obsequio=docenas-3

else:obsequio=0

    

montofinal=precio*productos-descuento



print("Compró",productos,"artículos, equivalente a",docenas,"docenas, por lo que el monto inicial a pagar sería de",monto,"$,por la cantidad de docenas que completó va a tener un descuento de",descuento,"$ entonces el subtotal es de",montofinal,"$. Ademas, por la cantidad de docenas compradas obtiene",obsequio,"articulo de regalo")

numero=263



if(numero>99 and numero<1000):

    primerdigit=numero%10

    tercerdigit=int(numero/100)



    if(primerdigit==tercerdigit):

        print("Numero",numero,"es igual al reves")

    else:print("Numero",numero,"no es igual al reves")

        

else:print("El numero debe ser de tres digitos")