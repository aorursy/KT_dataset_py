edad=21

if edad>=18:

    print("Eres mayor de edad")

else:

    print("Eres menor de edad")



    
horas=3

minutos=15

pago=(horas*15)+(minutos*.25)



print("Por", horas , "horas y", minutos , "minutos, debera pagar", pago , "pesos")
numero=6



if numero%2==0:

    print("El numero", numero ,"es par")

else:

    print("El numero", numero ,"es impar")
docenas=5

precio=200

montodelacompra = docenas * precio



if(docenas>3):

    descuento=.15

    montodedescuento=montodelacompra*descuento

    montoapagar=montodelacompra-montodedescuento

    obsequio=docenas-3

    print("El monto de la compra es", montodelacompra , "pesos, el monto del descuento es", montodedescuento ,  "pesos, el monto a pagar es", montoapagar , "pesos, y el numero de unidades de obsequio es", obsequio , "unidades.")

    

if(docenas<=3):

    descuento=.10

    montodedescuento=montodelacompra*descuento

    montoapagar=montodelacompra-montodedescuento

    obsequio=0

    print("El monto de la compra es", montodelacompra , "pesos, el monto del descuento es", montodedescuento ,  "pesos, el monto a pagar es", montoapagar , "pesos, y el numero de unidades de obsequio es", obsequio , "unidades.")

    
numero=313

primero=(int(numero/100))

ultimo=(numero%10)

if numero<1000 and numero>99:

    if (int(numero/100))<(numero%10) or (int(numero/100))>(numero%10):

        print("Al reves no es igual")

    else:

        print("Al reves es igual")




