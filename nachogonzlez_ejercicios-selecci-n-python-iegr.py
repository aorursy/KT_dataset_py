edad=17

if edad >=18:

    print("Esta persona es mayor de edad")

else:

    print("Esta persona no es mayor de edad")
minutos=121

horas=minutos/60

if minutos%60>0:

    horas=minutos//60+1

else:

    horas=minutos/60

pago=horas*15

print("Estuviste", minutos, "minutos, paga una cantidad de", pago, "pesos por", horas, "horas")
numero=187

if numero%2==0:

    print("El numero", numero, "es par")

else:

    print("El numero", numero, "es impar")
Docena=5

Precio=10

if Docena>3:

    print("Docenas=", Docena, "uds")

    print("Precio original=", "$",Precio*Docena)

    print("Descuento=", "$",(Precio*Docena)*.15)

    print("Precio con descuento=", "$",(Precio*Docena)-Precio*.15)

    print("Obsequio(s)=", Docena -3, "uds")

else:

    print("Docenas=", Docena, "uds")

    print("Precio original=", "$",Precio*Docena)

    print("Descuento=", "$",(Precio*Docena)*.1)

    print("Precio con descuento=", "$",(Precio*Docena)-Precio*.1)

    print("No hay obsequio")
Numero=666



if str(Numero) == str(Numero)[::-1]:

    print("El numero", Numero, "es palindromo")

else:

    print("El numero", Numero, "no es palindromo")