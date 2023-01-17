edad=15
if edad>=18:
    print("Eres mayor de edad")
else:
    print("Eres menor de edad")
minutos=61

if minutos%60>0:
    horas=minutos//60+1
else:
    horas=minutos/60
pago=horas*15
print("Estuviste",minutos, "minutos, paga la cantidad de",pago, "pesos por", horas,"horas")
    
numero= 11
if numero%2==0:
    print("el numero", numero, "es par")
else:
    print("el numero", numero, "no es par")

docenas=10
preciodocena=50

montocom = docenas*preciodocena

if docenas > 3:
    montodes=0.15*montocom
    obsequio=docenas-3
else:
    montodes=0.10*montocom
    obsequio=0
    
montopag=montocom-montodes

print("El monto de compra es", montocom, "pesos")
print("El monto de descuento es", montodes, "pesos")
print("El monto de pago es", montopag, "pesos")
print("Las unidades de obsequio por la compra son", obsequio)
numero=122

digito1 =numero // 100
digito2 =(numero % 100) // 10
digito3 =(numero % 100) % 10

if(digito3==digito1):
    print("El numero es igual al revés")
else:
    print("El numero NO es igual al revés")
