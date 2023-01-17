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

# monto de la compra, 
#el monto del descuento, 
#el monto a pagar 
#el número de unidades de obsequio por la compra de cierta cantidad de docenas del producto

docenas = 6

def get_info(docena):
    descuentoA= .10
    descuentoB= .15
    # Precio sugerido en pesos mexicanos
    precio = 100
    
    if docena > 3:
        # se aplica desc15 y se regala unidad
        monto_real = docenas*precio
        monto_descuento = monto_real*descuentoB
        monto_total = monto_real - monto_descuento
        unidades = docenas - 3
        
    else:
        # se aplica desc10 y no se regala unidad
        monto_real = docenas*precio
        monto_descuento = monto_real*descuentoA
        monto_total = monto_real - monto_descuento
        unidades = 0
        
    print("El monto de la compra es: "+str(monto_real)+", el monto del descuento es: "+str(monto_descuento)+", el monto a pagar es: "+str(monto_total)+" y la unidades regalada fueron: "+str(unidades))
    
get_info(docenas)
num = 134
num_inv = str(num)[::-1]

if len(str(num)) == 3:
    if str(num) == num_inv:
        print("los numeros son identicos")
    else:
        print("los numeros no son identicos")
else:
    print("agrege un numero de 3 cifras, por favor")