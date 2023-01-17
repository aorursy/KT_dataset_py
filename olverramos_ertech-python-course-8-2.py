from collections import deque
denominaciones = [100000, 50000, 20000, 10000, 5000, 2000, 1000]
bandejas = {}
for valor in denominaciones:
    bandejas[valor] = 0
bandejas[100000] = 5 
bandejas[50000] = 2
bandejas[20000] = 1
bandejas[10000] = 3
bandejas[5000] = 2
bandejas[2000] = 0
bandejas[1000] = 5
import random
def generate_serial(value):
 
    if value == 100000:
        serial = '12'
    elif value == 50000:
        serial = '18'
    elif value == 20000:
        serial = '15'
    elif value == 10000:
        serial = '21'
    elif value == 5000:
        serial = '26'
    elif value == 2000:
        serial = '34'
    elif value == 1000:
        serial = '45'
    else:
        return
    for _ in range(6):
        serial += str(random.randrange(10))

    return serial
def calcular_billetes(monto_global, valor):
    if valor > monto_global:
        return None, monto_global
    if valor == 0:
        return [], monto_global
    for denominacion, cantidad in bandejas.items():
        if cantidad > 0:
            resultado = []
            if denominacion <= valor:
                bandejas[denominacion] -= 1
                monto_global -= denominacion
                resultado = [(denominacion, generate_serial(denominacion))]
                billetes = calcular_billetes(monto_global, valor - denominacion)
                if billetes is not None:
                    resultado.extend(billetes)
                    return resultado, monto_global
                bandejas[denominacion] += 1
                monto_global += denominacion
                
    return None, monto_global

bandejas = { 100000: 5, 50000: 2, 20000: 1, 10000: 3, 5000: 2, 2000: 0, 1000: 5 }
monto_global = 665000
billetes, monto_global = calcular_billetes(monto_global, 478000)
print(billetes)
print(bandejas)
print(monto_global)