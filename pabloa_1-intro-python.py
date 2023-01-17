a = 10.5

a
type(a)
b = "hola"

b
type(b)
c = True

c
type(c)
aca_no_hay_nada = []

aca_no_hay_nada
type(aca_no_hay_nada)
aca_hay_cinco = [True, "dos", 3, [1,2,3,4], 'ultimo']

aca_hay_cinco
type(aca_hay_cinco)
aca_hay_cinco[0]
aca_hay_cinco[4]
aca_hay_cinco[-1]
aca_hay_cinco[:2]
aca_hay_cinco[-2:]
aca_hay_cinco[3:]
aca_hay_cinco[1:4]
a
a+5
2*a+10
b = [1,2,3]
b + b
b * 4
for elemento in b:

    print(elemento)
for elemento in b:

    print(elemento + elemento)
c = []

for elemento in b:

    c += [elemento + elemento]

c