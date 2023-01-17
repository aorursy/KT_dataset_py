# Listagem 3.1 - Hipotenusa

from math import sqrt

def hipotenusa(x, y):
    return sqrt(x ** 2 + y ** 2)
    
def main():
    a = float( input("a: ") )
    b = float( input("b: ") )
    print("Hipotenusa: ", hipotenusa(a, b) )
    
main()
# Versão alternativa

def hipotenusa(x, y):
    hyp = sqrt(x ** 2 + y ** 2)
    return hyp
def teste_escopo_local(x):
    tel = x ** 2
    return tel

teste_escopo_local(10)

print(tel)
def teste_escopo_global(x):
    global teg 
    teg = x ** 2
    return teg

teste_escopo_global(10)

print(teg)
def main():
    a = float( input("a: ") )
    b = float( input("b: ") )
    hip = hipotenusa(a, b)
    print("Hipotenusa: ", hip)
# Listagem 1.1

from math import pi
r = 12
area = pi * r ** 2
circ = 2 * pi * r
print("A área do círculo com raio", r, "eh", area)
print("A circunferência do círculo com raio", r, "eh", circ)
# Solução

from math import pi

def area(r):
    return pi * r ** 2

def circunferencia(r):
    return 2 * pi * r

def main():
    r = float(input("Entre com um raio: "))
    print("A área do círculo com raio", r, "eh", area(r))
    print("A circunf. do círculo com raio", r, "eh", circunferencia(r))

main()
def media(a,b):
    return (a+b)/2

def main():
    num1 = float(input("Entre com um numero: "))
    num2 = float(input("Entre com outro numero: "))
    print("A media entre esses números eh ", media(num1, num2) )
    
main()
# Em um arquivo chamado temp.py

def fahrenheit2celsius(tf):
    return (tf - 32) * 5/9
    
def main():
    tf = float(input("Entre com uma temperatura em Fahrenheit: "))
    print("{} Fahrenheit equivale a {} Celsius"
                          .format(tf, fahrenheit2celsius(tf)))

main()
def fahrenheit2celsius(tf):
    return (tf - 32) * 5/9

def celsius2kelvin(tf):
    return tf + 273.15

def fahrenheit2kelvin(tf):
    return celsius2kelvin( fahrenheit2celsius(tf) )

def main():
    tf = float(input("Entre com uma temperatura em Fahrenheit: "))
    print("{} Fahrenheit equivale a {} Celsius"
                          .format(tf, fahrenheit2celsius(tf)))
    
    print("{} Fahrenheit equivale a {} Kelvin"
                          .format(tf, fahrenheit2kelvin(tf)))

main()
# Em um arquivo chamado heart.py

def bat_card(idade):
    return 208 - 0.7 * idade

def main():
    idade = int( input("Entre com sua idade (em anos): ") )
    print("Seu batimento cardiaco maximo por minuto estimado eh", bat_card(idade) )
main()
# Em um arquivo chamado heron.py

from math import sqrt

def heron(a,b,c):
    p = (a + b + c) / 2
    return sqrt(p * (p - a) * (p - b) * (p - c))
    
def main():
    a = float( input("Entre com o lado a do triangulo: ") )
    b = float( input("Entre com o lado b do triangulo: ") )
    c = float( input("Entre com o lado c do triangulo: ") )
    print("A area do triangulo eh ", heron(a,b,c) )
    
main()
# Em um arquivo chamado juros.py

def saldo(p, r, t):
    return p * (1 + r) ** t
    
def main():
    p = float( input("Entre com o valor do principal: ") )
    r = float( input("Entre com a taxa de juros (ex: 0.04): ") )
    t = float( input("Entre com o tempo de rendimento (em anos): ") )
    #r = 0.015
    #t = 46.55555
    print("Em {} anos seu saldo sera de {}". format(t, saldo(p,r,t)) )
    
main()