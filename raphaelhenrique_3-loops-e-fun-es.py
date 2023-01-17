if 5 < 10: #Impresso
    print("<if> Se 5 for menor que 10, você vai ler essa mensagem.") 
    print("<if> Se 5 for maior que 10, a mensagem foi ocultada pelo código.")
if 5 > 10: #Oculto
    print("Você não vai ler essa mensagem.")
if 5 >= 10: #Oculto
    print("Você não vai ler essa mensagem.")
elif 5 <= 10: #Impresso
    print("<if> Se 5 for maior ou igual a 10, a mensagem foi ocultada pelo código.")
    print("<elif> Caso contrário, se 5 for menor ou igual a 10, está mensagem será escrita.")
if 5 > 10: #Oculto
    print("Você não vai ler essa mensagem.")
elif 5 >= 10: #Oculto
    print("Você não vai ler essa mensagem.")
else: #Impresso
    print("<if> Se 5 for maior que 10, a mensagem foi ocultada pelo código.")
    print("<elif> Se 5 for maior ou igual a 10, a mensagem foi ocultada pelo código.")
    print("<else> Caso não achar correspondência nas anteriores, está mensagem foi impressa.")
if 5 != 10: #Impresso
    print("<if> Se 5 for diferente de 10, está mensagem vai ter prioridade sob as demais.")
    print("<elif> Caso contrário, se 5 for menor que 10 vou esperar 'if' retornar falso, oculto.")
    print("<else> Se não tiver correspondência de 'if' e 'elif' fico como backup, oculto.")
elif 5 < 10: #Oculto
    print("Estou certo, mas em segundo plano.")
else: #Oculto
    print("Se for preciso estou por aqui.")
if 5 != 10: #Incompleto
    pass
contar = 1
while contar < 11:
    print(contar)
    contar += 1
contar = 1
while contar < 11:
    print(contar)
    if (contar == 5):
        break
    contar += 1
contar = 0
while contar < 10:
    contar += 1
    if contar == 5:
        continue
    print(contar)
contar = 0
while contar < 10:
    contar += 1
    print(contar)
else:
    print("Já estou mais calmo!")
string = "String"

for a in string:
 print(a)
lista = ["exemplo", "lista"]

for b in lista:
 print(b)
dicionario = {"dicionário" : "exemplo", "dict" : "data type", "index" : "localização"}

for c in dicionario:
 print(c)
 if c == "dict":
     break
for c in dicionario:
 if c == "dict":
     continue
 print(c)
for d in range(1,11,2):
     print(d)
tuple = ("questão 1:", "questão 2:", "questão 3:")
set = {"a) b) c) d) e)"}

for e in tuple:
    for f in set:
        print(e,f)