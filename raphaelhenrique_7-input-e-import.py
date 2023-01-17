print(" < Seja bem-vindo ao nosso site, por favor efeturar o seu cadastro > \n")

while True:
    usuario = input("Digite seu usuário:")
    if len(usuario) > 0 and usuario.isalpha():
     cadastro = usuario.lower()
     print("Obrigado!", cadastro, "usuário cadastrado!\n")
     break
    elif len(usuario) <= 0:
        print("Campo de usuário está vazio, por favor, tente novamente!")        
    else:
     print("Nome de usuário inválido! Por favor, não utilizar caracteres especiais ou números!")
    
while True:
    telefone = input("Digite seu telefone sem traço, com DDI e DDD:")    
    if len(telefone) >= 13 and telefone.startswith("55") and telefone.isdigit():
        cadastro1 = telefone
        print("Obrigado! número de celular:", cadastro1, "cadastrado!")
        break
    elif len(telefone) == 12 and telefone.startswith("55") and telefone.isdigit():
        cadastro2 = telefone
        print("Obrigado! número residencial:", cadastro2,"cadastrado!")
        break
    elif len(telefone) <= 0:
        print("Campo de telefone está vazio, por favor, tente novamente!")
    else:
        print("Este número de telefone não existe! Por favor, tente novamente!")
import re
while True:
    email = input("Digite seu email:")
    match = re.findall("[\w.]+@[\w.]+.com", email)
    if match:
     cadastro3 = email.lower()
     print("Obrigado! email:", cadastro3, "cadastrado!")
     break            
    else:
     print("Formato de e-mail inválido! Certifique-se de estar digitando corretamente!")
import math
print("Raiz quadrada de 25 =",int(math.sqrt(25)))
print("Pi =",float(math.pi))
import datetime
calendário = datetime.datetime.now()
print("Relógio do sistema =", calendário)
print("Formato brasileiro =", calendário.day, calendário.month, calendário.year, calendário.strftime("%A"), calendário.strftime("%H"), calendário.strftime("%M"), calendário.strftime("%p"))
import platform
print(platform.system(), platform.release())