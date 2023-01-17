arquivo = open('projeto_python3.txt', 'x')
print("open('C:\\Users\Public\Pasta_teste\projeto_teste.txt', 'x')")
arquivo.write("Olá, sejam bem-vindos ao arquivo projeto python 3!")
arquivo.close()

arquivo = open("projeto_python3.txt", "r")

print(arquivo.read())
arquivo = open("projeto_python3.txt", "a")

arquivo.write("\nSou um anexo, obrigado por me acrescentar ao texto!")
arquivo.close()

arquivo = open("projeto_python3.txt", "r")

print(arquivo.read())
import os
os.remove("projeto_python3.txt")

print(arquivo.read())
if os.path.exists("projeto_python3.txt"):
  os.remove("projeto_python3.txt")
else:
  print("Este arquivo já não existe mais!")