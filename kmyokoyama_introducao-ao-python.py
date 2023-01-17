import sys # Realiza a importação da biblioteca built-in sys.

sys.version
x = 1

print(x)

x = "uma string"

print(x)
um_inteiro = 100
um_real = 150.345 # Separador decimal eh ponto, nao virgula.
um_booleano = True # Sempre True ou False. A primeira letra maiuscula.
uma_string = "isso eh uma string" # Aceitamos aspas duplas.
outra_string = 'isso eh outra string' # E tambem aspas simples.
str1 = "hello"
str2 = "world"

str_final = str1 + ", " + str2 + "!"

print(str_final)
print(f"um_inteiro = {um_inteiro}, um_real = {um_real}")
print(f"um_booleano = {um_booleano}")
print(f"uma_string = {uma_string}, outra_string = {outra_string}")
lista_vazia = [] # Criamos uma lista vazia com os colchetes vazios.

lista_um_elemento = [10] # Apenas o elemento entre colchetes.

lst1 = [1, 2, 3]
lst2 = ["hello", "world", 4, 5]

print(f"lista_vazia = {lista_vazia}, lista_um_elemento = {lista_um_elemento}")
print(f"lst1 = {lst1} e lst2 = {lst2}")
lst3 = lst1 + lst2

print(lst3)
print(f"O primeiro elemento de lst1 eh {lst1[0]} e o segundo eh {lst1[1]}") # Contagem comeca em zero.

print(f"Não podíamos deixar de dizer também: {lst2[0]}, {lst2[1]}!")

lst2[1] = "Python"

print(f"Ou melhor: {lst2[0]}, {lst2[1]}!")
lst3 = [10, 20, 30, 40, 50]

print(lst3[1:4:2]) # Comeca na posicao 1, vai ate a posicao 3, de 2 em 2.
print(lst3[:4:2]) # Podemos omitir o inicio. Nesse caso, 0 eh assumido.
print(lst3[:4]) # Ainda mais comum, eh omitir o passo. Nesse caso, de 1 em 1 eh assumido.
print(lst3[1:]) # E tambem podemos omitir o fim. Nesse caso, o fim da lista eh assumido.
print(lst3[:]) # Podemos ate mesmo omitir tudo. Nesse caso, inicio eh assumido 0 e fim eh assumido como fim da lista.
print(lst3[::-1]) # Um caso util eh omitirmos inicio e fim e fazer passo -1. Nesse caso, obtemos a lista invertida.
lst1 = [1, 2, 3, 4] # Inicializando lst1 no comeco do bloco.

print(f"Lista lst1: {lst1}")

# Obtendo tamanho.
tamanho = len(lst1)
print(f"Tamanho da lista lst1: {tamanho}")

# Adicionando um elemento ao final da lista.
lst1.append(10)
print(f"Lista lst1 com 10 adicionado no final: {lst1}")

# Inserindo em uma determinada posicao.
lst1.insert(4, 8) # Adiciona o elemento 8 na posicao 4.
print(f"Lista lst1 com 8 adicionado na posição 4: {lst1}")

# Removendo um elemento da lista.
lst1.remove(8) # Remove primeira ocorrencia do 8.
print(f"Lista lst1 após remover elemento 8: {lst1}")

# Invertendo a lista.
lst1.reverse()
print(f"Lista lst1 invertida: {lst1}")

# Ordenando a lista.
lst1.sort()
print(f"Lista lst1 ordenada: {lst1}")
tupla_vazia = () # Criamos uma tupla vazia com apenas os parênteses vazios.

tupla_um_elemento = (10,) # Criamos uma tupla com um unico elemento colocando uma virgula apos o elemento.

tpl1 = (1, 2, 3)
tpl2 = ("hello", "tuples", 4, 5)

print(f"tupla_vazia = {tupla_vazia}, tupla_um_elemento = {tupla_um_elemento}")
print(f"tpl1 = {tpl1} e tpl2 = {tpl2}")
print(f"O primeiro elemento de tpl1 eh {tpl1[0]} e o segundo eh {tpl1[1]}") # Contagem comeca em zero.

print(f"Não podíamos deixar de dizer de novo: {tpl2[0]}, {tpl2[1]}!")
try:
    tpl2[0] = "bye" # Gera um erro do tipo TypeError.
except TypeError as err:
    print(err)
tpl1 = (10, 20, 30, 40, 50)

print(f"Tupla tpl1: {tpl1}")

# Obtendo tamanho.
tamanho = len(tpl1)
print(f"Tamanho da tupla tpl1: {tamanho}")

print(tpl1[1:4:2]) # Comeca na posicao 1, vai ate a posicao 3, de 2 em 2.
print(tpl1[:4:2]) # Podemos omitir o inicio. Nesse caso, 0 eh assumido.
print(tpl1[:4]) # Ainda mais comum, eh omitir o passo. Nesse caso, de 1 em 1 eh assumido.
print(tpl1[1:]) # E tambem podemos omitir o fim. Nesse caso, o fim da lista eh assumido.
print(tpl1[:]) # Podemos ate mesmo omitir tudo. Nesse caso, inicio eh assumido 0 e fim eh assumido como fim da lista.
print(tpl1[::-1]) # Um caso util eh omitirmos inicio e fim e fazer passo -1. Nesse caso, obtemos a lista invertida.
set_vazio = {} # Criamos um conjunto vazio com as chaves vazias.

set_um_elemento = {10} # Apenas um elemento entre as chaves.

set1 = {1, 2, 3}
set2 = {"hello", "tuples", 4, 5}

print(f"set_vazio = {set_vazio}, set_um_elemento = {set_um_elemento}")
print(f"set1 = {set1} e set2 = {set2}")
dic1 = {"Maria": 20, "Joao": 21}

idade_maria = dic1["Maria"]
idade_joao = dic1["Joao"]

print(f"A idade de Maria é {idade_maria}")
print(f"A idade de João é {idade_joao}")
lst1 = [1, 2, 3, 4, 5]
lst_comp = [x**2 for x in lst1] # Para cada elemento x de lst1, criamos lst_comp com os elementos x ao quadrado.

print(lst_comp)

set1 = {1, 2, 3, 4, 5}
set_comp = {x**3 for x in set1}

print(set_comp)
lst1 = [1, 2, 3, 4, 5]
lst_comp_cond = [x**2 for x in lst1 if x % 2 == 1] # Para cada elemento impar x de lst1, criamos lst_comp_cond com os elementos x ao quadrado.

print(lst_comp_cond)
lst1 = [1, 2, 3, 4, 5]
lst2 = [10, 20, 30, 40, 50]

lst_comp_complexa = [(x, y) for x in lst1 for y in lst2 if x * y > 100] # Pega os pares ordenados de lst1 x lst2 para os quais x * y > 100.

print(lst_comp_complexa)
x = 1
y = 2

if x == 0:
    print("x é zero")
elif x == 1 and y == 2:
    print("x é 1 e y é 2")
else:
    print("x não é 1 e y não é 2")
x = 10
while x > 0:
    print(f"x = {x}")
    x -= 1
print(list(range(1, 10, 2))) # Vai de 1 a 10 (exclusive) de 2 em 2.
print(list(range(1, 10))) # Vai de 1 a 10 (exclusive) de 1 em 1 (passo default).
print(list(range(10))) # Vai de 0 a 10 (exclusive) de 1 em 1 (passo default).
for i in range(10):
    print(f"i = {i}")
lst1 = [1, 2, 3, 4, 5]
tpl1 = (1, 2, 3, 4, 5)
set1 = {1, 2, 3, 4, 5}

print("\nNa lista:")
for elem in lst1:
    print(f"elem = {elem}")

print("\nNa tupla:")
for elem in tpl1:
    print(f"elem = {elem}")
    
print("\nNo conjunto:")
for elem in set1:
    print(f"elem = {elem}")
lst1 = [10, 20, 30, 40, 50]

for i, elem in enumerate(lst1):
    print(f"Posição {i} de lst1 = {elem}")
def quadrado(x):
    return x**2

print(f"2² = {quadrado(2)}")
print(f"3² = {quadrado(3)}")
def soma(x, y=1):
    return x + y

print(f"soma(2) = 2 + 1 = {soma(2)}")
print(f"soma(2, 3) = 2 + 3 = {soma(2, 3)}")
def potencia(expoente, base):
    return base**expoente

print(f"2³ = potencia(base=2, expoente=3) = {potencia(base=2, expoente=3)}")
print(f"3² = potencia(expoente=2, base=3) = {potencia(expoente=2, base=3)}")
def mostrar(primeiro, segundo, *args, **kwargs):
    print(f"O primeiro argumento é {primeiro}")
    print(f"O segundo argumento é {segundo}")
    
    for arg in args:
        print(f"Cada argumento coletado pela tupla é {arg}")
        
    for kwarg, valor in kwargs.items():
        print(f"Cada argumento {kwarg} coletado pelo dicionário é {valor}")
        

mostrar("hello", "world", "um", "argumento", "diferente",
     keyword_arg1="um argumento", keyword_arg2="diferente")
class Pessoa:
    def __init__(self, nome, idade):
        self.nome = nome
        self.idade = idade
        
    def apresentacao(self):
        return f"Olá, meu nome é {self.nome} e tenho {self.idade} anos"


maria = Pessoa("Maria", 20)
joao = Pessoa("João", 21)

print(maria.apresentacao())
print(joao.apresentacao())
class Funcionario(Pessoa):
    def __init__(self, nome, idade, ocupacao):
        super().__init__(nome, idade)
        self.ocupacao = ocupacao
        
    def apresentacao(self):
        return f"Olá, meu nome é {self.nome} e sou {self.ocupacao}"


maria = Funcionario("Maria", 20, "desenvolvedora")
joao = Funcionario("João", 21, "testador")

print(maria.apresentacao())
print(joao.apresentacao())
class Cargo:
    def __init__(self, cargo, carga_horaria):
        self._cargo = cargo
        self._carga_horaria = carga_horaria
        
    @property
    def carga_horaria(self):
        return str(self._carga_horaria) + " horas/semana"
    
    @carga_horaria.setter
    def carga_horaria(self, nova_carga_horaria):
        self._carga_horaria = nova_carga_horaria


estagiario = Cargo("Estagiário", 20)

print(f"Carga horária original do estagiário: {estagiario.carga_horaria}")

estagiario.carga_horaria = 10 # O final do semestre estava chegando...

print(f"Carga horária de final de semestre do estagiário: {estagiario.carga_horaria}")