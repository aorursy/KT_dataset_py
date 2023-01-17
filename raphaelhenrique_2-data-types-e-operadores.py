str = "Texto escrito é um exemplo de: string"
print(type(str), str, "\n")

integer = 10
print(type(integer), "Integer representa números inteiros, como:", integer, "\n")

float = 10.5
print(type(float), "Float representa casas decimais ou valores quebrados, como:", float, "\n")

complex = 1j
print(type(complex), "Complex é formado por valor real e imaginário:", complex, "\n")

list = ["a1","b2","c3","d4"]
print(type(list), "O valor de lista são compostas por colchetes [], como:", list, "\n")

tuple = ("a1","b2","c3","d4")
print(type(tuple), "O valor de Tuple são compostos por parênteses(), como:", tuple, "\n")

distancia = (range(10))
print(type(distancia), "Range é determinado pelo seu alcance:", distancia, "\n")

dict = {"nome" : "Raphael", "sobrenome" : "Santos", "profissão" : "Estudante", "projeto" : "Python3"}
print(type(dict), "Dicionário estabelece conjunto de valores:", dict, "\n")

set = {"a1", "b2", "c3", "d4"}
print(type(set),"Set criar uma lista por ordem aleatória e são compostos por chaves{}, como:" ,set, "\n")

frozen = frozenset({"a1", "b2", "c3", "d4"})
print(type(frozen), "Frozenset é uma função imutável, não autorizando modificações:" ,frozen, "\n")

bool = True
print(type(bool),"Bool traduz valores em Verdadeiro ou Falso:" ,bool, "\n")

bytes = b'''String, 
 em 
 Bytes'''
print(type(bytes),"Bytes converte as informações para linguaguem de máquina e são objetos imutáveis:" , bytes, "\n")

bytearray = bytearray(3)
print(type(bytearray),"Bytearray também convertem para linguaguem de máquina, porém trabalham com matrizes:" , bytearray, "\n")

memory = memoryview(bytes)
print(type(memory),"Memoryview retorna o valor de memória de um objeto:" , memory)
Adição = 10 + 5
print("Adição: 10 + 5 =", Adição)

Subtração = 10 - 5
print("\nSubtração: 10 - 5 =", Subtração)

Multiplicação = 10 * 5
print("\nMultiplicação: 10 * 5 =", Multiplicação)

Divisão = 10 / 5 
print("\nDivisão: 10 / 5 =", Divisão, "| Número neste caso sai em formato Float, com casa decimal")

Resto = 10 % 5
print("\nResto: 10 % 5 =", Resto)

Potência = 10 ** 5
print("\nPotência: 10 ** 5 =", Potência)

Parte_Inteira = 10 // 5 
print ("\nParte Inteira: 10 // 5 =", Parte_Inteira, "| Número neste caso sai em formato integer, sem casa decimal")
x = 10
y = 5
Igual = "x = 10 e y = 5"
print("'=' Começamos por igual, atribuindo valores para x e y:", Igual)

Mais_Igual = 10
Mais_Igual += 5
print("\n'+=' Equivale a (x = x + y), repetindo e acrescentando ao valor:", Mais_Igual)

Menos_Igual = 10
Menos_Igual -= 5
print("\n'-=' Equivale a (x = x - y), repetindo e diminuindo ao valor:", Menos_Igual)

Multi_Igual = 10
Multi_Igual *= 5
print("\n'*=' Equivale a (x = x * y), repetindo e multiplicando o valor:", Multi_Igual)

Div_Igual = 10
Div_Igual /= 5
print("\n'/=' Equivale a (x = x / y), repetindo e dividindo o valor em formato Float:", Div_Igual)

Porcentagem_Igual = 10
Porcentagem_Igual %= 5
print("\n'%=' Módulo retorna o resto da divisão, neste caso entre 10/5 =", Porcentagem_Igual)

Parte_Inteira_Igual = 10
Parte_Inteira_Igual //= 5
print("\n'//=' Floor Division equivale a uma divisão em parte inteira, seu formato resulta em integer:", Parte_Inteira_Igual)

Expoente_Igual = 10
Expoente_Igual **= 5
print("\n'**=' Equivale a (x^y) expoente:", Expoente_Igual)
set_exemplo = {0,1,2,3}
set_um = {0,1,2,3}
set_dois = {2,3,4,5}
set_compara = set_um
set_compara &= set_dois
print("'&' ou 'Bitwise And' serve para comparar combinação de valores em comum:", set_exemplo,"&", set_dois, "=", set_compara)

Barra_Igual = True
Barra_Igual |= False
print("\n'|' ou 'Bitwise Or' faz parte da matéria de raciocínio lógico, dentro da tabela verdade 'Ou' possui uma combinação de resultados entre valores verdadeiros e verdadeiro ou falso:", "True or False =", Barra_Igual)

Circunflexo_Igual = True
Circunflexo_Igual ^= True
print("\n'^' ou 'Bitwise Xor' representado por 'Ou...Ou', inválida os mesmos resultados V ou V e F ou F:", "True xor True =", Circunflexo_Igual)

Diferente = (~x)
print("\n'~' ou 'Bitwise Not' inverte em bit o seu valor:","10 em bit é 1010 '~' -11 em bit é 0101 =" , Diferente)
Igual2 = 10 == 5
print("'==' representa uma validação de valor, retornando verdadeiro ou falso caso os valores sejam comparativamente iguais:", "10 == 5 =", Igual2)

Nao_Igual = 10 != 5
print("\n'!=' não igual parte do mesmo princípio, porém retornando verdadeiro ou falso caso os valores sejam comparativamente diferentes:", "10 != 5 =", Nao_Igual)

Maior_que = 10 > 5
print("\n'>' é um símbolo matématico que representa 'maior que', retorna o valor de verdadeiro ou falso caso sejam comparativamente maior do que o outro:", "10 > 5 =", Maior_que)

Menor_que = 10 < 5
print("\n'<' é um símbolo matématico que representa 'menor que', retorna o valor de verdadeiro ou falso caso sejam comparativamente menor do que o outro:", "10 < 5 =", Menor_que)

Maior_igual = 10 >= 10
print("\n'>=' representa 'maior ou igual', retorna verdadeiro ou falso, todavia ele aceita valores em igualdade:", "10 >= 10 =", Maior_igual)

Menor_igual = 10 <= 10
print("\n'<=' representa 'menor ou igual', retorna verdadeiro ou falso, mas também aceita valores em igualdade:", "10 <= 10 =", Menor_igual)
E_exemplo = x > 5 and 5 < x
print("'And' significa 'E' em raciocínio lógico retornando valor de verdadeiro ou falso. Porém, apenas se ambos os valores forem verdade:", "10 é maior que 5 'E' 5 é menor que 10 =", E_exemplo)

Ou_exemplo = x > 5 or 5 > x
print("\n'Or' significa 'OU' em raciocínio lógico retornando valor de verdadeiro ou falso. É falso apenas quando ambos valores forem falsos, aceita portanto ambas verdadeiras ou apenas um valor verdadeiro:", "10 é maior que 5 'OU' 5 é maior que 10 =", Ou_exemplo)

Nao_exemplo = not(x > 5 and 5 < x)
print("\n'Not' reverte o resultado da afirmação:", "10 não é maior que 5 'E' 5 não é menor que 10 =", Nao_exemplo)
IS = x is y
print("'Is' ou 'É' representa um operador que retorna verdadeiro ou falso para variáveis que fazem parte do mesmo objeto:","10 is 5 =" ,IS)

IS_Not = x is not y
print("\n'Is Not' ou 'Não é' segue o mesmo príncipio, porém com valores que se mostrarem diferentes:", "10 is not 5 =", IS_Not)
lista = ["verdadeiro", "falso", "neutro"]
In_exemplo = ("verdadeiro" in lista)
print("'In' ou 'Dentro' retorna verdadeiro ou falso como forma de conferir a filiação de um valor dentro de uma lista específica:", "['verdadeiro', 'falso', 'neutro'], verdadeiro 'IN' lista =", In_exemplo)

Not_In_exemplo = ("falso" not in lista)
print("\n'Not In' ou 'Não dentro' segue o mesmo príncipio, porém retorna o valor contraditório ao da sua filiação, ou seja, identifica o valor sem uma correspondência:", "['verdadeiro', 'falso', 'neutro'], falso 'NOT IN' lista =", Not_In_exemplo)