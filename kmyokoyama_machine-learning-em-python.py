import numpy as np # Convencao chamar de np o NumPy.

print(np.__version__)
np.random.seed(42)
arr1 = np.array([1, 2, 2.71, 3.14])
print(arr1)

arr2 = np.array([1, 2, 3, 4], dtype='float32')
print(arr2)
m1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(m1)
apenas_zeros = np.zeros(5, dtype=int) # Cria array 1x5 com todos valores inteiros iguais a 0.
print(f"apenas_zeros = {apenas_zeros}")

apenas_um = np.ones(5, dtype=int) # Cria array 1x5 com todos valores inteiros iguais a 1.
print(f"apenas_um = {apenas_um}")

apenas_um_ndim = np.ones((3, 3)) # Cria array 3x3 com todos valores iguais a 1.
print(f"apenas_um_ndim = \n{apenas_um_ndim}")

tudo_euler = np.full((2, 3), 2.71, dtype='float64') # Cria array 2x3 com todos valores iguais a 2.71.
print(f"tudo_euler = \n{tudo_euler}")

sequencia = np.arange(1, 10, 2) # Similar a funcao range do Python (inicio, fim, passo).
print(f"sequencia = {sequencia}")

espaco_linear = np.linspace(0, 1, 5) # Cria 5 valores igualmente espacados entre 0 e 1.
print(f"espaco_linear = {espaco_linear}")

aleatorios_uniforme = np.random.random((3, 2)) # Cria array 3x2 com valores uniformemente distribuidos entre 0 e 1.
print(f"aleatorios_uniforme = \n{aleatorios_uniforme}")

aleatorios_normal = np.random.normal(10, 2, (3, 3)) # Cria array 3x3 com valores normalmente distribuidos com media 10 e desvio-padrao 2.
print(f"aleatorios_normal = \n{aleatorios_normal}")

aleatorios_int = np.random.randint(1, 60, 6) # Cria array 1x6 com valores inteiros uniformemente distribuidos entre 1 inclusive e 60 exclusive.
print(f"aleatorios_int = {aleatorios_int}")

identidade = np.eye(3) # Cria identidade 3x3.
print(f"identidade = \n{identidade}")

vazio = np.empty((2, 3)) # Cria array 2x3 inicialmente "vazio": os valores serao lixo de memoria.
print(f"vazio = \n{vazio}")
transposta = aleatorios_uniforme.T # Um numpy.ndarray que eh a transposta do array.
print(f"A transposta de aleatorios_uniforme = \n{transposta}")

tamanho = aleatorios_uniforme.size # Quantidade de elementos no array (soma da quantidade de elementos em todas suas dimensoes).
print(f"Tamanho de aleatorios_uniforme = {tamanho}")

dimensoes = aleatorios_uniforme.shape # Tupla com o tamanho de cada dimensao.
print(f"Dimensões de aleatorios_uniforme = {dimensoes}")

numero_dimensoes = aleatorios_uniforme.ndim # Numero de dimensoes do array.
print(f"Número de dimensões de aleatorios_uniforme = {numero_dimensoes}")

tamanho_bytes = aleatorios_uniforme.nbytes # Memoria alocada por todos elementos do array.
print(f"Tamanho da memória alocada por todos elementos de aleatorios_uniforme = {tamanho_bytes}")
tudo_euler_dobrado = 2 * tudo_euler # Multiplica por 2 todos elementos do array tudo_euler.
print(f"tudo_euler_dobrado = \n{tudo_euler_dobrado}")
normal = np.random.normal(0, 1,5) # Cria array 1x10 com valores normalmente distribuidos com media 0 e desvio-padrao 1.
print(f"normal = {normal}")

eh_negativo, eh_positivo = normal < 0, normal > 0 # Retorna uma tupla que eh "descompactada" nas duas variaveis.
print(f"eh_negativo = {eh_negativo}")
print(f"eh_positivo = {eh_positivo}")

# Masking

negativos = normal[eh_negativo]
positivos = normal[eh_positivo]
print(f"negativos = {negativos}")
print(f"positivos = {positivos}")
quantos_negativos = np.sum(eh_negativo)
quantos_positivos = np.sum(eh_positivo)

print(f"Quantidade de valores negativos em normal = {quantos_negativos}")
print(f"Quantidade de valores positivos em normal = {quantos_positivos}")

print("\nDe fato:")
print(f"Tamanho de negativos = {negativos.size}")
print(f"Tamanho de positivos = {positivos.size}")
algum_maior_que_05 = np.any(normal > 0.5)
print(f"Algum valor em normal maior que 0.5? {algum_maior_que_05}") # Verdadeiro. Existem dois valores maiores que 0.5.

todos_entre_0_1 = np.all((normal > 0) & (normal < 1))
print(f"Todos valores de normal entre 0 e 1? {todos_entre_0_1}") # Falso. Os valores vao de -1 a 1.
normal[eh_negativo] = 0
print(f"Valores negativos foram substituidos por 0: {normal}")

primeiro_ultimo = [0, 4]
normal[primeiro_ultimo] = 1.0
print(f"Substitui os valores do primeiro e último elementos por 1: {normal}")
print(f"aleatorios_normal = \n{aleatorios_normal}")

soma_todos_elementos = aleatorios_normal.sum()
print(f"Soma de todos elementos de aleatorios_normal = {soma_todos_elementos}")

soma_das_linhas = aleatorios_normal.sum(axis=1)
print(f"Soma de cada linha de aleatorios_normal: {soma_das_linhas}")

soma_das_colunas = aleatorios_normal.sum(axis=0)
print(f"Soma de cada coluna de aleatorios_normal: {soma_das_colunas}")

media_das_linhas = aleatorios_normal.mean(axis=1)
print(f"Média de cada linha de aleatorios_normal: {media_das_linhas}")

media_das_colunas = aleatorios_normal.mean(axis=0)
print(f"Média de cada coluna de aleatorios_normal: {media_das_colunas}")