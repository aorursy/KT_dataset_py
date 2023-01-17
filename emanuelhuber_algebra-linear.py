import numpy as np # Numpy é a biblioteca mais utilizada para álgebra linear
import matplotlib.pyplot as plt # Matplotlib possui diversas funções que facilitam a exibição de gráficos e figuras
x = 0
y = 0
# O código abaixo verifica se a sua implementação está correta, se estiver não deve aparecer uma mensagem de erro
assert(x + y == 10)
assert(type(x) == np.ndarray)
x = 0

# O código abaixo verifica se a sua implementação está correta, se estiver não deve aparecer uma mensagem de erro
assert(len(x) == 5)
assert((type(x)) == np.ndarray)
assert(x.shape == (5,))
x = 0

# O código abaixo verifica se a sua implementação está correta, se estiver não deve aparecer uma mensagem de erro
assert(x.shape == (10, 20))
assert(x.sum() == 0)
assert((type(x)) == np.ndarray)
x = 0

# O código abaixo irá mostrar a matriz em formato de imagem para facilitar a visualização.
assert(x.shape == (11, 11))
assert((type(x)) == np.ndarray)
plt.imshow(x)