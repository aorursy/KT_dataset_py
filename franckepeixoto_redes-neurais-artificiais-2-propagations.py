from numpy import dot, exp, array, append



def ReLU(x):

    return x * (x > 0)



def Sigmoid(x):

    return 1 / (1 + exp(-x))





i =  array([[0.1, 0.4, 0.5]])



w_ij = array([[0.1, 0.2, 0.4, 0.3],

              [0.5, 0.4, 0.7, 0.9],

              [0.2, 0.6, 0.3, 0.8]])



w_jk = array([[0.2],

              [0.3],

              [0.6],

              [0.1]])



z_ij = dot(i, w_ij) #Calcular as entradas e pesos da camada oculta

a_ij = ReLU(z_ij) # Aplicando a ReLU



z_jk = dot(a_ij, w_jk) #Após a função de ativação da camada de saída

a_jk = Sigmoid(z_jk) #Aplicando a Sigmoidn 



print ("Saída da RNA Feed Forward: ",a_jk)

from numpy import log10

def crossentropyerror(a, y):

    return - sum(y * log10(a) + (1 - y) * log10(1 - a))



print("Saida de nossa RNA:  ",a_jk)

y = array([[1]])

print("Nosso Ground Truth:  ",y)

print("Cross Entropy Error: ",crossentropyerror(a_jk, y))
import matplotlib.pyplot as plt

def SigmoidD(x):

    return Sigmoid(x) * (1 - Sigmoid(x))

x = array(range(-10, 10))

plt.plot(x, SigmoidD(x))

plt.title("Derivada - sigmóide")

plt.xlabel("x")

plt.ylabel("sigmóide")
def ReLUD(x):

    return 1. * (x > 0)

x = array(range(-10, 10))

plt.plot(x, ReLUD(x))

plt.title("Derivada - ReLU")

plt.xlabel("x")

plt.ylabel("ReLU")
print("Backpropagation da camada de saída para a camada oculta.\n")



dl_jk = -y/a_jk + (1 - y)/(1 - a_jk)

print("Derivada da Cross Entropy Loss em relação a nossa saida: {0}\n".format(dl_jk))



da_jk = SigmoidD(z_jk) 

print("Derivada da Signóide de entrada (antes da função de ativação) da camada de saída: {0}\n".format(da_jk))



dz_jk = a_ij 

print("Derivada das entradas da camada oculta (antes da função de ativação) \nem relação aos pesos da camada de saída: {0}\n".format(dz_jk))



gradient_jk = dot(dz_jk.T , dl_jk * da_jk) 

print("Chain Rule: {0}\n".format(gradient_jk))





print("Backpropagation da camada oculta para a camada de saída.\n")



dl_ij = dot(da_jk * dl_jk, w_jk.T) 

print("Derivada da Cross Entropy Loss em relação a entrada da camada oculta (após a função de ativação):\n{0}\n".format(dl_ij))



da_ij = ReLUD(z_ij)

print("Derivada da ReLU de entrada (antes da função de ativação) da camada oculta:\n{0}\n".format(da_ij))



dz_ij = i

print("Derivada das entradas da camada oculta (antes da função de ativação)\ncom os pesos da camada oculta: {0}\n".format(dz_ij))



gradient_ij = dot(dz_ij.T , dl_ij * da_ij)

print("Chain Rule:\n{0}".format(gradient_ij))



print("Novos pesos obtidos após o processo do Backpropagation.\n")

w_ij = w_ij - gradient_ij 

w_jk = w_jk - gradient_jk

print("w_ij:\n{0}\n".format(w_ij))

print("w_jk:\n{0}\n".format(w_jk))
print("Agora é só prever a saída com os novos pesos.\n")





i = array([[0.1, 0.4, 0.5]])

print("i: {0}\n".format(i))





w_ij = array([[ 0.10723859,  0.21085788,  0.42171576,  0.30361929],

              [ 0.52895435,  0.44343152,  0.78686304,  0.91447717],

              [ 0.23619293,  0.6542894,   0.4085788,   0.81809647]])

print("Novos pesos da camada oculta:\n{0}\n".format(w_ij))



 

w_jk = array([[ 0.3121981 ],

              [ 0.47372609],

              [ 0.77010679],

              [ 0.38592418]])

print("Novos pessoas da camada de saída:\n{0}\n".format(w_jk))





z_ij = dot(i, w_ij) 

print("Calcular o produto escalar das entradas e dos pesos da camada oculta:\n{0}\n".format(z_ij))



a_ij = ReLU(z_ij)

print("ReLU: {0}\n".format(a_ij))



z_jk = dot(a_ij, w_jk) 

print("Produto escalar após ativação: {0}\n".format(z_jk))



a_jk = Sigmoid(z_jk) 

print("Sigmoid: {0}\n".format(a_jk))