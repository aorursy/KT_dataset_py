inteiro = 10 

print(f"{inteiro}: {type(inteiro)}")



floating = 0.15

print(f"{floating}: {type(floating)}")



string = "oie!"

print(f"{string}: {type(string)}")



booleana = True

print(f"{booleana}: {type(booleana)}")
# Definindo a classe Ponto2D

class Ponto2D:



    # Definindo o inicializador dela

    def __init__(self, x, y):

        self.x = x

        self.y = y

        

    # Definindo uma função de produto escalar

    def produto_escalar(self, other):

        return self.x * other.x + self.y + other.y

        

        

ponto_um = Ponto2D(1, 2)

print(f"x e y do ponto_um: ({ponto_um.x}, {ponto_um.y})")



ponto_dois = Ponto2D(5, 10)

print(f"x e y do ponto_dois: ({ponto_dois.x}, {ponto_dois.y})")



escalar = ponto_um.produto_escalar(ponto_dois)

print(f"Produto interno dos pontos: {escalar}")
class Cliente:

    

    def __init__(self, nome, cpf):

        self.nome = nome

        self.cpf = cpf
cliente_exemplo = Cliente("Moacir Andrade", 13224112320) # CPF como número e não string pra facilitar no próximo exemplo :)

print(type(cliente_exemplo))
class Cliente:

    

    def __init__(self, nome, cpf):

        self.nome = nome

        self.cpf = cpf

        

    def is_cpf_valido(self):

        if self.cpf % 2 == 1:

            return True

        return False
cliente = Cliente("Enzo", 333)

cpf_valido = cliente.is_cpf_valido()



print(f"O CPF do cliente {cliente.nome} é valido? {cpf_valido}")



cliente = Cliente("Mietro", 20)

cpf_valido = cliente.is_cpf_valido()



print(f"O CPF do cliente {cliente.nome} é valido? {cpf_valido}")