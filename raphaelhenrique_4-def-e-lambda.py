def festival(dias):
    custo_organização = 200 * dias
    return custo_organização
def receita(dias):
    preço_ingresso = 10
    clientes = 100
    if dias == 3:
      preço_ingresso -= 10 * 0.5
    elif dias == 2:
      preço_ingresso -= 10 * 0.25      
    elif dias == 1:
      preço_ingresso == 10    
    return (clientes * preço_ingresso) * dias 
def artista(dias, gênero):
    if gênero == "Rock":
        gênero = 150
    elif gênero == "Sertanejo":
        gênero = 180
    elif gênero == "Pagode":
        gênero = 140
    elif gênero == "Eletrônica":
        gênero = 190    
    return gênero * dias
def balanço(dias, gênero):
    return receita(dias) - artista(dias, gênero) - festival(dias)
print("Receita (R$ 1.500) - Artista (R$ 540) - Festival (R$ 600) = R$", int(balanço(3,"Sertanejo")))
Conta = lambda x,y,z: x + y + z

int(Conta(1, 2, 3))
Lambda = ["Gregos", "Troianos", "Persas", "Minóicos"]

list(filter(lambda x:x == "Gregos", Lambda))
Conta1 = [1, 2, 3]
Conta2 = [4, 5, 6]
Conta3 = [0, -1, -2]

Resultado = list(map(lambda x,y,z: x+y+z, Conta1,Conta2,Conta3))

print("Os cálculos foram efetuados na vertical com resultado =", Resultado)
def gorjeta(y):
    return lambda x:x * y

serviço = gorjeta(0.1)

print("Se neste serviço tivessemos o total de R$ 50, portanto a gorjeta seria 10% = R$", int(serviço(50)))