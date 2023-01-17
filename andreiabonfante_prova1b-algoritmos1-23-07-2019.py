entrada = int(input())

print(" Entrada: ", entrada)



ano = entrada // 360

resto = entrada % 360

mes = resto // 30

resto = resto % 30

dias = resto

print(" Saida: ", ano," anos ",mes," meses ",dias," dias")



print("Digite o código da mercadoria ")

codigo = int(input())

total_compra = 0

total_venda = 0

total_lucro = 0

menor_dez = 0

dez_vinte = 0

maior_vinte = 0



while (codigo != -1):

    print("Digite o preço de compra da mercadoria ")

    preco_compra = float(input())

    total_compra = total_compra + preco_compra

    print("Digite o preço de venda da mercadoria ")

    preco_venda = float(input())

    total_venda = total_venda + preco_venda

    lucro = preco_venda - preco_compra

    print(" Lucro: ",lucro)

    total_lucro = total_lucro + lucro

    percent_lucro = int((lucro*100)/(preco_compra))

    print(" Percentual de lucro ", percent_lucro,"%")

    if percent_lucro < 10:

        menor_dez = menor_dez + 1

    else:

        if percent_lucro > 10 and percent_lucro < 20:

            dez_vinte = dez_vinte + 1

        else:

            maior_vinte = maior_vinte + 1

    print("Digite o código da mercadoria ")

    codigo = int(input())

print(" Total de compras: ",total_compra)

print(" Total de vendas: ",total_venda)

print(" Lucro total: ",total_lucro)
print("Digite um valor para x")

x = int(input())

print("Digite um valor para n")

n = int(input())



y = 0.0

fat = 1



contador = 1

while (contador <= n):

    fat = fat * contador

    print(" Proximo numero da serie = (",x,"+",contador,")/",contador,"!")

    print(" fatorial de ", contador," = ", fat)

    y = y + ((x + contador) / fat)

    contador = contador + 1

    

print("Valor de y: ", y)

salario_fixo = 0.0

comissao = 0.0



conta_funcionario = 1



while (conta_funcionario <= 20): # passa pelos 20 funcionarios ## TROCAR O 20 POR UM NUMERO MENOR PARA FAZER TESTES

    print("Digite numero inscrição do funcionario")

    inscricao = input()

    print("Digite valor salario_fixo")

    salario_fixo = float(input())

    print("Quantas tvs cores vendeu?")

    tvsc = int(input())

    if (tvsc>=10):

        comissao = comissao + (tvsc * 100)

    else:

        comissao = comissao + (tvsc * 50)

    print(" Quantas tvs preto e branco vendeu?")

    tvspb = int(input())

    if (tvspb >= 20):

        comissao = comissao + (tvspb * 40)

    else:

        comissao = comissao + (tvspb * 20)

    

    salario_fixo = salario_fixo - (salario_fixo * 0.08) # desconta INSS

    print("Salario fixo com desconto INSS = ",salario_fixo)

    salario_bruto =  salario_fixo + comissao

    print("Salario + comissoes = ", salario_bruto)

    if (salario_bruto >= 3000):

        print("Neste salario há desconto de 5% de IR")

        salario_liquido = salario_bruto - (salario_bruto * 0.05)

    else:

        print("Neste salario não há desconto de IR")

        salario_liquido = salario_bruto

    

    print("Numero de inscrição: ",inscricao," Salario Bruto = ",salario_bruto, " Salario Liquido = ", salario_liquido)    

    conta_funcionario = conta_funcionario + 1 # atualiza para o proximo funcionario


