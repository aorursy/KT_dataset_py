#VARIAVEIS



nome = 'Maria' # variavel do tipo string (sequencia de caracteres)

idade = 34 # variavel do tipo integer (inteiro)

salario = 5800.0 # variavel do tipo float (real)

nacionalidade = 'brasileira' # variavel do tipo string (sequencia de caracteres)
#SAIDA DE DADOS : FORMA 1 (PLACEHOLDERS)



print("A usuaria %s tem %d anos, ganha R$ %.2f reais e eh %s." %(nome, idade, salario, nacionalidade))

# Para cada tipo de variavel ha uma letra que representa o placeholder (espaco reservado). %s para strings, %d para inteiros e %f para float.

# .2 em %.2f representa o numero de casa de decimais do numero real que deseja-se formatar.
#SAIDA DE DADOS : FORMA 2 (PLACEHOLDERS)



print("A usuaria {:s} tem {:d} anos, ganha R$ {:.2f} reais e eh {:s}.".format(nome, idade, salario, nacionalidade))

# Esta e a maneira mais usual de placeholders. O metodo .format() permite criar e customizar strings.

# Qualquer tipo de variavel pode ser inserido em string.format(var1,var2,...).
#SAIDA DE DADOS : FORMA 3 (INTERPOLACAO)



print(f"A usuaria {nome} tem {idade} anos, ganha R$ {salario:.2f} reais e eh {nacionalidade}.")

# F-strings é uma maneira simples de incorporar variaveis de qualquer tipo dentro de strings literais.
#SAIDA DE DADOS : FORMA 4 (CONCATENACAO DE STRINGS COM VARIAVEIS DE OUTROS TIPOS)



print("A usuaria ", nome, " tem ",idade, " anos, ganha R$ ",salario, " reais e eh",nacionalidade,".")
#SAIDA DE DADOS : FORMA 5 (CONCATENACAO DE STRINGS)



print("A usuaria " + nome + " tem " + str(idade) + " anos, ganha R$ "+ str(salario) + " reais e eh " + nacionalidade + ".")