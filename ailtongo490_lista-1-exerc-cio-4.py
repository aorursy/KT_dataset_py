def operacoes(lista):

    dados = {}

    dados["maisCaracteres"] = lista[0]

    dados["mediaDeVogais"] = 0

    vogais = ['a', 'e', 'i', 'o', 'u']

    dados["ocorrenciasPrimeiroElemento"] = 0

    dados["palavraLexicograficamenteMaior"] = sorted(lista)[len(lista) - 1]

    dados["palavrasCompostas"] = 0

    dados["vizinhosIguais"] = 0

    dados['sublistas'] = []

    valorPalavra = 0

        

    valorPalavraMaior = valorPalavra

    

    for indice,elemento in enumerate(lista):

        if(len(dados["maisCaracteres"]) < len(elemento)):

            dados["maisCaracteres"] = elemento

            

        cont = 0

        valorPalavra = 0

        for caractere in elemento:

            caractere = caractere.lower()

            if caractere in vogais:

                cont += 1

                

        for caractere in elemento:

            if(caractere == "-" or caractere == "'"):

                dados["palavrasCompostas"] += 1

                break

            

        dados["mediaDeVogais"] += cont

        

        if(lista[0] == elemento):

            dados["ocorrenciasPrimeiroElemento"] += 1

            

        if(valorPalavraMaior < valorPalavra):

            dados["palavraLexicograficamenteMaior"] = elemento

            valorPalavraMaior = valorPalavra

        

        if (indice + 1) < len(lista):

            if elemento == lista[indice + 1]:

                dados['vizinhosIguais'] += 1

                

        for i,n in enumerate(lista):

            if indice != i:

                dados['sublistas'].append([elemento,n])

            

    dados["mediaDeVogais"] = round(dados["mediaDeVogais"] / len(lista), 2)

    

    return dados
operacoes(["b","gr'adsafdsaf","k-","b","zo","A","i","p","zavfaev"])