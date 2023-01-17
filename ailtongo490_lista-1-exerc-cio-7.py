def faltas(lista):

    f = {}

    f["total"] = 0

    maisFaltas = -1

    menosFaltas = -1

    

    for jogo in lista:

            

        for indice in range(2):

            cont = 0

            for time in lista:

                for i in range(2):

                    if(jogo[indice] == time[i]):

                        cont += time[2][i]

            if(cont > maisFaltas):

                f["maisFaltas"] = jogo[indice]

                maisFaltas = cont

            if((cont < menosFaltas and cont != 0) or menosFaltas < 0):

                f["menosFaltas"] = jogo[indice]

                menosFaltas = cont

        

        f["total"] += jogo[2][0] + jogo[2][1]

    

    return f
copa = [['Brasil', 'Italia', [5, 90]],['Brasil', 'Espanha', [5, 2]], ['Italia', 'Espanha', [7,7]]]



faltas(copa)
1% 2