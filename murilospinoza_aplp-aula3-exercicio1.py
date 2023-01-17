import csv
def analisaCopas():
    diretorioCsv = 'WorldCups.csv'
    
    publicoAnoFinalZero = 0
    publicoTotal = 0

    qtdeCopas = 0
    qtdePartidas = 0

    qtdeGols = 0
    qtdeGolsPeriodo = 0

    qtdeSedeCampea = 0

    qtdeBrasilFinalista = 0
    anosFrancaTerceira = []
    ganhadores = {}
    
    with open(diretorioCsv, 'r') as arquivoCsv:
        copas = csv.reader(arquivoCsv)
        
        # pula o cabecalho
        next(copas)
        
        for copa in copas:
            qtdeCopas += 1

            copaAno = int(copa[0])
            copaSede = copa[1]
            copaRanking = (copa[2], copa[3], copa[4], copa[5])
            copaGols = int(copa[6])
            copaPartidas = int(copa[8])
            copaPublico = int(copa[9].replace('.', ''))
            
            if copaAno % 10 == 0:
                publicoAnoFinalZero += copaPublico
            
            if copaAno >= 1954 and copaAno <= 1990:
                qtdeGolsPeriodo += copaGols
            
            publicoTotal += copaPublico
            
            qtdePartidas += copaPartidas
            qtdeGols += copaGols
            
            if copaSede == copaRanking[0]:
                qtdeSedeCampea += 1
            
            if  'Brazil' in copaRanking:
                qtdeBrasilFinalista += 1
            
            if 'France' == copaRanking[2]:
                anosFrancaTerceira.append(copaAno)
            
            if copaRanking[0] in ganhadores:
                ganhadores[copaRanking[0]] += 1
            else:
                ganhadores[copaRanking[0]] = 1

    with open('WorldCupsOutput.txt', 'w', encoding='utf-8') as output:
        output.write('Soma de público das copas com final 0: ' + str(publicoAnoFinalZero) + '\n')
        output.write('Quantidade total de gols entre as copas de 1954 e 1990: ' + str(qtdeGolsPeriodo) + '\n')
        output.write('Média de público: ' + str(publicoTotal / qtdeCopas) + '\n')
        output.write('Média de gols por partida: ' + str(round(qtdeGols / qtdePartidas, 2)) + '\n')
        output.write('Quantidade de vezes em que o país sede foi campeão: ' + str(qtdeSedeCampea) + '\n')
        output.write('Quantidade de vezes em que o time do Brasil ficou entre uma das 4 primeiras posições: ' + str(qtdeBrasilFinalista) + '\n')
        output.write('Ano das edições em que o time da França finalizou em terceiro lugar: ' + str(','.join(str(n) for n in anosFrancaTerceira)) + '\n')
        output.write('Quantidade de vitórias por país, classificada em ordem crescente do número de títulos:\n')

        ganhadoresOrdenados = sorted(ganhadores.items(), key=lambda ganhador: ganhador[1])

        for ganhador in ganhadoresOrdenados:
              output.write('\t' + ganhador[0] + ':' + str(ganhador[1]) + '\n')
analisaCopas()