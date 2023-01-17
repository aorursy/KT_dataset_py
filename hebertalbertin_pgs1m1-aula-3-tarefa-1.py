import csv
def read_csv(file):
    
    public_sum_zero = 0
    goals_sum_54_90 = 0
    total_cups = 0
    public_mean = 0
    total_matches = 0
    matches_goals_mean = 0
    country_team_wins = 0
    brazil_first_four_positions = 0
    years_france_third = []
    winners = {}
    
    with open(file, 'r') as f:
        reader = csv.reader(f)
        
        # pula o cabecalho
        next(reader) 
        
        for line in reader:
            total_cups += 1
            
            year = int(line[0])
            country = line[1]
            winner = line[2]
            second = line[3]
            third = line[4]
            fourth = line[5]
            goals = int(line[6])
            matches = int(line[8])
            attendance = int(line[9].replace('.', ''))
            
            if year % 10 == 0:
                public_sum_zero += attendance
            
            if year >= 1954 and year <= 1990:
                goals_sum_54_90 += goals
            
            public_mean += attendance
            
            total_matches += matches
            matches_goals_mean += goals
            
            if country == winner:
                country_team_wins += 1
            
            if  'Brazil' == winner or 'Brazil' == second or 'Brazil' == third or 'Brazil' == fourth:
                    brazil_first_four_positions += 1
            
            if 'France' == third:
                years_france_third.append(year)
            
            if winner in winners:
                winners[winner] += 1
            else:
                winners[winner] = 1
            
        # output
        with open('WorldCupsOutput.txt', 'w', encoding='utf-8') as output:
            output.write('Soma de público das copas com anos final 0 (1930, 1950, etc): ' + str(public_sum_zero) + '\n')
            output.write('Quantidade total de gols entre as copas de 1954 e 1990, inclusive: ' + str(goals_sum_54_90) + '\n')
            output.write('Média de público: ' + str(public_mean / total_cups) + '\n')
            output.write('Média de gols por partida: ' + str(matches_goals_mean / total_matches) + '\n')
            output.write('Quantidade de vezes em que o país sede foi campeão: ' + str(country_team_wins) + '\n')
            output.write('Quantidade de vezes em que o time do Brasil ficou entre uma das 4 primeiras posições: ' + str(brazil_first_four_positions) + '\n')
            output.write('Ano das edições em que o time da França finalizou em terceiro lugar: ' + str(years_france_third) + '\n')
            output.write('Quantidade de vitórias por país, classificada em ordem crescente do número de títulos:\n')
            for win in sorted(winners.items(), key=lambda key_value: key_value[1]):
                output.write('{0}:{1}\n'.format(win[0], win[1]))
read_csv('../input/world-cups/WorldCups.csv')
