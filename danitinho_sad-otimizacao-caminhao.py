#otimização de um poblema simples de capacidade de carga de um caminhão de entregas



from __future__ import print_function

from ortools.algorithms import pywrapknapsack_solver







def main():

    # Create the solver.

    solver = pywrapknapsack_solver.KnapsackSolver(

        pywrapknapsack_solver.KnapsackSolver.

        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')



    #valor das entregas

    values = [

        1362, 4483, 829, 2130, 4431, 7667, 8230, 852, 4932, 12325, 6670, 8392, 6500, 3748, 428, 1447,

        7368, 2256, 6324, 12467, 1220, 1624, 5432, 355, 925, 1210, 2722, 452, 5250, 4323, 5184, 54728,

        84587, 7348, 48258, 14845, 54826, 7438, 21360, 3645, 8455, 145789, 2374, 4373, 334, 1360, 1469, 34689, 2276,

        312

    ]

    #pelos das cargas em cubagem, valor calculado para cargas nesse tipo de transporte

    #calculo da cubagem: ALTURA x LARGURA x COMPRIMENTO x 300 -> FATOR DE CUBAGEM

    weights = [[

        752, 20, 530, 22, 80, 294, 11, 81, 270, 64, 59, 168, 9, 236, 13, 68, 15, 42, 129, 40,

        472, 47, 652, 322, 26, 48, 556, 6, 293, 84, 12, 14, 18, 56, 7, 297, 93, 44, 721,

        13, 86, 66, 31, 65, 19, 79, 220, 65, 52, 13,

    ]]

    

    #capacidade média de um veículo de transporte em meio urbano, 3 tolenadas

    capacities = [3000]



    #priontar a saída

    solver.Init(values, weights, capacities)

    computed_value = solver.Solve()



    packed_items = []

    packed_weights = []

    total_weight = 0

    print('Valor total da carga =', computed_value)

    for i in range(len(values)):

        if solver.BestSolutionContains(i):

            packed_items.append(i)

            packed_weights.append(weights[0][i])

            total_weight += weights[0][i]

    print('Cubagem total das cargas:', total_weight)

    print('Cargas carregadas:', packed_items)

    print('Peso das cargas:', packed_weights)





if __name__ == '__main__':

    main()