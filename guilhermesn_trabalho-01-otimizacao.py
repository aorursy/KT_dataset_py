from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

from ortools.linear_solver import pywraplp

from ortools.sat.python import cp_model

import matplotlib

import matplotlib.pyplot as plt

import numpy as np





# valComb = 5 (valor do litro de combustível)

# valAlu = 50 (Valor do alugue do carro por hora)

# dist = 200 (distância a percorrer)

# velo = 120 (velocidade do carro)

# cons = 0.0000002 * (velo ^ 3) (função de consumo em litros pela velocidade)

# valConsDist = (dist / cons) * valComb (valor gasto para percorrer distância)

# valTotal = valConsDist + (valAlu * (dist / velo)) (valor total do percurso)

x = np.linspace(start = 0, stop = 300) 

plt.plot(x, (x**3)*0.0000002)  

plt.xlabel('Velocidade (Km/H)', fontsize=15, color='green')

plt.ylabel('Litros (L/Km)', fontsize=15, color='green')








def custoDeViagem():

    model = cp_model.CpModel()

    # [END model]

    solver = pywraplp.Solver('custoDeViagem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    

    infinity = solver.infinity()

    

    # [Variaveis]

    

    valComb = 5

    valAlu = 50

    dist = 200

    k = 0.002

    

    #valComb = solver.Var(0.0, infinity, 'valComb')

    #valAlu = solver.Var(0.0, infinity, 'valAlu')

    #dist = solver.Var(0.0, infinity, 'dist')

    

    valConsDist = solver.IntVar(0.0, infinity, 'valConsDist')

    cons = solver.IntVar(0.0, infinity, 'cons')

    velo = solver.IntVar(0.0, infinity, 'velo')

    valTotal = solver.IntVar(0.0, infinity, 'valTotal')

    

    # [Constraints]



    #model.Add(valComb = 5)

    #model.Add(valAlu = 50)

    #model.Add(dist = 200)  

    model.Add(cons = k * (velo * velo))

    model.Add(valConsDist = (dist / cons) * valComb)

    

    # [Resolve]

    

    solver = cp_model.CpSolver()

    solver.Minimize(valTotal = valConsDist + (valAlu * (dist / velo)))

    status = solver.Solve(model)



    if status == cp_model.FEASIBLE:

        print('Velocidade ideal = %i' % solver.Value(velo))

        print('Custo da viagem = %i' % solver.Value(valTotal))



custoDeViagem()