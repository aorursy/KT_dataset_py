import numpy as np
import pennylane as qml

from pennylane import expval, var
from matplotlib import pyplot as plt
pi = np.pi
def initialize_theta(layers):
    row = layers*2
    column = 4
    theta = np.random.uniform(low=0, high=2*pi, size=(row, column))
    
    
    return theta
def initialize_random_phi():
    phi = np.random.randn(16,)
    return phi

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def simulate_circuit(theta):


    layers = theta.shape[0] // 2

    for i in range(layers):

        # yellow block
        for j in range(4): #This loops over all four qubits and apply RX gate with proper param extracted from theta
            qml.RX(theta[2*i][j], wires=j)

        # Green block
        for j in range(4): #THis loops over all 4 qubits and apply RZ gate with proper param extracted from theta
            qml.RZ(theta[(2*i)+1][j], wires=j)

        for control in range(3): # Double qubit CZ gates has been applied here
            for target in range(control+1, 4):
                qml.CZ(wires=[control, target])


    return dev.state



def calc_distance(theta):
    
    psi_of_theta = simulate_circuit(theta) 
    diff = psi_of_theta - phi #phi is the global one here, used for all the layers
    mod_sqr = diff.real**2 + diff.imag**2
    distance = np.sum(mod_sqr) #calculates the sum of squares of corresponding vector elements of psi and phi
    
    # returns the cost (mentioned as 'distance' in problem statement)
    return distance
def optimize(theta, iterations, print_distance=False, plot_distance=False):
    
    
    distance_hist = [] #keeps track of cost after each iterations
    
    optimizer = qml.optimize.GradientDescentOptimizer() 
    
    # callback: this flag is initialized to keep track of repeating costs, which is further used to terminate the iteration
    # when optimization is completed approx
    flag = 0
    
    print("Optimizing", end='')
    
    for i in range(iterations): 
        theta = optimizer.step(calc_distance, theta)
        distance = calc_distance(theta) #calls the ampltude() to calculate cost of the circuit
        print('.', end='')
        
        # this code chunk is used to compare current cost with the previous one to check repetations
        if(len(distance_hist)>0):
            if(round(distance_hist[-1],2)==round(distance,2)): #here upto 2 decimal places round off has been done
                flag += 1
            else:
                flag = 0
        distance_hist.append(distance)
        if(print_distance):
            print("Iteration=> "+str(i)+", distance=> "+str(distance))
        
        # Whenever flag becomes 3, means 3 times same cost has been calculated, we can assume the params to be
        # optimized and can terminate the gradient descent
        if(flag == 3):
            break
    
    print('\nOptimization Completed')
        
        
    
    
    cache = {"min_distance": distance_hist[-1],
             "optimized_theta": theta}
    
    return cache
