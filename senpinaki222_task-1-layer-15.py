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
# This function returns a random valued numpy array phi, with respect to which cost of the quantum circuit will be calculated
# dimension of phi will be same as the dimension of np array returned by make_circuit() 
def initialize_random_phi():
    phi = np.random.randn(16,)
    return phi
def simulate_circuit(theta):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(theta):
    #This make_circuit function genrates the quantum circuit. It takes only the parameters theta as the argument
    #The no. of layers has not to be passed explicitely, as it is calculated from dimension of theta


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


        return qml.expval(qml.PauliZ(wires=0)),  qml.expval(qml.PauliZ(wires=1)), qml.expval(qml.PauliZ(wires=2)), qml.expval(qml.PauliZ(wires=3))
       
    circuit(theta)
    
    #returns the 2^4 = 16 dimensioned output statevector of the simulated circuit
    return dev.state
# This function takes the theta, calls the make_circuit() func and calculates the cost of vector returned 
# by the make_circuit() with respect to global variable psi
def calc_distance(theta):
    
    psi_of_theta = simulate_circuit(theta) 
    diff = psi_of_theta - phi #phi is the global one here, used for all the layers
    mod_sqr = diff.real**2 + diff.imag**2
    distance = np.sum(mod_sqr) #calculates the sum of squares of corresponding vector elements of psi and phi
    
    # returns the cost (mentioned as 'distance' in problem statement)
    return distance
# this takes param of the circuit, iteratioin and eta as necessary arguments
# also two optional args to print and plot the cost for each layer count
def optimize(theta, iterations, print_distance=False, plot_distance=False):
    
    
    distance_hist = [] #keeps track of cost after each iterations
    
    optimizer = qml.optimize.RotosolveOptimizer() 
    
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
        
        
    if(plot_distance):
        plt.style.use("seaborn")
        plt.plot(distance_hist, "b", label="Optimization")
        plt.ylabel("Distance value")
        plt.xlabel("Optimization steps")
        plt.legend()
        plt.show()
    
    #returns the final_cost, maximum iteration used and optimized parameter values
    cache = {"min_distance": distance_hist[-1],
             "iter": i+1,
             "optimized_theta": theta}
    
    return cache
# main function

phi = initialize_random_phi() #initialized a vector phi as a referenc of cost for the whole process
iter_list = [] # it will store no. of iterations used for each no. of layers
min_distance_list = [] # it will store the final cost of circuit for each no. of layers
optimized_theta_list = [] #similarly stores the optimized params for each no. of layers

max_layers = 15 # so we will start from a single layer circuit and go upto 30 layers circuit


for layers in range(1,max_layers+1):
    print("\n\n Layer count in Circuit => "+str(layers))
    theta = initialize_theta(layers) #initializes random theta values which will be optimized further
    
    # calls the optimize() function which will first create a circuit with the passed theta values 
    # then optimize the parameters using RotosolveOptimizer() algorithm and finally returns min_distance, iteration used and optimized params
    # although the iteration is initialized as 50, if it fulfils the callback condition, it will be terminated
    cache = optimize(theta, 50) 
    
    iter_list.append(cache['iter'])
    min_distance_list.append(cache['min_distance'])
    
    print("\n Iteration used => "+str(cache['iter']))
    print(" Minimum distance => "+str(cache['min_distance']))
    print("\n_______________________________________")
    print("_______________________________________")
    


# plots final cost as a function of no. of layers used in the circuit
plt.style.use("seaborn")
plt.plot(np.arange(1,max_layers+1),min_distance_list, "b")
plt.ylabel("Minimum distance")
plt.xlabel("No. of layers")
#plt.ylim(0,0.1)
plt.title("Final cost as a function of no. of layers")
plt.show()

# plots he maximum iterations used as a function of no. of layers in the specific circuit
plt.plot(np.arange(1,max_layers+1),iter_list, "g")
plt.ylabel("Iteration used")
plt.xlabel("No. of layers")
plt.title("Total no. of iterations used as a function of no. of layers")
plt.show()
