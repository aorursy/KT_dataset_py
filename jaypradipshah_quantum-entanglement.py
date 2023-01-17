import numpy as np

zero_state = np.array([[1],[0]])

one_state = np.array([[0],[1]])
print("The |0> is \n", zero_state)

print("The |1> is \n", one_state)
print("The shape of the Vector |0> is",zero_state.shape)

print("The shape of the Vector |1> is",one_state.shape)
!pip install qiskit #Install qiskit on colab

!pip install pylatexenc #Required for matplotlib support to render the images of circuit

from IPython.display import clear_output #Clear the output after installation

clear_output()
#QuantumCircuit is the class that we shall use for our circuit

#Aer is required to get the required backend support

#execute will execute our circuit

from qiskit import QuantumCircuit, Aer, execute
circuit_es1 = QuantumCircuit(2,2) #initializing our circuit with two qubits and two bits
circuit_es1.h(0) #Applying hadamard Gate on the first qubit

circuit_es1.cx(0,1) #Applying the CNOT Gate with zeroth qubit as the control qubit 

                    #and the first qubit as the target qubit

circuit_es1.measure(0,0) #Measuring the zeroth qubit and storing its output to zeroth bit

circuit_es1.measure(1,1)
#Let's visualize our circuit

circuit_es1.draw('mpl')
simulator = Aer.get_backend('qasm_simulator') #Using qasm_simulator, we can simulate our circuit on an actual quantum device

result = execute(circuit_es1,backend=simulator, shots = 1024).result() #shots = number of time we want to execute the circuit
counts = result.get_counts() #Get the results for each simulation
from qiskit.visualization import plot_histogram #To visualize our results

plot_histogram(counts)
circuit_es2 = QuantumCircuit(2,2)

circuit_es2.x(0) #Applying the NOT Gate on the first qubit to get the state |01>

circuit_es2.h(0) 

circuit_es2.cx(0,1)

circuit_es2.measure(0,0) 

circuit_es2.measure(1,1)
#Let's visualize our circuit

circuit_es2.draw('mpl')
simulator = Aer.get_backend('qasm_simulator') 

result = execute(circuit_es2,backend=simulator, shots = 1024).result() 

counts = result.get_counts()

plot_histogram(counts)
circuit_es3 = QuantumCircuit(2,2)

circuit_es3.x(1) #Applying the NOT Gate on the second qubit to get the state |10>

circuit_es3.h(0) 

circuit_es3.cx(0,1)

circuit_es3.measure(0,0) 

circuit_es3.measure(1,1)
#Let's visualize our circuit

circuit_es3.draw('mpl')
simulator = Aer.get_backend('qasm_simulator') 

result = execute(circuit_es3,backend=simulator, shots = 1024).result() 

counts = result.get_counts()

plot_histogram(counts)
circuit_es4 = QuantumCircuit(2,2)

circuit_es4.x(0)

circuit_es4.x(1) #Applying the NOT Gates on the two qubits to get the state |11>

circuit_es4.h(0) 

circuit_es4.cx(0,1)

circuit_es4.measure(0,0) 

circuit_es4.measure(1,1)
#Let's visualize our circuit

circuit_es4.draw('mpl')
simulator = Aer.get_backend('qasm_simulator') 

result = execute(circuit_es4,backend=simulator, shots = 1024).result() 

counts = result.get_counts()

plot_histogram(counts)