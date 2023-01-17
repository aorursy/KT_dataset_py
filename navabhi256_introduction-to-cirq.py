# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Code from the official Tensorflow channel on youtube!
!pip install cirq
import cirq
circuit = cirq.Circuit() #creating a circuit object

(q0,q1) = cirq.LineQubit.range(2) #defining two qubits(bits but possessing the powers of superposition!)



#applying a hadamard gate on the qubit q0, and applying a control NOT gate on both qubits

circuit.append([cirq.H(q0), cirq.CNOT(q0,q1)]) 



#measure the qubits

circuit.append([cirq.measure(q0), cirq.measure(q1)])



#self-explanatory

print(circuit)
#Let's run a simulation with the circuit!



circuit = cirq.Circuit() 

(q0,q1) = cirq.LineQubit.range(2)



circuit.append([cirq.H(q0), cirq.CNOT(q0,q1)]) 

circuit.append([cirq.measure(q0), cirq.measure(q1)])



sim = cirq.Simulator()

results = sim.run(circuit, repetitions=10)



#So what we did is run a simulation using this circuit, and we did it for ten repetitions

print(results)
device = cirq.google.Bristlecone

print(device)

#This is the representation of the layout of the qubits for the device 
import cirq



device = cirq.google.Bristlecone



circuit = cirq.Circuit(device=device)

a0,a1 = cirq.GridQubit(5,5), cirq.GridQubit(5,6)

b0, b1 = cirq.GridQubit(6,5), cirq.GridQubit(6,6)

circuit.append([cirq.CZ(a0,a1),cirq.CZ(b0,b1)])



print(circuit)
#We are very very very far away from having quantum computers capable of doing tasks such as...

#...factoring incredibly large numbers, which would break modern encryption, but we've made...

#...some form of progress at least! The people at Google are doing some pretty cool stuff, so...

#...make sure to check out the official Tensorflow channel to learn more!