from IPython.display import Image
Image("../input/neuralnetwork/Screenshot from 2020-06-21 16-18-13.png", width = '600px')
from IPython.display import Image
Image("/kaggle/input/neuralnetwork/3.png", width = '800px')
import numpy as np 
import matplotlib.pylab as plt
def relu_(x):
    return np.maximum(0, x)

fig = plt.figure(figsize=(7,7))
ax1 = plt.subplot2grid((1,1), (0,0))
plt.style.use('ggplot') 

x = np.arange(-10.0,10.0, 0.1)
y = relu_(x)

plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3, 
          markerfacecolor='blue', label='ReLU Curve')
plt.legend()
plt.show()

class relu: # Reference Link: https://nobulingual.com/?p=913
    
    def __init__(self, value):
        self.value = value
    
    def ReLU(self):
        if float(self.value) < 0 or float(self.value) == 0:
            self.value = 0
        else:
            self.value = self.value
        return self.value

Input_Layer = np.random.randn()
Output_Value = relu(Input_Layer)
print("The Random number is : " + str(Input_Layer))
print("The ReLU value is : {}".format(Output_Value.ReLU()))
from IPython.display import Image
Image("/kaggle/input/neuralnetwork/5.png", width = '400px')
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

fig = plt.figure(figsize=(7,7))
ax1 = plt.subplot2grid((1,1), (0,0))
plt.style.use('ggplot')

x = np.arange(-10.0,10.0, 0.1)
y = sigmoid(x)

plt.plot(x, y, color='orange', linestyle='dashed',
         linewidth = 3, markerfacecolor='blue', label='ReLU Curve')
plt.legend()
plt.show()
class sigmoid:
    
    def __init__(self, value):
        self.value = value

    def Sigmoid(self):
        sIgmoid = 1/(1+np.exp(self.value))
        return sIgmoid
    
Input_Layer = np.random.randn()
Output_Value = sigmoid(Input_Layer)
print("The Random number is : " + str(Input_Layer))
print("The Sigmoid value is : {}".format(Output_Value.Sigmoid()))
from IPython.display import Image
Image("../input/neuralnetwork/Screenshot from 2020-06-21 21-25-55.png", width = '600px')
from IPython.display import Image
Image("../input/neuralnetwork/6.png", width = '600px')
Input_1 = np.random.randn();  Weight_1 = np.random.randn()
Input_2 = np.random.randn();  Weight_2 = np.random.randn()
Input_3 = np.random.randn();  Weight_3 = np.random.randn()

Input_Value = [Input_1, Input_2, Input_3]
Weight_value = [Weight_1, Weight_2, Weight_3]
Bias_Value = np.random.randn()

print("Input Values: "+ str(Input_Value))
print("--------------------------------------------------------\n")
print("Weight value: "+ str(Weight_value))
print("--------------------------------------------------------\n")
print("Bias Values: "+ str(Bias_Value))
print("--------------------------------------------------------\n")
# Multiplying and Adding Input, Weight and Bias Value 

result = Input_Value[0]*Weight_value[0] + Input_Value[1]*Weight_value[1] + Input_Value[2]*Weight_value[2] + Bias_Value 
    
print("The Multiplying and Adding Result is : "+ str(result))
print("--------------------------------------------------------\n\n")
print("------------- Applying Sigmoid Function ----------------\n")


sigmoid_output = sigmoid(result)
print("The Sigmoid value is : "+ str(sigmoid_output))
print("--------------------------------------------------------\n\n")


print("-------------- Applying ReLU Function ------------------\n")
relu_output = relu_(sigmoid_output)
print("The ReLU value is : "+str(relu_output))
print("--------------------------------------------------------")
from IPython.display import Image
Image("/kaggle/input/neuralnetwork/2.png", width = '600px')
Input_1 = np.random.randn();  Weight_1 = np.random.randn()
Input_2 = np.random.randn();  Weight_2 = np.random.randn()
Input_3 = np.random.randn();  Weight_3 = np.random.randn()

Weight_4 = np.random.randn();  Weight_7 = np.random.randn()
Weight_5 = np.random.randn();  Weight_8 = np.random.randn()
Weight_6 = np.random.randn();  Weight_9 = np.random.randn()

Bias_Value1 = np.random.randn()
Bias_Value2 = np.random.randn()
Bias_Value3 = np.random.randn()

# we are creating the random input value 
Input_Value   = [Input_1, Input_2, Input_3]          

# we are creating the random weight value for the first neuron
Weight_value = [[Weight_1, Weight_2, Weight_3],  
# we are creating the random weight value for the second neuron
               [Weight_4, Weight_5, Weight_6],   
# we are creating the random weight value for the third neuron
               [Weight_7, Weight_8, Weight_9]]   

# we are creating the random Bias value for the each neuron 
Bias_Value    = [Bias_Value1, Bias_Value2, Bias_Value3]


print("Input Values: "+ str(Input_Value))
print("--------------------------------------------------------\n")
print("Weight value 1: "+ str(Weight_value1) )
print("Weight value 2: "+ str(Weight_value2) )
print("Weight value 3: "+ str(Weight_value3) )
print("--------------------------------------------------------\n")
print("Bias Values: "+ str(Bias_Value))
print("--------------------------------------------------------\n\n")
# Multiplying and Adding Input, Weight and Bias Value 
print("-- Multiplying and Adding Input, Weight and Bias Value --\n")

layer_output = []
for neuron_weights, neuron_bias in zip(Weight_value, Bias_Value):
    neuron_output = 0
    for n_input, weight in zip(Input_Value,neuron_weights):
        neuron_output = neuron_output + n_input*weight
    neuron_output = neuron_output +neuron_bias
    layer_output.append(neuron_output)
#                 (or)

# result = [Input_Value[0]*Weight_value[0][0] + Input_Value[1]*Weight_value[0][1] + Input_Value[2]*Weight_value[0][2] + Bias_Value[0], 
#           Input_Value[0]*Weight_value[1][0] + Input_Value[1]*Weight_value[1][1] + Input_Value[2]*Weight_value[1][2] + Bias_Value[1], 
#           Input_Value[0]*Weight_value[2][0] + Input_Value[1]*Weight_value[2][1] + Input_Value[2]*Weight_value[2][2] + Bias_Value[2]] 

print("Multiplying and Adding result : "+ str(layer_output))
print("--------------------------------------------------------\n\n")


print("------------- Applying Sigmoid Function ----------------\n")
sigmoid_layer = []
for i in layer_output:
    sigmoid_output = sigmoid(i)
    sigmoid_layer.append(sigmoid_output)
print("Sigmoid value : "+ str(sigmoid_layer))
print("--------------------------------------------------------\n\n")



print("-------------- Applying ReLU Function ------------------\n")
relu_layer = []
for i in sigmoid_layer:
    relu_output = relu_(i)
    relu_layer.append(relu_output)
print("ReLU value : "+str(relu_layer))
print("--------------------------------------------------------")