import numpy as np 
import scipy.special as sp
import matplotlib.pyplot as plt
%matplotlib inline

# Read the data
data_file = open('../input/emnist-digits-train.csv','r')
data = data_file.readlines()
data_file.close()
# all_values = data[0].split(',')
# image_array = np.asfarray(all_values[1:]).reshape((28,28))
# plt.imshow(image_array,cmap='Greys',interpolation='None')
class MNISTNeuralNetwork:
    
   def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        
        
        self.learning_rate = learning_rate
                
        self.weight1 = np.random.normal(0.0,pow(input_nodes,-0.5),(hidden_nodes,input_nodes)) #100,784
        self.weight2 = np.random.normal(0.0,pow(hidden_nodes,-0.5),(output_nodes,hidden_nodes)) #10,100
        
        
        self.activation_fun = lambda x:sp.expit(x)
        self.derivation_of_activation = lambda x:x*(1-x)
        
   def feedforward(self,input_list,actual_output_list):
        
        self.input_list = np.array(input_list,ndmin=2).T # 784,1
        self.actual_output_list = np.array(actual_output_list,ndmin=2).T #10,1
        
        self.hidden_layer_output = self.activation_fun(np.dot(self.weight1,self.input_list)) #100,1
        self.output = self.activation_fun(np.dot(self.weight2,self.hidden_layer_output)) #10,1
        
         
    
   def backprop(self):
        
        """error = actual-predicted"""
        error = self.actual_output_list - self.output #10,1
        
        
        self.weight2 += self.learning_rate * np.dot((2*error*self.derivation_of_activation(self.output)),self.hidden_layer_output.T) #10,100
        self.weight1 += self.learning_rate * np.dot(np.dot(self.weight2.T,2*error*self.derivation_of_activation(self.output))*self.derivation_of_activation(self.hidden_layer_output),self.input_list.T)
        
   def test_query(self,input_list):
        
        input_list = np.array(input_list,ndmin=2).T
        
        
        hidden_layer_output = self.activation_fun(np.dot(self.weight1,input_list)) #100,1
        output = self.activation_fun(np.dot(self.weight2,hidden_layer_output)) #10,1
        
        return output

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
nn = MNISTNeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
epochs = 5

for i in range(epochs):
    
    for record in data:
    
        all_values = record.split(',')
        input_image_array = (np.asfarray(all_values[1:])/255 *0.99) +0.01

        target_output_values = np.zeros(output_nodes) +0.01
        target_output_values[int(all_values[0])] = 0.99

        nn.feedforward(input_image_array,target_output_values)
        nn.backprop()

print("Training Complete")    
testing_file = open('../input/emnist-digits-test.csv','r')
test_data = testing_file.readlines()
testing_file.close()
correct_labels_array = []
output_label = []
score_board=[]

for record in test_data:
    
    all_values = record.split(',')
    
    correct_label = int(all_values[0])
    correct_labels_array.append(correct_label)
    
    image_array = (np.asfarray(all_values[1:])/255 *0.99)+0.01
    target_output_values = np.zeros(output_nodes)+0.01
    target_output_values[int(all_values[0])] = 0.99
    
    output = nn.test_query(image_array)
    label = np.argmax(output)
    
    output_label.append(label)
    
    if correct_label == label:
        score_board.append(1)
    else :
        score_board.append(0)
        pass

# print(score_board)
# print(correct_labels_array)
# print(output_label)        
    
score = np.mean(np.array(score_board,ndmin=2),axis=1)
print(score)    
