# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plot graph



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#hidden_layer is node in hidden layer

#hidden_layer sample [5,5,X] X for clasification = number of class

def neural_network_create_weight_bias(feature_list, hidden_layer):

    #create empty weight list

    weight = []

    #create empty bias list

    bias = []

    #for in layer of hidden layer

    for i in range(len(hidden_layer)):

        

        #create bias_i 

        #bias appear in hidden_layer only

        #create bias_i row = 1,column = number of node of  hidden_layer[i]

        bias_i_column = hidden_layer[i]

        #create bias_i

        bias_i = np.random.randn(1, bias_i_column)

        

        #create weigth_i

        if i <= 0:

            #first weigth_i

            #create weigth_i row = column of feature

            weight_i_row = feature_list.shape[1]

            

        else:

            #other weigth_i

            #create weigth_i row = number of node of hidden_layer[i-1]

            weight_i_row = hidden_layer[i-1]

        

        #create weigth_i column = number of node of  hidden_layer[i]

        weight_i_column = hidden_layer[i]            

        #create weigth_i

        weight_i = np.random.randn(weight_i_row, weight_i_column)

        

        #improve weight and bias

        #improve bias_i using devine by square root of number of node in hidden_layer[i]

        bias_i = bias_i/np.sqrt(hidden_layer[i])

        #improve weight_i using devine by square root of number of node in hidden_layer[0]

        weight_i = weight_i/np.sqrt(hidden_layer[0])

        

        #add list

        #add bias_i to bias list

        bias.append(bias_i)

        #add weight_i to weight list

        weight.append(weight_i)

        

    return weight, bias
def neural_network_forward(feature_list, weight, bias, activate_function):

    #create empty list of output

    output = []

    #create empty list of activated_output

    activated_output = []

    

    #for activate_function

    for i in range(len(activate_function)):

        

        if i <= 0:

            #first layer calculate output_i from feature_list

            output_i = np.dot(feature_list, weight[i]) + bias[i]

        else:

            #first layer calculate output_i from last activated_output

            output_i = np.dot(activated_output[-1], weight[i]) + bias[i]

        

        #compute_activated_output by activate_function[i]

        activated_output_i = neural_network_compute_activated_output(output_i, activate_function[i])

                

        #add output_i and activated_output_i to list

        output.append(output_i)

        activated_output.append(activated_output_i)

        

    return output,activated_output
def neural_network_compute_activated_output(output_i, activate_function):

    if type(activate_function) == str:

        

        if activate_function == 'sigmoid':

            #sigmoid : activated_output_i = 1/(1+e^(-output_i))

            activated_output_i = 1/(1 + np.exp(-output_i))

            

        elif activate_function == 'tanh':

            #hyperbolic tangent : activated_output_i = ((e^output_i)-(e^-output_i))/((e^output_i)+(e^-output_i))

            activated_output_i = (np.exp(output_i) - np.exp(-output_i))/(np.exp(output_i) + np.exp(-output_i))

            

        elif activate_function == 'ReLU':

            #rectified linear unit : activated_output_i = if output_i <= 0 : 0, if output_i > 0 : output_i

            activated_output_i = output_i * (output_i > 0)

            

        elif activate_function == 'softmax':

            #softmax : e^output_i/sum(e^output_i)

            activated_output_i = np.exp(output_i)/np.exp(output_i).sum(axis=1, keepdims = True)

            

    elif type(activate_function) == list:

        

        if activate_function[0] == 'PReLU':

            #parametric rectified linear unit : activated_output_i = if output_i <= 0 : output_i * new_slope , if output_i > 0 : output_i 

            #remark slope != 1

            activated_output_i = output_i * (output_i > 0) + activate_function[1] * output_i * (output_i <= 0)

            

    return activated_output_i
def neural_network_compute_different(output_i, activated_output_i, activate_function_i):

    

    if type(activate_function_i) == str:

        

        if activate_function_i == 'sigmoid':

            different_i = activated_output_i * (1 - activated_output_i)

            

        elif activate_function_i == 'tanh':

            different_i = 1 - activated_output_i**2

            

        elif activate_function_i == 'ReLU':

            different_i = (output_i > 0)

            

    elif type(activate_function_i) == list:

        

        if activate_function_i[0] == 'PReLU':

            different_i = (output_i > 0) + activate_function_i[1] * (output_i <= 0)

            

    return different_i
def neural_network_compute_error(delta_i, different_i):

    error_i = delta_i * different_i

    return error_i
def find_error(target_list, output, error_type):

    

    if error_type == 'SSE':

        error = find_sum_square_error(target_list, output)

        

    elif error_type == 'MSE':

        error = find_mean_square_error(target_list, output)

        

    elif error_type == 'MAE':

        error = find_mean_absolute_error(target_list, output)

        

    elif error_type == 'MAPE':

        error = find_mean_absolute_percentage_error(target_list, output)

        

    elif error_type == 'Entropy':

        error = find_entropy_error(target_list, output)

        

    elif error_type == 'Binary':

        error = find_binary_class_error(target_list, output)

        

    elif error_type == 'Multiclass':

        error = find_multi_class_error(target_list, output)

        

    return error
def find_sum_square_error(target_list, output):

    sum_square_error = ((target_list - output)**2).sum()

    return sum_square_error
def find_mean_square_error(target_list, output):

    number_of_sample = target_list.shape[0]

    sum_square_error = ((target_list - output)**2).sum()

    mean_square_error = sum_square_error/number_of_sample

    return mean_square_error
def find_mean_absolute_error(target_list, output):

    number_of_sample = target_list.shape[0]

    mean_absolute_error = (np.abs(target_list - output)).sum()/number_of_sample

    return mean_absolute_error
def find_mean_absolute_percentage_error(target_list, output):

    number_of_sample = target_list.shape[0]

    mean_absolute_percentage_error = np.abs((target_list - output)/target_list).sum()*100/number_of_sample

    return mean_absolute_percentage_error
def find_entropy_error(target_list, output):

    log_output = np.log(output)

    entropy_error = (-target_list*log_output).sum()

    return entropy_error
def find_binary_class_error(target_list, output):

    number_of_sample = target_list.shape[0]

    _target_list = np.round(target_list, 0)

    _output = np.round(output, 0)

    binary_class_error = 100*(_target_list != _output).sum()/number_of_sample

    return binary_class_error
def find_multi_class_error(target_list, output):

    number_of_sample = target_list.shape[0]

    argmax_of_target_list = np.argmax(target_list, axis=1)

    argmax_of_output = np.argmax(output, axis=1)

    multi_class_error = 100*(argmax_of_target_list != argmax_of_output).sum()/number_of_sample

    return multi_class_error
def create_onehot_target(label):

    

    #define unique_label for column

    unique_label = len(np.unique(label))

    

    #define number_of_label for row

    number_of_label = label.shape[0]

    

    #create zeros metrix column = number_of_label, row = unique_label

    onehot = np.zeros([number_of_label, unique_label])

    

    for i in range(number_of_label):

        #add 1 at label type for each row in zeros metrix

        onehot[i, label[i]] = 1

        

    return onehot
def neural_network_classification_find_weight_bias(feature_list, target_list, hidden_layer, activate_function, weight = [], bias = [], epoch = 1000, learning_rate = 0.01):

   

    #number_of_layer = length of hidden_layer

    number_of_layer = len(hidden_layer)

    #number_of_trianing_data = row of feature_list

    number_of_trianing = feature_list.shape[0]

    

    #if weight == [] create a new one

    if not weight:        

        weight, bias = neural_network_create_weight_bias(feature_list, hidden_layer)

        

    #create empty error_list

    error_list = []

    percent = 0

    for i in range(epoch):

        

        #print progress

        new_percent = int(i*100/epoch)

        if(new_percent > percent):

            percent = new_percent

            print(percent)

            

        #calculate output and activated output for each hidden layer

        output, activated_output = neural_network_forward(feature_list, weight, bias, activate_function)

        

        #find error using output of last layer

        #find error by entropy error using "Entropy"

        error = find_error(target_list, activated_output[-1], 'Entropy')

         #collact error in error_list for error trend

        error_list.append(error)

        

        #calculate slope of weight and bias by backpropagation

        slope_of_weight, slope_of_bias = neural_network_classification_backpropagation(feature_list, weight, bias, output, activated_output, target_list, activate_function)

        

        #loop = number of layer for create new weight,bias using gradient descent

        for layer_i in range(number_of_layer):

            #gradient descent weight_new = weight - (learning_rate * (1/n) * slope_of_weight)

            weight[layer_i] = weight[layer_i] + (learning_rate * (1/number_of_trianing) * slope_of_weight[layer_i])

            bias[layer_i] = bias[layer_i] + (learning_rate * (1/number_of_trianing) * slope_of_bias[layer_i])

            

    return weight, bias, error_list
def neural_network_classification_backpropagation(feature_list, weight, bias, output, activated_output, target_list, activate_function):

    

    #number_of_layer = length of activate_function

    number_of_layer = len(activate_function)

    

    #create empty list

    slope_weight = []

    slope_bias = []

    

    #for begin in last layer

    index_of_last_layer = number_of_layer - 1

    #step -1, stop at 0    

    #range(start,stop,step)

    for i in range(index_of_last_layer, -1, -1):

        

        if i >= number_of_layer - 1:

            #last layer delta = target_list - activated_output of last layer 

            delta_i = target_list - activated_output[i]

            different_i = 1

        else:

            #other layer delta_i = error_i of layer [i+1] . transpose(weight of layer [i+1])

            #remark use error_i = error_i of layer [i+1] because error_i is collected by previous round of loop

            delta_i = np.dot(error_i, weight[i+1].T)

            different_i = neural_network_compute_different(output[i], activated_output[i], activate_function[i])

            

        #collect error_i to error_i for calculate previous layer

        error_i = neural_network_compute_error(delta_i, different_i)

        

        if i <= 0:

            #at first layer slope_weight_i = transpose(feature_list) . error_i

            slope_weight_i = np.dot(feature_list.T, error_i)

        else:

            #other layer slope_weight_i = transpose(activated_output of [i-1] layer) . error_i

            slope_weight_i = np.dot(activated_output[i-1].T, error_i)

            

        #slope_bias_i = sum(error_i)

        slope_bias_i = error_i.sum(axis=0)

        

        #add slope_weight_i,slope_bias_i to list

        slope_weight.append(slope_weight_i)

        slope_bias.append(slope_bias_i)

        

    #convert [slope_weight_3,slope_weight_2,slope_weight_1] to [slope_weight_1,slope_weight_2,slope_weight_3] 

    slope_weight =  slope_weight[::-1]

    slope_bias =  slope_bias[::-1]

    

    return slope_weight, slope_bias
def train_test_split(feature_list, target_list, train_size_percent = 80):

    

    #define N

    number_of_data = feature_list.shape[0]

    

    #random for split

    arr_rand = np.random.rand(number_of_data)

    

    #split random array using train_size_percent 

    split = arr_rand < np.percentile(arr_rand, train_size_percent)

    

    #split

    feature_list_train = feature_list[split]

    target_list_train = target_list[split]

    feature_list_test =  feature_list[~split]

    target_list_test = target_list[~split]

    

    return feature_list_train,target_list_train,feature_list_test,target_list_test
#read data values

raw_csv = pd.read_csv('../input/mnist-csv/train.csv')

raw_data = raw_csv.values

feature = raw_data[:,1:]

target = raw_data[:,0]
#selection feature

# all_zero_index = np.argwhere(np.all(feature[..., :] == 0, axis=0))

# new_feature_list = np.delete(feature, all_zero_index, axis=1)

# feature = new_feature_list
#split train-test data

feature_list_train,target_list_train,feature_list_test,target_list_test = train_test_split(feature,target)
#normalize with divide by 255( value of image is 0 to 255)

feature_list_train = feature_list_train/255

feature_list_test = feature_list_test/255



#create one hot matrix for classification

target_list_train_onehot = create_onehot_target(target_list_train)

target_list_test_onehot = create_onehot_target(target_list_test)
#last layer of clasification = unique target

unique_label = len(np.unique(target_list_train))



#defind node in hidden layer

hidden_layer = [15,15,unique_label]
#define activated for each layer 

#for clasification use softmax in last layer

activated_function = [['PReLU',0.1], ['PReLU',0.1], 'softmax']
#random initial weight, bias for train model

weight, bias = neural_network_create_weight_bias(feature_list_train, hidden_layer)
#train model

weight, bias, error_list = neural_network_classification_find_weight_bias(feature_list_train, target_list_train_onehot, hidden_layer, activated_function, weight = weight, bias = bias, epoch = 10000, learning_rate = 0.1)
#plot error list

plt.plot(error_list)
#predict Yhat_train

Zhat_train, Yhat_train = neural_network_forward(feature_list_train, weight, bias, activated_function)

#find Yhat_train muticlass error

error_train = find_error(target_list_train_onehot, Yhat_train[-1], 'Multiclass')

#print

print(error_train)
#predict Yhat_test

Zhat_test, Yhat_test = neural_network_forward(feature_list_test, weight, bias, activated_function)

#find Yhat_test muticlass error

error_test = find_error(target_list_test_onehot, Yhat_test[-1], 'Multiclass')

#print

print(error_test)
#view error prediction image

not_match_count = 0

#for in feature_list_test

for i in range(len(feature_list_test)):

        #argmax predicted value

        predicted = np.argmax(Yhat_test[-1], axis=1)[i]

        #label value

        label = np.argmax(target_list_test_onehot, axis=1)[i]

        #check predicted

        if(predicted != label):

            #count not_match]

            not_match_count += 1

            #show first 10 not match image

            if(not_match_count <= 10):

                plt.title('Predicted: {}, Actual: {}'.format(predicted,label))

                plt.imshow((feature_list_test[i]*255).reshape(28,28), cmap='gray')

                plt.show()
#view percent error and not match count

print("not match count : {0}".format(not_match_count))

print("feature list test count : {0}".format(len(feature_list_test)))

print("accuracy : {:.4f}%".format((1-(not_match_count/len(feature_list_test)))*100))