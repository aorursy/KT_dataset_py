import numpy as np # linear algebra
import pandas as pd

test_input_xor=np.array([[0,0],[0,1],[1,0],[1,1]])
correct_outputs_xor=[False, True, True, False]

weight1_AND=1.0
weight2_AND=1.0
bias_AND=-2.0


weight1_OR=2.0
weight2_OR=2.0
bias_OR=-1.0

weight1_NOT=-1.0
weight2_NOT=-2.0
bias_NOT=0.0
def train_Model_Logic(weight1,weight2,test_inputs,bias,name="And"):
    linear_combination=np.multiply(weight1,test_inputs[:,:1])+np.multiply(weight2,test_inputs[:,1:2])+bias
    output=1*(linear_combination>=0)
    return output
def train_XOR(train_inputs,correct_outputs,weight1_and,weight2_and,bias_and,weight1_or,weight2_or,bias_or,weight1_not,weight2_not,bias_not,name="XOR"):
    layer1_and=train_Model_Logic(weight1_and,weight2_and,train_inputs,bias_and,name="AND")
    layer1_or=train_Model_Logic(weight1_OR,weight2_OR,train_inputs,bias_OR,name="OR")
    layer2_not=train_Model_Logic(weight1_NOT,weight2_NOT,np.concatenate((layer1_and,np.zeros((4,1))),axis=1).reshape(4,2),bias_NOT,name="NOT")
    layer3_and=train_Model_Logic(weight1_and,weight2_and,np.concatenate((layer1_or,layer2_not),axis=1).reshape(4,2),bias_and,name="AND")

    num_wrong=np.array_equal(layer3_and,correct_outputs)
    out={'Input1':train_inputs[:,:1].ravel().T,'Input2':train_inputs[:,1:2].ravel().T,'Correct Output':correct_outputs,'output':layer3_and.T.ravel()}
    frame=pd.DataFrame(out)
    if not num_wrong:
        print("Successfully build {0} logic gate".format(name))
    else:
        print("Something went wrong")
    return frame
frame=train_XOR(test_input_xor,correct_outputs_xor,weight1_AND,weight2_AND,bias_AND,weight1_OR,weight2_OR,bias_OR,weight1_NOT,weight2_NOT,bias_NOT)
frame

