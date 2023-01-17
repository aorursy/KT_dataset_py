# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
test_input_and=np.array([[0,0],[0,1],[1,0],[1,1]])
correct_outputs_and=[False,False,False,True]
weight1_and=1.0
weight2_and=1.0
bias_and=-2.0
test_input_or=np.array([[0,0],[0,1],[1,0],[1,1]])
correct_outputs_or=[False,True,True,True]

weight1_or=2.0
weight2_or=2.0
bias_or=-1.0
test_input_not=np.array([[0,0],[0,1],[1,0],[1,1]])
correct_outputs_not=[True, False, True, False]

weight1_not=1.0
weight2_not=-2.0
bias_not=0.0

def train_Model_Logic(weight1,weight2,test_inputs,correct_outputs,bias,name="And"):
    outputs=[]
    linear_combination=np.multiply(weight1,test_inputs[:,:1])+np.multiply(weight2,test_inputs[:,1:2])+bias
    output=(linear_combination>=0)
    num_wrong=np.array_equal(output,correct_outputs)
    #outputs.append([test_inputs[:,:1].T,test_inputs[:,1:2].T,correct_outputs.T,linear_combination.T,output.T])
    out={'Input1':test_inputs[:,:1].ravel().T,'Input2':test_inputs[:,1:2].ravel().T,'Correct Output':correct_outputs,'linear_combination':linear_combination.T.ravel(),'output':output.T.ravel()}
   # output_frame=pd.DataFrame(outputs,columns=['Input1','Input2','Correct Output','linear_combination','output'])
    frame=pd.DataFrame(out)
    if not num_wrong:
        print("Successfully build {0} logic gate".format(name))
    else:
        print("Something went wrong")
    return frame
output_and=train_Model_Logic(weight1_and,weight2_and,test_input_and,correct_outputs_and,bias_and)

output_and
output_or=train_Model_Logic(weight1_or,weight2_or,test_input_or,correct_outputs_or,bias_and,name='OR')
output_or
output_not=train_Model_Logic(weight1_not,weight2_not,test_input_not,correct_outputs_not,bias_not,name="NOT")
output_not
