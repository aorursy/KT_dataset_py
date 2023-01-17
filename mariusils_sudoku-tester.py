from keras.utils import CustomObjectScope, to_categorical
from keras.initializers import glorot_uniform
from keras.models import model_from_json
import numpy as np
import timeit

json_file = open('../input/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):

    loaded_model = model_from_json(loaded_model_json)


# load weights into new model
loaded_model.load_weights("../input/Sudoku_Solver.h5")
print("Loaded model from disk")

Big_X = ['102600854076000900800053017657000000280700005340010000915020436403960570068035291']


real = ['132679854576184923894253617657342189281796345349518762915827436423961578768435291']

for i in range(len(Big_X)):

    Big_X[i] = np.array(list(Big_X[i]), dtype="float32")

    real[i] = np.array(list(real[i]), dtype="float32")


sudoku = np.copy(Big_X)
    
start = timeit.timeit()

count = 0

while True:
    
    count +=1
    
    X = to_categorical(Big_X).astype("float32")
    
    solved = loaded_model.predict(X)

    solved = np.array(solved)

    beta = (1/7.5)*np.log(count)
    
    x1, x2, x3 = np.shape(solved)

    for i in range(int(x2)):     

        for  n, item in enumerate(solved[:, i]):

            if np.max(item)>0.85 and np.random.rand()>beta:
            
                sudoku[0][n]=int(np.argmax(item)+1)
                
    Big_X = np.copy(sudoku)
    
    if count == 10000:
        
        break
                
print(sudoku - real[i])
   
end = timeit.timeit()
print(end - start)