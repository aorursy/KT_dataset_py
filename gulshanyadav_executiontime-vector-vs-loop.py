import numpy as np # linear algebra

import time



x = np.random.rand(100000)  # creates 1 lack random numbers

y = np.random.rand(100000)



start_time = time.time()

mult_result = np.dot(x,y)

end_time = time.time()



print(mult_result)

print(f"Vector Execution time = {((end_time-start_time)*1000)}")



loop_result = 0

start_time = time.time()



for i in range (100000):

    loop_result += x[i]*y[i]

end_time = time.time()    



print(loop_result)

print(f"for Loop Execution time = {((end_time-start_time)*1000)}") 
