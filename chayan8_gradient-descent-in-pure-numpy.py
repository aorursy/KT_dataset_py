# Import NumPy and initialize Variable x and target y



import numpy as np



x = np.array([1,3,5])

y = np.array([5,12,18])
w1 =((x*y).mean() - x.mean()*y.mean())/((x**2).mean() - (x.mean())**2)

w1
w0 = y.mean() - w1*x.mean()

w0
# Step 1: Initializing weights to 0, 

# a (learning parameter) to 0.04, 

# and an array MSE for storing MSE at each iteration



W0_new = 0

W1_new = 0

a = 0.04

MSE = np.array([])
for iteration in range(1,11):

    

    y_pred = np.array([])                        # The Predicted target

    error = np.array([])                         # The errors per iterations : (Ŷ-Y)

    error_x = np.array([])                       # The (Ŷ-Y).X term for update rule

    

    W0 = W0_new                                  # Step 1 and 4: Initializing new weights/Assigning the updated weights

    W1 = W1_new



    # Step 2:    

    

    for i in x:                                  # Iterating X row by row for calculating the Ŷ and error

        y_pred = np.append(y_pred,(W0 + W1*i))   # Ŷ = W0 + W*X 



    error = np.append(error,y_pred-y)            # Calculating the error for each sample 

    error_x = np.append(error_x, error*x)        # Calculating the (Ŷ-Y).X term for update rule

    MSE_val = (error**2).mean()                  # Calculating the MSE    

    MSE = np.append(MSE,MSE_val)

    

    # Step 3:   



    W0_new = W0 - a*np.sum(error)               # Calculating the updated W0   

    W1_new = W1 - a*np.sum(error_x)             # Calculating the updated W1
print('W0 by gradient descent= ',W0_new)

print('W1 by gradient descent= ',W1_new)
print('y_pred: ',y_pred)

print('error:  ',error)
import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(MSE,'b-o')

plt.title('Mean Square error per iteration')

plt.xlabel('Iterations')

plt.ylabel('MSE value')

plt.show()