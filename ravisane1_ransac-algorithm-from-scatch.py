import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv('../input/red-wine-quality/winequality-red.csv')

x = dataset.iloc[:,:-1].values

y_o = dataset.iloc[:,-1].values



#plt.scatter(x, y_o)

print(f"Total number of data points: {len(y_o)}")
def RANSAC(itr, limit, tau, x, y_o, m):

    """

    Inputs- itr : no of iteration

            limit : the value of k^2 in chebyshev inequality

            tau : minimum fraction of points that must satisfy chebyshev inequality

            x : x coordinate data

            y_o : y coordinate data

            m : number of unknows

            

    Outputs- best_slope : best slope calculated from RANSAC

             best_intercept : best intercept calculated from RANSAC

    """

    mode = tau * len(y_o)

    max_counter = 0 # counter to store highest number of points considered as inliers

    

    for i in range(itr):

        print(f"\nItr No:{i+1}")

        s = np.shape(x)

        index = np.random.randint(0, s[0], m+1) #randomly select m+1 points from given dataset

        

        Y_vector = [] #create a Y_vector = X_matrix * unknown_vector

        X_matrix = np.zeros((m+1, m+1))

        for j in range(m+1):

            X_matrix[j,:] = np.concatenate((x[index[j]], np.ones((1))))

            Y_vector = np.concatenate((Y_vector, y_o[index[j]:index[j]+1]), axis = 0)

        Y_vector = np.reshape(Y_vector,(len(Y_vector),1))

        

        i = np.eye(m+1, m+1)

        X_inverse = np.linalg.lstsq(X_matrix, i, rcond=None)[0]

        unknown = np.matmul(X_inverse, Y_vector)    # get the value of unknown_vector = X_inverse * Y_vector

        

        y_new = np.zeros((len(y_o))) + unknown[-1]

        for j in range(m):

            y_new[:] += unknown[0]*x[:,j]  #get new line that is y_new value



        Error = (y_o - y_new)**2  #get error square

        mu = Error.mean() #get mean of error square

        std = Error.std() #get standard dev of error square

        #print(mu, std)



        counter = 0 #to store number of points satisfy chebyshev inequality



        #loop to count number of points satisfy chebyshev inequality

        for j in range(0,len(y_o)):

            if np.abs((Error[j]-mu)/std) < (limit/mu):

                counter += 1

                

        if counter > max_counter: # to store best solution

            max_counter = counter

            best_value = unknown

    

        '''#show plot of new line 

        plt.plot(x, y_new, c = 'red')

        plt.scatter(x, y_o)

        plt.show()'''

        

        print(f"Points satisfied: {counter}")



        #loop termination criterior

        if counter >= mode:

            break

            

    if max_counter == 0: #if no solution then store coefficient value = 0

        best_value = np.zeros((len(y_o), 1))

    return best_value, max_counter

#init parameters of RANSAC 

itr = 500 #no of iteration

limit = 25 #the value of k^2 in chebyshev inequality

tau = 0.6 #minimum fraction of points that must satisfy chebyshev inequality

number_of_unknowns = np.shape(x)[1]

calculated_value, inliers = RANSAC(itr, limit, tau, x, y_o, number_of_unknowns)
print("The equation of line is: \n")

print("y = ", end = "")

for j in range(number_of_unknowns):

    print(f"{calculated_value[j]}*x[{j}] + ", end = "")

print(f"{calculated_value[number_of_unknowns]}")



print(f"\n\n Fraction of points satisfied inliers = {inliers/len(y_o)}")