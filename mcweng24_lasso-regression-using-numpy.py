import numpy as np



data = np.genfromtxt('../input/mystery.dat', delimiter = ',')



#Create a function to get the output

def predicted_values(X, w):

    # X will be n x (d+1)

    # w will be (d+1) x 1

    predictions = np.matmul(X,w) # n x 1

    return(predictions)
#rho computation

def rho_compute(y,X,w,j):

    #y is the response variable

    #X is the predictor variables matrix

    #w is the weight vector

    #j is the feature selector

    X_k = np.delete(X,j,1) #Remove the j variable i.e. j column

    w_k = np.delete(w,j) #Remove the weight j

    predict_k = predicted_values(X_k, w_k)

    residual = y - predict_k

    rho_j = np.sum(X[:,j]*residual)

    return(rho_j)
#z computation for unnormalised features

def z_compute(X):

    z_vector = np.sum(X*X, axis = 0) #Compute sum for each column

    return(z_vector)
def coordinate_descent(y,X,w,alpha,z,tolerance):

    max_step = 100.

    iteration = 0

    while(max_step > tolerance):

        iteration += 1

        #print("Iteration (start) : ",iteration)

        old_weights = np.copy(w)

        #print("\nOld Weights\n",old_weights)

        for j in range(len(w)): #Take the number of features ie columns

            rho_j = rho_compute(y,X,w,j)

            if j == 0: #Intercept is not included with the alpha regularisation

                w[j] = rho_j/z[j]

            elif rho_j < -alpha*len(y):

                w[j] = (rho_j + (alpha*len(y)))/z[j]

            elif rho_j > -alpha*len(y) and rho_j < alpha*len(y):

                w[j] = 0.

            elif rho_j > alpha*len(y):

                w[j] = (rho_j - (alpha*len(y)))/z[j]

            else:

                w[j] = np.NaN

        #print("\nNew Weights\n",w)

        step_sizes = abs(old_weights - w)

        #print("\nStep sizes\n",step_sizes)

        max_step = step_sizes.max()

        #print("\nMax step:",max_step)

        

        

    return(w, iteration, max_step)
#Initialise the data



#101 rows for both input (x) and output (y)

x = data[:,0:100] # 100 predictors - columns 0 to 99

y = data[:,100] # 1 response variable - column 100



#Obtain feature matrix by adding column of 1s to input matrix x

X = np.column_stack((np.ones((x.shape[0],1)),x)) #101 columns



#Initialise weight/parameter vector, w, to be a zero vector

w = np.zeros(X.shape[1], dtype = float)



#Pre-compute the z_j term

z = z_compute(X)



#Set the alpha and tolerance level

alpha = 0.1

tolerance = 0.0001



#Obtain the following from the coordinate descent:

#1. Optimum weight parameter

#2. Number of iterations

#3. Maximum step size at the last iteration



w_opt, iterations, max_step = coordinate_descent(y,X,w,alpha,z,tolerance)
#Print out the optimised weights

np.set_printoptions(precision = 3, suppress = True)

print("Intercept is:",w_opt[0])

print("\nCoefficients are:\n",w_opt[1:101])

print("\nNumber of iterations is:",iterations)
#Sort to see which are the most important features

values = np.sort(abs(w_opt[1:101]))[::-1] 

index = np.argsort(abs(w_opt[1:101]))[::-1] + 1 #Add 1 to not show zero-index



np.set_printoptions(precision = 3, suppress = True)

print(np.column_stack((index,values)))
#Compare with sklearn's Lasso

from sklearn import linear_model

mystery = linear_model.Lasso(alpha = 0.1, tol = 0.0001) #alpha is just the regulariser term

mystery.fit(x,y)



print("sklearn Lasso intercept :",mystery.intercept_)

print("\nsklearn Lasso coefficients :\n",mystery.coef_)

print("\nsklearn Lasso number of iterations :",mystery.n_iter_)
values = np.sort(abs(mystery.coef_))[::-1]

index = np.argsort(abs(mystery.coef_))[::-1] + 1



np.set_printoptions(precision = 3, suppress = True)

print(np.column_stack((index,values)))
#Comparison within the same cell



print("sklearn Lasso intercept :",mystery.intercept_)

print("\nsklearn Lasso coefficients :\n",mystery.coef_)

print("\nsklearn Lasso number of iterations :",mystery.n_iter_)

print("\n------------------------------------------------------------------------\n")

print("Intercept is:",w_opt[0])

print("\nCoefficients are:\n",w_opt[1:101])

print("\nNumber of iterations is:",iterations)
w_opt[1:101]
print("Values are compared to sklearn's Lasso results:")

np.set_printoptions(precision = 10, suppress = False)

error = np.zeros(len(w_opt[1:101]), dtype = float)

for j in range(len(w_opt[1:101])):

    if(w_opt[1:101][j] == 0 and mystery.coef_[j] == 0):

        error[j] = 0.

    else:

        error[j] = abs(((w_opt[1:101][j] - mystery.coef_[j])/mystery.coef_[j])*100)



print("Maximum difference in coefficients (%):",np.max(error))

print("Average difference in coefficients (%):",np.mean(error))



intercept_diff = abs(((w_opt[0] - mystery.intercept_)/mystery.intercept_)*100)

print("Difference in intercept (%):",intercept_diff)


