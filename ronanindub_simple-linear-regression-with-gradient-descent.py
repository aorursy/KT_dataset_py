import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Set up the X values.



# Anscombe's Quartet - Set 1 - X Values



X=np.array([10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0 ,12.0 ,7.0, 5.0], dtype="float64")



X.flags.writeable = False   # Protecting the contents of X - make it immutable (ie. read only).
# Set up the Y values.





# Anscombe's Quartet - Set 1 - Y Values



Y=np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68], dtype="float64")



Y.flags.writeable = False  # Protecting the contents of Y - make it immutable (ie. read only).

# Setting up some global values and also plotting the data to get a feel for it.



Xmin=np.min(X)

Xmax=np.max(X)

Ymin=np.min(Y)

Ymax=np.max(Y)

print("X min & max:", Xmin, Xmax)

print("Y min & max:", Ymin, Ymax)





# plt.xlim(Xmin-1,Xmax+1)

plt.xlim(0,Xmax+1)

plt.ylim(0,Ymax+1)



plt.scatter(X,Y, 10, color = 'blue')

plt.show()
X_s=np.empty_like(X, dtype="float64")     # X_s  X standardised

Y_s=np.empty_like(Y, dtype="float64")     # Y_s  Y standardised



X_s = (X - Xmin)/(Xmax-Xmin)

Y_s = (Y - Ymin)/(Ymax-Ymin)





for i in range(0, X.size):

    print("i= ", i, "\t\tx {:6.2f}".format(X_s[i]),"\ty {:6.2f}".format(Y_s[i]))







plt.scatter(X_s,Y_s, 10, color = 'blue')

plt.show()
def CalculateNewY(X_orig, slope, intercept):

    

    Y_calc = np.empty_like(X_orig, dtype="float64")



    Y_calc = X_orig*slope+intercept

        

    return Y_calc
def CalculateSSE(original_Y, predicted_Y):

    theSSE=0.0

    

    for i in range(0, original_Y.size):

        theSSE += (original_Y[i]-predicted_Y[i])**2

        

    theSSE = theSSE/2

    

    return theSSE
def SumPartialDerivativeOf_m (original_Y, calculated_Y , original_X):

    

    theSPD_m = 0.0

    

    for i in range(0, original_Y.size):

        theSPD_m += original_X[i] *(calculated_Y[i]-original_Y[i])      



    return theSPD_m
def SumPartialDerivativeOf_c (original_Y, calculated_Y ):

    

    theSPD_c = 0.0

    

    for i in range(0, original_Y.size):

        theSPD_c +=  calculated_Y[i] - original_Y[i]



        

    return theSPD_c
# Helper function



def DrawLineFromFormula(slope, intercept, color):

    plt.xlim(-0.05, 1.05)

    plt.ylim(-0.05, 1.05)

    x = np.arange(-100, 100, 0.1)

    plt.plot(x, slope*x+intercept, color)

    return
# The Iteration



# This is where we iterate until the optimistaion equation has stopped getting any better



def trials( m = 1, c = 0.75 , r= 0.01, acceptableDifference = 0.000001, maxNumOfTrials = 10000 ):



    SSE_storage = []

    

    recordOfIterations = []

    recordOfSlope = []

    recordOfIntercept = []

    recordOfSSE = []

    



    for i in range(0, maxNumOfTrials):    

    

        Y_hat = CalculateNewY(X_s, m, c)



        ourSSE = CalculateSSE(Y_s,Y_hat)

    

        SSE_storage.append(ourSSE)    # This list is used to store the SSE errors - used for plotting later.

        

 

        if ( i > 0):

            

            if ( abs(oldSSE-ourSSE) < acceptableDifference):

                print("\nAfter ", i, "iterations - we are done({:.10f})!\n\nOld SSE:".format(acceptableDifference),oldSSE, " new SSE: ", ourSSE, "\t Difference < {:12.10f}".format(oldSSE-ourSSE),"\n" )

                

                # Make sure to store the last value !

        

                recordOfIterations.append(i)

                recordOfSlope.append(m)

                recordOfIntercept.append(c)

                recordOfSSE.append(ourSSE)        

                break

    

            if( ourSSE > oldSSE):

                print("Error adjustment process going the wrong way ...abort.")

                break





        ourSPD_m = SumPartialDerivativeOf_m( Y_s, Y_hat, X_s)

        ourSPD_c = SumPartialDerivativeOf_c( Y_s, Y_hat)



        m = m - r*ourSPD_m

        c = c - r*ourSPD_c



        if (i%100 == 0):

#       print("{:12}".format(i),"{:12.6f}".format(m), "{:12.6f}".format(c), "{:16.14f}".format(ourSSE))

            recordOfIterations.append(i)

            recordOfSlope.append(m)

            recordOfIntercept.append(c)

            recordOfSSE.append(ourSSE)

        

        if((i%100 ==0)):

            DrawLineFromFormula(m, c, 'g--')

            

        oldSSE = ourSSE

        



# Show the table of values



    whatHappened = pd.DataFrame({"Iterations":recordOfIterations, 

                                "Slope":recordOfSlope,

                                "c":recordOfIntercept,    

                                "SSE":recordOfSSE

                            })



    pd.set_option('display.max_rows', None)

    display(whatHappened)



# Plot the original points and the final line.



    plt.scatter(X_s, Y_s, 30, color = 'blue')



    DrawLineFromFormula(m, c, 'black')

    

    plt.show()



# Plot the SSE - it should show a nice decrease.

# Plotting the error

    

    plt.title("SSE Plot")

    plt.xlabel('Number of iterations')

    plt.ylabel('SSE')

    plt.plot(SSE_storage)

    plt.show()

    



    return m,c

m_slope = 1.1

c_intercept = 2.0

r_learning_rate = .01



m,c = trials( m_slope, c_intercept, r_learning_rate, 0.0000000001, 3000)

print(m, c)

print("Ymin:", Ymin)

print("Y Range:", Ymax - Ymin)



c_final = (c * (Ymax - Ymin)) + Ymin



print("m {:6.4f}".format(m),"\t final c {:6.4f}".format(c_final))



# plt.xlim(-1,Xmax+1)

# plt.ylim(2,Ymax+1)



x = np.arange(0, Xmax+1, 0.1, dtype="float64")

y = np.empty_like(x, dtype="float64") 

y = m*x + c_final



# plt.plot(x, y, 'black')

# Plot the original points and the final line.



points = np.arange(Xmin-1, Xmax+1, 0.1)

plt.plot(points, m*points+c_final, 'g--')





plt.scatter(X,Y, 10, color = 'blue')

plt.show()