#All libraries are imported, the data loaded,cleaned and made ready for action

import pandas as pd

import numpy as np

from sklearn.cross_validation import  train_test_split

import math

from sklearn import preprocessing, svm

from sklearn import metrics

import matplotlib.pyplot as plt

from matplotlib import cm

import time

from mpl_toolkits.mplot3d import axes3d



start_time = time.time()

df =  pd.read_csv('../input/data.csv', header=0)

df[df.columns[1]] = df[df.columns[1]].map( {'B': 0, 'M': 1} ).astype(int)

df.replace('?', -99999, inplace=True)

df.drop(['id','Unnamed: 32'], 1, inplace=True)

X = np.array(df.drop(['diagnosis'], 1))

y = np.array(df['diagnosis'])

Features=X

Labels=y

#The parameter space to be tested is defined and the number of simulations set





# I encourage the user to tweak these values in order to increase accuracy of the estimate,as well as to

# influence processing time.

C_values=[math.pow(10,x/20) for x in range(-3,22)]   

Gamma_values=[math.pow(10,x/20) for x in range(-46,-30)]  

number_of_simulations=15



#The Space for the 3D plotting is prepared.

opt_dict={}

Y=np.array(Gamma_values)

X=np.array(C_values)

X, Y = np.meshgrid(X, Y)

Z=X*Y
# This is the core of the calculations: a hierarchical function consisting of three levels.

# At the bottom level is the working level, where the magic happens. For the number of simulations

# defined above, a SVM is trained and tested at always different splits of the data.The values of Gamma

# and C remain constant and are defined in the upper levels. The accuracy for each simulation is saved.



def Level_three_simulations(X_est,y_est,C_index,Gamma_index):

    accuracies=[]

    for i in range(number_of_simulations):

        X = preprocessing.scale(X_est)

        X_train, X_test, y_train, y_test = train_test_split(X, y_est, test_size=0.25)

        clf = svm.SVC(C=C_values[C_index],gamma=Gamma_values[Gamma_index], kernel='rbf')

        clf.fit(X_train, y_train)



        predictions = clf.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, predictions)

        accuracies.append(accuracy)

    return accuracies



# Level two defines the C alues for level three. Here, the final value mean value for each

# C/Gamma combination is computed.



def Level_two_C(X,y,Gamma_index):

    C_dependent_acc=[]

    for C_index in range(len(C_values)):

        accuracies=Level_three_simulations(X,y,C_index,Gamma_index)

        mean_acc = (sum(accuracies) / len(accuracies))

        C_dependent_acc.append(mean_acc)

    return C_dependent_acc



# Level one loops through all Gamma values and passes them to level two. Here, at level one 

# the final data matrix Z is created.



def Level_one_Gamma(X,y):

    for Gamma_index in  range(len(Gamma_values)):

        C_dependent_acc = Level_two_C(X,y,Gamma_index)

        Z[Gamma_index]=C_dependent_acc

    return Z
#The functions are actually called

Z=Level_one_Gamma(Features,Labels)



# The value and coordinates of the best parameter combination is retrieved.

maxi=np.amax(Z)

indices=np.where(Z==Z.max())



#The coordinates for the 'C path' and the 'Gamma path' are established

max_coord_C=np.argmax(Z,axis=1)

C=[]

Gamma=[]

for index in range(len(max_coord_C)):

    Gamma.append(C_values[max_coord_C[index]])

    C.append(Gamma_values[index])



max_coord_C=np.argmax(Z,axis=0)

C1=[]

Gamma1=[]

for index in range(len(max_coord_C)):

    Gamma1.append(Gamma_values[max_coord_C[index]])

    C1.append(C_values[index])
# Everyting is being plotted

# In normal Python programs, the plot can be rotated. Without this function, the resulting plot

# can appear somewhat impenetrable. I therefore recommend,to run the code in regular Python programs.



fig = plt.figure(figsize=(10,10))

ax = fig.gca(projection='3d')



ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.3, cmap=cm.BuPu)

ax.plot(Gamma, C, np.amax(Z,axis=1), label='Path along Gamma')

ax.plot(C1, Gamma1, np.amax(Z,axis=0), label='Path along C')

ax.scatter(C_values[indices[1]], Gamma_values[indices[0]], maxi, marker='*', color='r', label='Maximum Accuracy')



ax.set_title('Parameter Space C-Gamma')

ax.set_xlabel('C')

ax.set_ylabel('Gamma')

ax.set_zlabel('Accuracy')

ax.legend()



ax.view_init(azim=30)     #The angle of viewing within the notebook can be adjusted here 



plt.show()
# The results are printed

print('The best combination of the Parameters Gamma and C are: Gamma = ',Gamma_values[indices[0]],', C=',C_values[indices[1]])

print('The accuracy obtained with this combination is', maxi*100,'%. The out-of-sample accuracy on a Cross Validation set is likely to be lower.')

if indices[0]==0 or indices[0]== len(Gamma_values)-1 or indices[1]==0 or indices[1]==len(C_values)-1:

    print('Warning: At least one of the estimated optimal paramters lies on the margin of the tested space. There is a chance the optimal combination is missed.')

    print('Please re-run the program and adjust the values for Gamma and C accordingly.')

print("--- %s seconds ---" % (time.time() - start_time))