import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

plt.rcParams['figure.figsize'] = [15, 10]

import warnings

warnings.filterwarnings('ignore')

dt = pd.DataFrame()
# This function uses 3 points coordinates, with a random_state 0 taken by defauls

def reset_clusters(M1 = [0,0], M2 = [2,2], M3 = [-2,2], random_state_cl = 0): 

    

    # First cluster ( 50 samples )

    np.random.seed(random_state_cl)

    X1 = M1[0] + np.random.randn(1,50)[0]

    Y1 = M1[1] + np.random.randn(1,50)[0]

    L1 = [1]*(X1.size)





    # Second cluster ( 60 samples )

    X2 = M2[0] + np.random.randn(1,60)[0]

    Y2 = M2[1] + np.random.randn(1,60)[0]

    L2 = [2]*(X2.size)





    # Third cluster ( 40 samples )

    X3 = M3[0] + np.random.randn(1,40)[0]

    Y3 = M3[1] + np.random.randn(1,40)[0]

    D1 = {'X3' : X3,

          'Y3' : Y3}

    L3 = [3]*(X3.size)





    X = np.concatenate((X1,X2,X3))

    xmin = min(X) 

    xmax = max(X)



    Y = np.concatenate((Y1,Y2,Y3))

    ymin = min(Y) 

    ymax = max(Y)



    L = np.concatenate((L1,L2,L3))



    # Creating the data frame 

    dt['X'] = X          # x coordinates

    dt['Y'] = Y          # y coordinates

    dt['L_true'] = L     # true label showing which cluster each point of the set is really belonging to

    dt['L_pred'] = 0     # will serve us to store the label to which cluster each point is predicted to belong 



    return dt
print("Original Data")

dt = reset_clusters()

plt.figure()

plt.scatter(dt['X'], dt['Y'], c = dt['L_true'])
def reset_centroids(random_state_ce = 0):

    np.random.seed(random_state_ce)

    

    # it is obvious that our centroids are related to the data points, that's why we call the reset_cluster funcion first

    dt = reset_clusters(random_state_cl = random_state_ce) 

    

    xmax = np.max(dt['X'])

    xmin = np.min(dt['X'])

    ymax = np.max(dt['Y'])

    ymin = np.min(dt['Y'])

    

     #I used random() to generate random numbers between 0 and 1, so we can ajust the centroids depending on data intervals

    

    ax, bx, cx = xmin + np.random.random()*xmax, np.random.random()*xmax, np.random.random()*xmax

    ay, by, cy = ymin + np.random.random()*ymax, np.random.random()*ymax, np.random.random()*ymax

    

    # besides returning the centroids coordinates, it is better to return the data frame for a later use

    return dt, [ax, ay], [bx, by], [cx, cy] 
dt, c1, c2, c3 = reset_centroids()

M1 = [0,0]

M2 = [2,2]

M3 = [-2,2]

plt.scatter(dt['X'], dt['Y'], c = dt['L_true'], label = 'Original Data')

plt.scatter([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]], c= 'r', s= 150, marker = '^', label = 'Initial Centroids')

plt.scatter([M1[0],M2[0],M3[0]], [M1[1],M2[1],M3[1]], s = 200, marker = '*', c = 'orange', label = 'Real centers')

plt.legend( loc='lower left')

plt.show()
def k_means(d, random_state_km = 0, plotting = False):

    

    # the real centers

    M1 = [0,0]

    M2 = [2,2]

    M3 = [-2,2]

    

    # initiate centroids and data points

    dt, c1, c2, c3 = reset_centroids(random_state_ce = random_state_km)

    ax, bx, cx, ay, by, cy = c1[0], c2[0], c3[0], c1[1], c2[1], c3[1]

    

    # plotting original data with initial centroids

    if plotting :



        plt.figure()

        plt.scatter(dt['X'], dt['Y'], c = dt['L_true'], label = 'Original Data')

        plt.scatter([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]], c= 'r', s= 150, marker = '^', label = 'Initial Centroids')

        plt.title("Original Data")

        plt.legend( loc='lower left')

        plt.show()

    

    

    # we create lists where to store centroid coordinates in order to visualize their movement path

    axs = [ax]

    ays = [ay]

    bxs = [bx]

    bys = [by]

    cxs = [cx]

    cys = [cy]

    if plotting :

        plt.scatter(ax, ay, c= 'r', s= 100, alpha = 0.2, marker = '^')

        plt.scatter(bx, by, c= 'b', s= 100, alpha = 0.2, marker = '^')

        plt.scatter(cx, cy, c= 'g', s= 100, alpha = 0.2, marker = '^')

            

            

    # int the scores list we will store the accuracy of the model after evry update ( iteration )

    scores = []

    

    # list of iteration order ( useful for the plot of the scores )

    iterations = []

    for k in range(d):



        # we create lists to store the points assigned to each centroid

        Anx = []

        Bnx = []

        Cnx = []



        Any = []

        Bny = []

        Cny = []

        for i in range(dt.shape[0]): # loop over x and y coordinates for all points

            x = dt['X'][i]

            y = dt['Y'][i]

            

            # calculate the distances between a point and the 3 centroids

            da = np.sqrt((x-ax)**2 + (y-ay)**2)

            db = np.sqrt((x-bx)**2 + (y-by)**2)

            dc = np.sqrt((x-cx)**2 + (y-cy)**2)

            

            # classify the point depending on the closest centroid (the minimal distance)

            # we don't know yet which real cluster ( 1, 2 or 3 ) this point is belonging to

            # that's why we just note ( class_A class_B or class_C )

            # we will decode this classification later using the maximal accuracy 

            if da == min(da,db,dc):

                dt['L_pred'][i] = 'class_A'

                Anx.append(x)

                Any.append(y)

            elif db == min(da,db,dc):

                dt['L_pred'][i] = 'class_B' 

                Bnx.append(x)

                Bny.append(y) 

            elif dc == min(da,db,dc):

                dt['L_pred'][i] = 'class_C'

                Cnx.append(x)

                Cny.append(y)



        # Updating the centroids x and y coordinates, and adding the new values to the path lists

        ax = np.mean(np.array(Anx))

        axs.append(ax)

        ay = np.mean(np.array(Any))

        ays.append(ay)

        bx = np.mean(np.array(Bnx))

        bxs.append(bx)

        by = np.mean(np.array(Bny))

        bys.append(by)

        cx = np.mean(np.array(Cnx))

        cxs.append(cx)

        cy = np.mean(np.array(Cny))

        cys.append(cy)

        

        

        labelize(dt)             # helps us decode which label ( 1, 2 or 3 ) every class ( A, B and C ) are really meaning 

        switch_labels(dt)        # uses the the results of "labelize" then assign each class to the real label

        # those functions are defined later

        



        # updating scores and itertions lists

        scores.append(test_score(dt))

        iterations.append(k)

        

        

        # plotting the path of centroids every iteration

        if plotting :

            pt = plt.figure

            plt.scatter(dt['X'], dt['Y'], c = dt['L_pred'])

            plt.scatter(ax, ay, c= 'r', s= 150, marker = '^', alpha = ((k+d)/2)/(2*d))

            plt.scatter(bx, by, c= 'b', s= 150, marker = '^', alpha = ((k+d)/2)/(2*d))

            plt.scatter(cx, cy, c= 'g', s= 150, marker = '^', alpha = ((k+d)/2)/(2*d))



    # plotting the coordinates of centroids and centers for the last time ( convergence supposed )

    if plotting :

        pt = plt.figure

        plt.scatter(dt['X'], dt['Y'], c = dt['L_pred'])

        plt.scatter(ax, ay, c= 'r', s= 100, marker = '^', label = "Centroid 1")

        plt.scatter(bx, by, c= 'b', s= 100, marker = '^', label = "Centroid 2")

        plt.scatter(cx, cy, c= 'g', s= 100, marker = '^', label = "Centroid 3")

        plt.scatter([M1[0],M2[0],M3[0]], [M1[1],M2[1],M3[1]], s = 200, marker = '*', c = 'orange', label = 'Real centers')

        plt.title("Data after 3 Means clustering")

        plt.legend()



    

    # plotting a line of the centroids paths

    if plotting : 

        axxx = plt.plot(axs, ays, c = 'r')

        axxx = plt.plot(bxs, bys, c = 'b')

        axxx = plt.plot(cxs, cys, c = 'g')

        

        # plotting the accuracy calculated in the scores list

        d = plt.figure()

        plt.plot(scores, 'o-', label = "accuracy")

        plt.plot(iterations, [np.max(scores)]*len(iterations), label = "max score")

        plt.title("score")

        plt.xlabel("iterations")

        plt.ylabel("score")

        plt.legend( )



        plt.show()

        

        #print("The maximum score is = ", np.max(scores))

        #print("The score at convergence is = ", scores[-1])  # those two scores are not always equal

        

    return scores[-1]

        

def test_score(dt):

    # the score is calculated by cumulating the sum of a boolean mask generated by the test dt['L_true'] == dt['L_pred']

    # It reflects how well the predicted class matches with the real data labels after preprocessing the labels 

    return ((dt['L_true'] == dt['L_pred']).cumsum()/dt.shape[0])[dt.shape[0]-1]





def labelize(dt):

    # this function helps us recognize and match predicted classes with their adequate label

    # it is using an iterative principle, we try all possibilities ( 3² = 9 in case k = 3)

    # then we choose the couples (class, label) giving the maximum possible accuracy

    # it finally returns a dictionnary { key : class, value : label }

    s_labels = ['class_A', 'class_B', 'class_C']

    i_labels = [1, 2, 3]

    pred_labels = []

    for s in s_labels :

        c_scores = []

        for i in i_labels : 

            c = np.max((dt[dt['L_pred'] == s]['L_true'] == i).cumsum())

            c_scores.append(c)

        pred_labels.append(1 + np.argmax(np.array(c_scores)))

    

    return dict(zip(s_labels,pred_labels))



def switch_labels(dt):

    # this function is acting on the dataframe using the results of the "labelize" function 

    # it applies the the modifications from 'labelize' dictionnary on the 'L_pred' series in the data

    dt['L_pred'] = dt['L_pred'].map(labelize(dt))
k_means(20, 5, plotting = True)
rdm = [] # here we store random states

sc = []  # here we store accuracies

    

for i in range(0, 20):

    rdm.append(i)

    sc.append(k_means(20, i))



plt.xlabel("Random State values")

plt.ylabel("Accuracy value")

plt.plot(rdm, sc, 'o-')
from sklearn.cluster import KMeans # importing the model



# we must convert the data into a set of lists of each point coordinates so we can fit it to th model ( [[x1,y1],.....] )

D = []

for x, y in zip(dt['X'], dt['Y']):

    D.append([x, y])

    

D = np.array(D)

print(len(D))
model = KMeans(n_clusters= 3, random_state=5)

model.fit(D)

model.predict(D)

plt.scatter(D[:,0], D[:,1], c = model.predict(D))
df = pd.DataFrame()



df['X'] = D[:, 0]

df['Y'] = D[:, 1]

df['L'] = model.predict(D)



dic = { 1 : 1, 

        0 : 2,

        2 : 3}

df['L'] = df['L'].map(dic)

model_score = ((df['L'] == dt['L_true']).cumsum()/df.shape[0])[df.shape[0]-1] 

print("Kmeans built in has a score of ", model_score)  

# the same method of evaluation is done for the sklearn kmeans model predictions



compare_score = ((df['L'] == dt['L_pred']).cumsum()/df.shape[0])[df.shape[0]-1] 

print("The rate of harmony between our model and the sklean's is ", compare_score)  

# the same method to compare the built in model and our iterative one