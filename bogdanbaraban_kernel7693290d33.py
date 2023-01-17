import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.cluster import AgglomerativeClustering,KMeans,SpectralClustering

from sklearn.metrics import mean_absolute_error,mean_squared_error

from scipy import stats

from collections import OrderedDict

%matplotlib inline
train = pd.read_csv('../input/train.csv')

train_x,test_x,train_y, test_y = train_test_split(train['GrLivArea'],train.SalePrice,test_size=0.2, shuffle=False)
#visualization 

plt.figure(figsize=(12,6))

plt.plot(train_x,train_y,'ro',label='train points')

plt.plot(test_x,test_y,'go',label='test points')

plt.legend();
class KNN_regression(object):

    """

    KNN for regression

    

    Arguments:

    ---------

    k -- number of nearest neighbours

    X -- input matrix

    """

    def euclidean_dist(self,x,y):

        """

        Euclidean distance between points x and y

        """

        if type(x)==np.float64:

            return (((y-x)**2)**0.5)

        return sum((y[i]-x[i])**2 for i in range(len(x)))**0.5

    

    def manhattan_dist(self,x,y):

        """

        Manhattan distance between points x and y

        """

        if type(x)==np.float64:

            return (abs(x-y))

        return sum(abs(x[i]-y[i]) for i in range(len(x)))



    

    def __init__(self,k=None, X_train=None, X_test = None, Y_train = None, Y_test = None, window_width = None, inverted_weights= False):

        self.X_train, self.X_test  = X_train, X_test

        self.Y_train, self.Y_test  = Y_train, Y_test

        self.inverted_weights = inverted_weights

        if k is not None:

            self.k = k

            self.parzen = False

        else:#using Parzen window  

            self.w = window_width

            self.parzen = True

      

        

    def get_Neighbours(self,point):

        """

        Arguments:

            point -- chosen point from X_test

        Returns:

            list of k closest neighbours [[distance,index],...]

        """

        distances = {self.euclidean_dist(self.X_train[i],point):i for i in range(len(self.X_train))}

        distances = OrderedDict(sorted(distances.items()))

        if self.parzen:

            i = sum(1 if d<=self.w else 0 for d in list(distances.values()))

            return list(distances.items())[:i]

        else:

            k_nearest = list(distances.items())[:self.k]

            return (k_nearest)



        

    def get_Weights(self,neighbours):

        """

        Arguments:

        ---------

        neighbours -- list of k closest neighbours [[distance,index],...]

        Returns:

        ---------

        list of inverted weights - highest score for the closest point

        """

        distances = [row[1] for row in neighbours]

        inverted_distances = [sum(distances)/(d+1) for d in distances]

        return [d/sum(inverted_distances) for d in inverted_distances]

    

    def predict(self):

        prediction = []

        for point in self.X_test:

            #get neighbours

            neighbours = self.get_Neighbours(point)

            #calculate weight based on proximity of each point

            if self.inverted_weights:

                weights = self.get_Weights(neighbours)

                #predict price based on their values

                prediction.append(self.predict_weighted_average(neighbours,weights))

            else:

                prediction.append(self.predict_average(neighbours))

            

        self.pred = np.array(prediction)

        return self.pred

    

    def predict_average(self, neighbours):

        """

        Arguments:

            neighbours -- list of k closest neighbours [[distance,index],...]

        Returns:

            prediction based on average

        """

        indices = [row[1] for row in neighbours]

        values = self.Y_train[indices]

        return np.mean(values)

    

    def predict_weighted_average(self,neighbours,weights):

        """

        Arguments:

            neighbours -- list of k closest neighbours [[distance,index],...]

            weights -- list of inverted weights

        Returns:

            prediction based on weighted average

        """

        indices = [row[1] for row in neighbours]

        values = self.Y_train[indices]

        

        return np.dot(values,weights)



    

    def calculate_error(self,param="squared"):

        if param=="squared":

            return mean_squared_error(self.Y_test,self.pred)

        else:

            return mean_absolute_error(self.Y_test,self.pred)

    

    

    def set_width(self, w):

        self.w = w

    

    def set_k(self,k):

        self.k = k

        

    def get_X_test(self):

        return self.X_test

    def get_pred(self):

        return self.pred
model = KNN_regression(10,X_train=train_x,X_test=test_x,Y_train=train_y,Y_test=test_y)

pr = model.predict()



print("KNN without inverted weights")

print(model.calculate_error("squared"))

print(model.calculate_error("absolute"))



#visualize(X_tr,Y_tr,X_t,Y_t,list(pr),True)

#visualize(X_tr,Y_tr,X_t,Y_t,list(pr))
#whts = model.get_Weights(nbrs)

#indices = [row[1] for row in nbrs]

#m = X_tr[indices][:,2]

#np.dot(m,whts)





model = KNN_regression(10,X_train=X_tr,X_test=X_t,Y_train=Y_tr,Y_test=Y_t,inverted_weights=True)

pr = model.predict()



print("Using inverted weights")

print(model.calculate_error("squared"))

print(model.calculate_error("absolute"))

visualize(X_tr,Y_tr,X_t,Y_t,list(pr),True)

visualize(X_tr,Y_tr,X_t,Y_t,list(pr))
model = KNN_regression(X_train=X_tr,X_test=X_t,Y_train=Y_tr,Y_test = Y_t, window_width=5.0)

pr = model.predict()

print("Parzen regression")

print(model.calculate_error())

print(model.calculate_error("absolute"))

visualize(X_tr,Y_tr,X_t,Y_t,list(pr),True)

visualize(X_tr,Y_tr,X_t,Y_t,list(pr))
accuracy_mse_knn, accuracy_mse_knn_w,accuracy_mse_parzen = [],[],[]

model = KNN_regression(1,X_train=X_tr,X_test=X_t,Y_train=Y_tr,Y_test=Y_t)

for i in range(1,20):

    model.set_k(i)

    pr = model.predict()

    accuracy_mse_knn.append(model.calculate_error('absolute'))



    

model = KNN_regression(1,X_train=X_tr,X_test=X_t,Y_train=Y_tr,Y_test=Y_t,inverted_weights=True)

for i in range(1,20):

    model.set_k(i)

    pr = model.predict()

    accuracy_mse_knn_w.append(model.calculate_error('absolute'))





model = KNN_regression(X_train=X_tr,X_test=X_t,Y_train=Y_tr,Y_test = Y_t, window_width=1.0)

for i in range(5,20):

    model.set_width(float(i))

    pr = model.predict()

    accuracy_mse_parzen.append(model.calculate_error('absolute'))
plt.figure(figsize=(15,5))



plt.subplot(131)

plt.title("KNN MSE")

plt.plot(accuracy_mse_knn)



plt.subplot(132)

plt.title("Weighted KNN MSE")

plt.plot(accuracy_mse_knn_w)



plt.subplot(133)

plt.title("Parzen MSE")

plt.plot(accuracy_mse_parzen)



plt.suptitle("Accuracy comparison")

plt.show()
class Kernel:

    """

    Nadaraya-Watson weighted average regressor using kernel functions

    """

    def euclidean_dist(self,x,y):

        """

        Euclidean distance between points x and y

        """

        if type(x)==np.float64:

            return (((y-x)**2)**0.5)#1 dim

        return sum((y[i]-x[i])**2 for i in range(len(x)))**0.5 #2 dim



    def __init__(self,X_train,Y_train,X_test,Y_test, params, lam = 0.2):

        

        self.X_train, self.X_test = X_train, X_test

        self.Y_train, self.Y_test = Y_train, Y_test

        self.lam = lam

        if params:

            self.params = params

        else:

            print("wrong set of parameters")

        

    def kernel(self,x0,x1):

        """

        Arguments:

        ----------

        x0 -- test point

        x1 -- train point

        Returns:

        -------

        value of Kernel function for a specific pair and parameter

        """

        u = round(self.euclidean_dist(x0,x1),5)/self.lam

        self.u.append(u)

       # print(u,x0,x1)

        if u>1:

            return 0

        else:

            if self.current_param == "parabolic": #Epanenchikov

                return 0.75 * (1-u**2)

            

            elif self.current_param == "quartic":#Біквадратне

                return 15*((1-u**2)**2)/16



            elif self.current_param == "gaussian":

                return np.exp(-0.5 * u**2) / ((2 * np.pi)**0.5)

        

            elif self.current_param == "cosine":#Косинусоидальное

                return np.pi/4 * np.cos(u*np.pi/2)

            

            elif self.current_param == "triangle":

                return 1-abs(u)

            

            elif self.current_param == "uniform":

                return 0.5

            return 0.5  

        

    def predict(self,param):

        self.u = []

        

        if param.lower() in self.params:

            self.current_param = param

        else:

            self.current_param = param[0].lower()

        

        prediction = []

        

        for point in self.X_test:

            K =[self.kernel(x,point) for x in self.X_train]#values of a kernel function

            prediction.append(sum(k*y for k,y in zip(K,self.Y_train))/sum(K))                      

        self.pred = prediction

        

        return self.pred

        

    

    def calculate_error(self,param="squared"):

        if param=="squared":

            return mean_squared_error(self.Y_test,self.pred)

        else:

            return mean_absolute_error(self.Y_test,self.pred)

        

    def setParam(self,param):

        self.current_param=param

        

    def set_lam(self,l):

        self.lam=l

    

    def getDistances(self,point):

        return [round(self.euclidean_dist(x,point),5) for x in self.X_train]

    def getU(self):

        return self.u
parameters = ["parabolic","quartic","gaussian"]

k = Kernel(train_x,train_y,test_x,test_y,parameters,lam=5)

p = k.predict("gaussian")



print("Nadaraya-Watson Kernel regressor")

print(k.calculate_error("squared"))

print(k.calculate_error("absolute"))



#visualize(X_tr,Y_tr,X_t,Y_t,list(p),True)
model = Kernel(X_tr,Y_tr,X_t,Y_t,parameters,lam=2)

def acc_NW(model,m,n,kernel="parabolic",error="absolute"):

    """

        m,n - range for lambda

    """

    acc = []

    for i in range(m,n):

        model.set_lam(i)

        pr = model.predict(kernel)

        acc.append(model.calculate_error(error))

    return acc



accuracy_nadarayaW_Epanenchikov = acc_NW(model,4,25,"parabolic") 

accuracy_nadarayaW_Gaussian = acc_NW(model,4,25,"gaussian")

accuracy_nadarayaW_quartic = acc_NW(model,4,25,"quartic")

plt.figure(figsize=(15,5))



plt.subplot(131)

plt.title("Epanenchikov kernel")

plt.plot(accuracy_nadarayaW_Epanenchikov)



plt.subplot(132)

plt.title("Gaussian kernel")

plt.plot(accuracy_nadarayaW_Gaussian)



plt.subplot(133)

plt.title("Quartic kernel")

plt.plot(accuracy_nadarayaW_quartic)



plt.suptitle("Nadaraya-Watson regression accuracy comparison")

plt.show()
class LOWESS(Kernel):

    """

    

    """

    

    def __init__(self,X_train,Y_train,X_test,Y_test,params):

        

        super().__init__(X_train,Y_train,X_test,Y_test,params)

        self.coeffs = np.ones(self.X_train.shape)

        self.N = len(X_train)

    

    

    def smooth(self,it,param):

        self.u = []

        if param in self.params:

            self.setParam(param)

        else:

            self.setParam(self.params[0])

            

        for i in range(it):

            self.A = [sum(self.coeffs[i]*self.Y_train[i]*self.kernel(self.X_train[i],self.X_train[j]) for i in range(self.N) if i!=j)/sum(self.coeffs[i]*self.kernel(self.X_train[i],self.X_train[j]) for i in range(self.N)) for j in range(self.N)]

            self.coeffs = np.array([self.kernel(self.A[i],self.Y_train[i]) for i in range(self.N)])

        return "Success"

    

    

    def calculate_accuracy(self,param="squared"):

        if param=="squared":

            return mean_squared_error(self.X_test[:,2],self.pred)

        else:

            return mean_absolute_error(self.X_test[:,2],self.pred)

l = LOWESS(X_tr,Y_tr,X_t,Y_t,parameters)
l.smooth(3,"uniform")