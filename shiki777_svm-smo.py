#My github: https://github.com/Sh1k17
import numpy as np

import time

import math

import random
def loadData(filename):

    fr = open(filename,'r')

    x,y = [],[]

    for line in fr.readlines():

        curline = line.strip().split(',')

        if int(curline[0]) in [0,1]:

            x.append([int(num) / 255 for num in curline[1:]])

            if int(curline[0]) == 0:

                y.append(1)

            else:

                y.append(-1)

    x = np.array(x)

    y = np.array(y)

    return x,y
class SVM:

    def __init__(self,x_train,y_train,x_val,y_val):

        self.x_train = x_train[:5000]

        self.y_train = y_train[:5000]

        self.x_val = x_val

        self.y_val = y_val

        self.m,n = self.x_train.shape

        self.sigma = 10

        self.k_matrix = self.cal_kernel()

        self.alpha = np.zeros(self.m)

        self.C = 200

        self.b = 0 

        self.E = np.zeros(self.m)

        self.support_vector_index = []

        

        

    def cal_kernel(self,):

        k = np.zeros((self.m,self.m))

        for i in range(self.m):

            x = self.x_train[i,:]

            for j in range(i,self.m):

                z = self.x_train[j,:]

                ans = np.linalg.norm(x - z,ord=2)

                ans = np.exp(-1 * ans / (2 * self.sigma**2))

            k[i,j] = ans

            k[j,i] = ans

        return k

    def satisfy_conditions(self,idx):

        g = self.get_G_Function_i(idx)

        y = self.y_train[idx]

        if self.alpha[idx] > self.C or self.alpha[idx] < 0: return False

        if self.alpha[idx] == 0: return y * g >= 1

        if self.alpha[idx] == self.C: return y * g <= 1

        if 0 < self.alpha[idx] < self.C: return y * g == 1

        

    def get_G_Function_i(self,idx):

        sum_ = 0

        for i in range(self.m):

            sum_ += self.y_train[i] * self.alpha[i] * self.k_matrix[i,idx]

        return sum_ + self.b

    

    def get_first_alpha(self,):

        for i in range(self.m):

            if 0 < self.alpha[i] < self.C and not self.satisfy_conditions(i):

                return i

        for i in range(self.m):

            if not self.satisfy_conditions(i):

                return i

        

    def get_second_alpha(self,idx):

        for i in range(self.m):

            self.E[i] = self.get_E_function_i(i)

        idx_second = -1

        ans = float('-inf')

        for i in range(self.m):

            if ans < np.abs(self.E[i] - self.E[idx]):

                idx_second = i

                ans = np.abs(self.E[i] - self.E[idx])

        return idx_second

    

    def get_E_function_i(self,idx):

        g = self.get_G_Function_i(idx)

        return g - self.y_train[idx]

    

    def fit(self,epochs):

        for epoch in range(epochs):

            start = time.time()

            idx_1 = self.get_first_alpha()

            idx_2 = self.get_second_alpha(idx_1)

            alpha_1 = self.alpha[idx_1]

            alpha_2 = self.alpha[idx_2]

            y_1 = self.y_train[idx_1]

            y_2 = self.y_train[idx_2]

            if y_1 == y_2:

                L = max(0,alpha_2 + alpha_1 - self.C)

                H = min(self.C,alpha_1 + alpha_2)

            else:

                L = max(0,alpha_2 - alpha_1)

                H = min(self.C,self.C + alpha_2 - alpha_1)

            

            

            K11 = self.k_matrix[idx_1,idx_1]

            K12 = self.k_matrix[idx_1,idx_2]

            K22 = self.k_matrix[idx_2,idx_2]

            yita = K11 + K22 - 2 * K12

            

            alpha_2_new_unc = alpha_2 + y_2 * (self.E[idx_1] - self.E[idx_2]) / yita

            if alpha_2_new_unc > H: alpha_2_new = H

            elif  alpha_2_new_unc < L: alpha_2_new = L

            else: alpha_2_new = alpha_2_new_unc    

            alpha_1_new = alpha_1 + y_1 * y_2 * (alpha_2 - alpha_2_new)

            b_1_new = -self.E[idx_1] - y_1 * K11 * (alpha_1_new - alpha_1) - y_2 * K12 *(alpha_2_new - alpha_2) + self.b

            b_2_new = -self.E[idx_2] - y_1 * K12 * (alpha_1_new - alpha_1) - y_2 * K22 *(alpha_2_new - alpha_2) + self.b

            

            if 0 < alpha_1_new < self.C: b_new = b_1_new

            elif 0 < alpha_2_new < self.C: b_new = b_2_new

            else: b_new = (b_1_new + b_2_new) / 2

            self.alpha[idx_1] = alpha_1_new

            self.alpha[idx_2] = alpha_2_new

            self.b = b_new

            self.E[idx_1] = self.get_E_function_i(idx_1)

            self.E[idx_2] = self.get_E_function_i(idx_2)

            print("Epoch {} costs {:.2f} seconds.".format(epoch,time.time() - start))

            

        for i in range(self.m):

            if self.alpha[i] > 0:

                self.support_vector_index.append(i)

    def cal_single_kernel(self,x,z):

        ans = np.linalg.norm(x - z,ord=2)

        ans = np.exp(-1 * ans / (2 * self.sigma**2))

        return ans



    def predict(self,x):

        result = 0

        for i in self.support_vector_index:

            tmp = self.cal_single_kernel(self.x_train[i,:],x)

            result += tmp * self.alpha[i] * self.y_train[i]

        result += self.b

        return np.sign(result)



    def test(self,x_val,y_val):

        correct = 0

        for i in range(x_val.shape[0]):

            y_pred = self.predict(x_val[i])

            if y_pred == y_val[i]: correct += 1

        return correct / x_val.shape[0]
x_train,y_train = loadData('/kaggle/input/mnist-percetron/mnist_train.csv')

x_val,y_val = loadData('//kaggle/input/mnist-percetron/mnist_test.csv')
model = SVM(x_train,y_train,x_val,y_val)

model.fit(1)
acc_train = model.test(model.x_train,model.y_train)

print("The accuracy of train dataset is {:.4f}".format(acc_train))
acc_val = model.test(x_val,y_val)

print("The accuracy of val dataset is {:.4f}".format(acc_val))