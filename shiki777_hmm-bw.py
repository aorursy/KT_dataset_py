import numpy as np

import time
class HMM:

    def __init__(self,A,B,Pi,O):

        self.A = A

        self.B = B

        self.Pi = Pi

        self.O = O

        self.N = self.A.shape[0]

        self.M = self.B.shape[1]

        self.T = self.O.shape[0]

    

    def forward(self,):

        alpha = np.zeros((self.T,self.N))

        for i in range(self.N):

            alpha[0,i] = self.Pi[i] * self.B[i,self.O[0]]

        for t in range(self.T - 1):

            for i in range(self.N):

                alpha[t + 1,i] = self.B[i,self.O[t + 1]] * np.dot(alpha[t],self.A[:,i])

        Polambda = np.sum(alpha[self.T - 1])

        return alpha,Polambda

    

    def backward(self,):

        beta = np.zeros((self.T,self.N))

        for i in range(self.N):

            beta[self.T - 1,i] = 1

        for t in range(self.T - 2,-1,-1):

            for i in range(self.N):

                beta[t,i] = np.dot(self.A[i]*self.B[:,self.O[t + 1]],beta[t + 1])

        Polambda = np.dot(self.B[:,self.O[0]] * beta[0],self.Pi)

        return beta,Polambda

    

    def cal_gamma(self,alpha,beta):

        gamma = np.zeros((self.T,self.N))

        for t in range(self.T):

            for i in range(self.N):

                gamma[t,i] = alpha[t,i] * beta[t,i] / np.dot(alpha[t],beta[t])

        return gamma

    

    def cal_xi(self,alpha,beta):

        xi = np.zeros((self.T - 1,self.N,self.N))

        for t in range(self.T - 1):

            for i in range(self.N):

                for j in range(self.N):

                    numerator = alpha[t,i] * self.A[i,j] * self.B[j,self.O[t+1]] * beta[t+1,j]

                    denominator = sum( sum(     

                        alpha[t,i1] * self.A[i1,j1] * self.B[j1,self.O[t+1]] * beta[t+1,j1] 

                        for j1 in range(self.N) )   # the second sum

                        for i1 in range(self.N) )    # the first sum

                    xi[t,i,j] = numerator / denominator

        return xi

    

    def Baum_Welch(self,):

        

        self.A = np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])

        self.B = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])

        self.Pi = np.array([0.2,0.4,0.4])

        

        

#         self.A = np.array([[1/3,1/3,1/3],[1/3,1/3,1/3],[1/3,1/3,1/3]])

#         self.B = np.array([[0.5,0.5],[0.5,0.5],[0.5,0.5]])

#         self.Pi = np.array([1/3,1/3,1/3])

        print(self.A)

        print(self.B)

        print(self.Pi)

        V = [k for k in range(self.M)]

        times = 0

        while times <= 500:

            alpha,Polambda = self.forward()

            beta,Polambda = self.backward()

            gamma = self.cal_gamma(alpha,beta)

            xi = self.cal_xi(alpha,beta)

            

            for i in range(self.N):

                for j in range(self.N):

                    numerator = sum(xi[t,i,j] for t in range(self.T-1))

                    denominator = sum(gamma[t,i] for t in range(self.T-1))

                    self.A[i, j] = numerator / denominator

                    

            for j in range(self.N):

                sum_ = np.sum(gamma[:,j])

                for k in range(self.M):

                    numerator = sum(gamma[t,j] for t in range(self.T) if self.O[t] == V[k]) 

                    self.B[j,k] = numerator / sum_

            

            for i in range(self.N):

                self.Pi[i] = gamma[0,i]

            times += 1

        print(self.A)

        print(self.B)

        print(self.Pi)

        return self.A,self.B,self.Pi

    

    def viterbi(self,):

        delta = np.zeros((self.T,self.N))

        psi = np.zeros((self.T,self.N))

        I = np.zeros(self.T,np.int)

        for i in range(self.N):

            delta[0,i] = self.Pi[i] * self.B[i,self.O[0]]

        for t in range(1,self.T):

            for i in range(self.N):

                delta[t,i] = (self.B[i,self.O[t]] * delta[t - 1] * self.A[:,i]).max()

                psi[t,i] = (delta[t - 1] * self.A[:,i]).argmax()

        P_T = delta[self.T - 1].max()

        I[self.T - 1] = delta[self.T - 1].argmax()

        for t in range(self.T - 2,-1,-1):

            I[t] = psi[t + 1,I[t + 1]]

        return I
A = np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])

B = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])

Pi = np.array([0.2,0.4,0.4])

O = np.array([0,1,0])
model = HMM(A,B,Pi,O)
a = model.forward()

a
b = model.backward()

b
v = model.viterbi()

v
model.Baum_Welch()