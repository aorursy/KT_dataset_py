# https://github.com/Sh1k17
import numpy as np

import time

import random
class EM:

    def __init__(self,dataset_size,epochs,mu0 = -2,sigma0 = 0.5,alpha0 = 0.3,mu1 = 0.5,sigma1 = 1,alpha1 = 0.7):

        self.mu0 = mu0

        self.sigma0 = sigma0

        self.alpha0 = alpha0

        self.mu1 = mu1

        self.sigma1 = sigma1

        self.alpha1 = alpha1

        self.dataset_size = dataset_size

        self.epochs = epochs

        self.dataset = self.loadData(mu0 = mu0,sigma0 = sigma0,alpha0 = alpha0,mu1 = mu1,sigma1 = sigma1,alpha1 = alpha1,\

                                     dataset_size= self.dataset_size)

    

    def loadData(self,mu0,sigma0,alpha0,mu1,sigma1,alpha1,dataset_size):

        raw_dataset_1 = np.random.normal(mu0,sigma0,int(dataset_size * alpha0))

        raw_dataset_2 = np.random.normal(mu1,sigma1,int(dataset_size * alpha1))

        dataset = np.append(raw_dataset_1,raw_dataset_2)

        np.random.shuffle(dataset)

        return dataset

    

    def get_gamma_j_k(self,mu,sigma,dataset,alpha):

        return 1 / (np.sqrt(np.pi * 2) * sigma) * np.exp(-1 * np.square(dataset - mu) / 2 / sigma ** 2)

    

    def E_step(self,mu0,sigma0,alpha0,mu1,sigma1,alpha1):

        gamma_1_old = alpha0 * self.get_gamma_j_k(mu0,sigma0,self.dataset,alpha0)

        gamma_2_old = alpha1 * self.get_gamma_j_k(mu1,sigma1,self.dataset,alpha1)

        

        gamma_1_new = gamma_1_old / (gamma_1_old + gamma_2_old)

        gamma_2_new = gamma_2_old / (gamma_1_old + gamma_2_old)

        return gamma_1_new,gamma_2_new

    

    def M_step(self,mu0,mu1,gamma_1,gamma_2):

        mu0_new = np.sum(np.dot(gamma_1,self.dataset)) / np.sum(gamma_1)

        mu1_new = np.sum(np.dot(gamma_2,self.dataset)) / np.sum(gamma_2)

        

        sigma0_new = np.sqrt(np.sum(np.dot(gamma_1,np.square(self.dataset - mu0))) / np.sum(gamma_1))

        sigma1_new = np.sqrt(np.sum(np.dot(gamma_2,np.square(self.dataset - mu1))) / np.sum(gamma_2))

        

        alpha0 = np.sum(gamma_1) / gamma_1.shape[0]

        alpha1 = np.sum(gamma_2) / gamma_2.shape[0]

        return mu0_new,sigma0_new,alpha0,mu1_new,sigma1_new,alpha1

    

    def fit(self,mu0,sigma0,alpha0,mu1,sigma1,alpha1):

        print("Initial parameters set:")

        print("mu0:{:.2f} sigma0:{:.2f} alpha0:{:.2f} mu1:{:.2f} sigma1:{:.2f} alpha1:{:.2f} \n".format( \

               mu0,sigma0,alpha0,mu1,sigma1,alpha1))

        print("Original parameters set:")

        print("mu0:{:.2f} sigma0:{:.2f} alpha0:{:.2f} mu1:{:.2f} sigma1:{:.2f} alpha1:{:.2f} \n".format( \

               self.mu0,self.sigma0,self.alpha0,self.mu1,self.sigma1,self.alpha1))



        for epoch in range(self.epochs):

            gamma_1,gamma_2 = self.E_step(mu0,sigma0,alpha0,mu1,sigma1,alpha1)

            mu0,sigma0,alpha0,mu1,sigma1,alpha1 = self.M_step(mu0,mu1,gamma_1,gamma_2)

        print("After training,the parameters set:")

        print("mu0:{:.2f} sigma0:{:.2f} alpha0:{:.2f} mu1:{:.2f} sigma1:{:.2f} alpha1:{:.2f}".format( \

               mu0,sigma0,alpha0,mu1,sigma1,alpha1))
model = EM(dataset_size=10000,epochs=10000)
model.fit(0,1,0.5,1,1,0.5)