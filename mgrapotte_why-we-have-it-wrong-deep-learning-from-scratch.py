# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def sigmoid(x):

        y =1/(1 + exp(-x))

        return y

    

#Todo : ReLu and friends definition
def RMSE(x,y):

        z = (x-y)**2

        return z



#Todo : friends definition
class Perceptron():



    #La classe perceptron correspond à la définition d'un perceptron d'un réseau neuronal profond.



    def __init__(self,function,entry=3,bias=0.):

        self.function = function #fonction de normalisation

        self.entry = entry #taille du vecteur d'entrée

        self.weights =np.linspace(0.,1.,entry) #initialisation des poids

        self.bias = bias #initialisation du biais



    def set_function(self,function):

        #met à jour la fonction de normalisation

        self.function = function



    def set_weights(self,weights):

        #met à jour les poids

        self.weights=weights



    def set_bias(self,biais):

        #met à jour le biais

        self.biais = biais





    def compute(self,input):

    #Compute calcul la sortie d'un perceptron.

        output = 0

        #Erreur si la dimension du vecteur en entrée ne correspond pas

        if len(input) != len(self.weights):

            raise NameError('La dimension du vecteur en entrée ne correspond pas avec la dimension prédéfinie pour le perceptron.')



        #Calcul FeedForward

        for i in range(len(input)):

            output = output+self.weights[i]*input[i]

        output = output+self.bias

        output=self.function(output)

        return output
class Couche():



    #La classe couche correspond à la définition d'une couche composée de plusieurs neurones.



    def __init__(self,nb_neurones,function,entry):

        #initialise les perceptrons de la couche

        neurones = [Perceptron(function,entry)]

        for i in range(nb_neurones-1):

            neurones.append(Perceptron(function,entry))

        self.neurones = neurones

        self.nb_neurones = nb_neurones



    def compute_couche(self,input):

        #calcul la sortie d'une couche

        result = np.zeros(self.nb_neurones)

        for i in range(self.nb_neurones):

            result[i] = self.neurones[i].compute(input)

        return result
class Reseau():



    #La classe Reseau correspond à l'ensemble du réseau de neurones "fully connected"



    def __init__(self,input_size,function,cost_function,nb_couches=3

            ,couche_size=3,output_size=2):



        self.input_size = input_size



        #initialise les couches du réseau

        couches = [Couche(couche_size,function,entry = self.input_size)]

        for i in range(nb_couches-1):

            couches.append(Couche(couche_size,function,entry=couche_size))



        #initialisation de la fonction coût

        self.cost_function = cost_function



        #ajout de la couche de sortie

        couches.append(Couche(output_size,function,entry=couche_size))



        #fin de l'initialisation des couches du réseau

        self.couches = couches



    def feed_forward(self,entree):



        #Cette fonction calcule la sortie du réseau de neurones



        if len(entree) != self.input_size:

            raise NameError("l'entrée n'est pas de la bonne taille")

        output = np.copy(entree)

        for i in range(len(self.couches)):

            output = self.couches[i].compute_couche(output)



        return output

    

    #todo : define the training function with the backprop. algorithm.