import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import urllib.request

import category_encoders as ce

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.preprocessing import OrdinalEncoder

#Function for reading in the Fishing Data:

#Output is a panda df

def dtDataImport(url):

    rawData = urllib.request.urlopen(url)

    rawLines = rawData.readlines() 

    numClass = int(rawLines[0])

    classList = rawLines[1].decode().strip().split(",") 

    numVar = int(rawLines[2]) 

    attribs = [attrib[0] for attrib in [elem.strip().decode().split(",") for elem in rawLines[3:numVar+3]]]+['Class'] 

    df = pd.DataFrame([elem.strip().decode().split(",") for elem in rawLines[numVar+4:]], columns=attribs) 

    return(df) 
fishing = dtDataImport('https://cis.gvsu.edu/~wolffe/courses/cs678/projects/fishing.data')

#print(fishing.head())
#Function of encoding the data



def p4encoder(df):

    ce_ord = ce.OrdinalEncoder(cols = list(df.columns))

    df = ce_ord.fit_transform(df)

    return(df.to_numpy()-1)



encode_fishdf = p4encoder(fishing)



print(encode_fishdf)
#split np array to features and class

def splitter(array):

    X, y = array[:, :-1], array[:, -1]

    y = y.reshape(len(y),1)

    return(X,y)



fishFeats, fishClss = splitter(encode_fishdf)



#print(fishFeats)

#print(fishClss)

#Generating weights and bias based on the array



def wtsBias(array):

    weights = np.random.rand(len(array[0,]),1) #generate a list of numbers between 0-1 the same length as your first row

    bias = np.random.rand(1) #generate anumber between 0-1

    return(weights,bias)



fishWts, fishBias = wtsBias(fishFeats)



#print(fishWts)

#print(fishBias)
#Calculating Sigmoid

def sigmoidCalc(alph):

    return(1 / (1+np.exp(-alph)))
#Derivitive of sigmoid

def derSigmoid(x):

    return sigmoidCalc(x)*(1-sigmoidCalc(x))
#Making the feed forward step

def feedForward(feats,wts,bias):

    return(sigmoidCalc(np.dot(feats,wts)+bias))



fishYs = feedForward(fishFeats,fishWts,fishBias)

print(fishYs)
#Obtain error from front



def backPropSum(Ys,Clss):

    error = Ys-Clss

    return(error.sum(),error)

    

fishErrorSum,fishErrors = backPropSum(fishYs,fishClss)

print(fishErrorSum)

print(fishErrors)
#function for comparing a dataset with the built weights on the NN



def predictNN(Feats, Clss, wts, bias):

    preds=[]

    for row in Feats:

        preds.append(feedForward(row,wts,bias))

    np.concatenate(preds).ravel()

    

    desPreds=[]

    for i in preds:

        if i>0.5:

            desPreds.append('yes')

        else:

            desPreds.append('no')

    desPreds = np.array(desPreds)

    

    desClss=[]

    for p in Clss:

        if p>0.5:

            desClss.append('yes')

        else:

            desClss.append('no')

    desClss = np.array(desClss)

    

    cm = pd.crosstab(desPreds, desClss, rownames = ['Actual'], colnames = ['Predicted'], margins=True)

    cs = classification_report(desPreds, desClss,output_dict=True).get('accuracy')

    

    return([cm,cs])
#Function for bulk NN with no hidden layer



def NNnoHLBulk(array, lr, epochs):

    

    Feats, Clss = splitter(array)

    

    wts, bias = wtsBias(Feats)

    

    for epoch in range(epochs):

        inputs = Feats

        

        z = feedForward(Feats,wts,bias)

        

        sumError, error = backPropSum(z,Clss)

        #print(sumError)

        

        dcost_dpred = error

        dpred_dz = derSigmoid(z)

        

        z_delta = dcost_dpred * dpred_dz

        

        inputs = Feats.T

        wts -= lr * np.dot(inputs,z_delta)

        

        for i in z_delta:

            bias -= lr * i

            #print("bias:",bias)

    

    return(predictNN(Feats,Clss,wts,bias))

    

fishres = NNnoHLBulk(encode_fishdf,0.005,1000)



print(fishres[0])

print(fishres[1])
#Function to return a plot of the accuracy of several Neural Nets



EPrnge = np.arange(5,500,5)

LRrnge = np.arange(0.0001,3,0.1)



def plotNNs(df, Eprng, Lrrng, Lr = 0.03, epoch = 500):

    

    accsEpoch = []

    accsLr = []

    

    for i in Eprng:

        accsEpoch.append(NNnoHLBulk(df, Lr, i)[1])

    

    plt.plot(Eprng,accsEpoch)

    plt.title('Epochs')

    plt.ylabel('Accuracy')

    plt.show()

    

    for i in Lrrng:

        accsLr.append(NNnoHLBulk(df, i, epoch)[1])

    

    plt.plot(Lrrng,accsLr)

    plt.title('Learning Rates')

    plt.ylabel('Accuracy')

    plt.show()



plotNNs(encode_fishdf, EPrnge, LRrnge)    
#Function for creating a single hidden layer of weights



def wtsBiasHL(array,hlNodeRatio):

    outWeights = np.random.rand(len(array[0,]),1) #generate a list of numbers between 0-1 the same length as the array

    hlBias = np.random.rand(1) #generate a number between 0-1

    hlWeights = np.random.rand(len(array[0,])*int(round(len(outWeights)*hlNodeRatio,1)),1).reshape(int(round(len(outWeights)*hlNodeRatio,1)),int(round(len(outWeights),1))) #generate a list of numbers between 0-1 the same length as your first row

    inputBias = np.random.rand(len(array[0,]),1) #generate a list of weights between the bias and the hiddlen layers

    return(hlWeights,inputBias,outWeights,hlBias)



fishhlWts, fishInBias, fishOutwts, fishhlBias= wtsBiasHL(fishFeats, 1)



print("Input to HL weights",fishhlWts,"\n\nInputBias", fishInBias,"\n\nWeights from hidden layer wts",fishOutwts, "\n\nHiddenLayer Bias", fishhlBias)
#feedforward for hidden layer fish



#4 inputs

#16 weights

#4 bias



#output is 4HL node values



def feedForwardHL(nodes,wts,bias):

    return(sigmoidCalc(np.dot(nodes,wts)+bias.T))



def feedForward(feats,wts,bias):

    return(sigmoidCalc(np.dot(feats,wts)+bias))



def outputErrorCalc(Clss, Ys):

    error = (Clss-Ys)*Ys*(1-Ys)

    return(error)



def hidLayerErrorCalc(hiddenLayerNode, outPutError, hlWeight):

    error = hiddenLayerNode*(1-hiddenLayerNode)*(hlWeight.T*outPutError)

    return(error)



FFhlfish = feedForwardHL(fishFeats,fishhlWts,fishInBias)

print("HiddenLayerNodeValues:",FFhlfish)



FFoutfish = feedForward(FFhlfish, fishOutwts, fishhlBias)

print("Output node value of ys(14X1:",FFoutfish)



outErrorfish = outputErrorCalc(fishClss,FFoutfish)

print("\noutputError or Ys error(14X1):",outErrorfish)



hidErrorfish = hidLayerErrorCalc(FFhlfish,outErrorfish,fishOutwts)

print("\nHiddenLayerErrors(14X4):",hidErrorfish)

#Function for bulk NN with a single hidden layer



#This Function works up until one needs to update the weights.  The error is that the weights are not organized in a matrix that is compatible with the "outWts" line.



def NNBulk(array, lr, epochs, hiddenLayerRatio):

    

    Feats, Clss = splitter(array)

    

    hlWts, inputBias, outWts, hlBias = wtsBiasHL(Feats,hiddenLayerRatio)  

    

    

    for epoch in range(epochs):

        

        #FeedForward_____________________________________________

        hidLayerNode = feedForwardHL(Feats,hlWts,inputBias)

        

        outNode = feedForward(hidLayerNode,outWts,hlBias)

        

        #FeedBack output________________________________________________

        

        outError = outputErrorCalc(Clss,outNode) #(Clss-Ys)*Ys*(1-Ys)

                

        totError = (0.5*np.power((Clss-outNode),2)).sum()

        print("totError:",totError)

            

            

        hidError = hidLayerErrorCalc(hidLayerNode, outError, outWts)

        

        

        

        #Update Weights__________________________________________________

        

        for i in range(len(fishing.index)-1):

        

            hlBias += lr * outError[i,] #Should be 1 x 1

        

        

            hlWts += lr * hidError[i,] * Feats[i,] #should be 4X4

        

        

            for j in range(len(fishing.columns)-2):

                

                outWts[j,] += lr * outError[i,] * hidLayerNode[i,j]

            

                inputBias[j,] += lr * hidError[i,j] #should be 4X1

        

        outNode = feedForward(hidLayerNode,outWts,hlBias)

        

        newError = (0.5*np.power((Clss-outNode),2)).sum()

        

        if round(newError, 5) == round(totError, 5):

            print('Convergence has been achieved at epoch: ', epoch)

            break;

    

   #print(predictNN(Feats,Clss,wts,bias))

    

fishres = NNBulk(encode_fishdf,0.1,10000,1)



print(fishres)

#Constructing the example from the HW



#THis has an error now that I've been mucking with some of the functions.



hwInputs = [0,1]

hwTarget = [1]

hwInBias = [1,1]

hwInwts = [[1,-1],[0.5,2]]

hwHlBias = [1]

hwHlwts = [1.5,-1]

lr = 0.5



#FeedForward__________________________

hwHlnodes = feedForward(hwInputs,hwInwts,hwInBias)

print("hwHlnodes",hwHlnodes)



yy = feedForward(hwHlnodes,hwHlwts,hwHlBias)

print("yy:",yy)



#BackPropagate____________________________



Eyfunc = outputErrorCalc(hwTarget,yy)

print("Eyfunc:",Eyfunc)



hwHlErrorfunc = hidLayerErrorCalc(hwHlnodes,Eyfunc,hwHlwts)

print("hwHlErrorfunc:",hwHlErrorfunc)



#UpdateWeights________________________________



hwHlwts += lr*Eyfunc*hwHlnodes

print("hwHlwts:",hwHlwts)



hwHlBias += lr*Eyfunc

print("HiddenLayerBias:",hwHlBias)



hwInwts += lr*hwHlErrorfunc*hwInputs

print("hwInwts:",hwInwts)



hwInBias += lr*hwHlErrorfunc

print("InputBias:",hwInBias)
#I've left this as its a nice function that has to do with how you import data into the NB



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#These functions I have kept around for now but I'm not really using.



#Back Propagation for a hidden layer



def outputErrorCalc(Clss, Ys):

    error = (Clss-Ys)*Ys*(1-Ys)

    return(error)



def hidLayerErrorCalc(hiddenLayerNode, outPutError, hlWeight):

    error = hiddenLayerNode*(1-hiddenLayerNode)*(hlWeight*outPutError)

    return(error)



def feedForward(nodes,wts,bias):

    return(sigmoidCalc(np.dot(nodes,wts)+bias))





fishYs = feedForward(fishFeats,fishOutwts,fishhlBias)

print(fishYs)