import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import random

import pylab

from string import punctuation, digits

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

pylab.rc('figure', figsize=(10,7))
def perceptron_single_step_update(feature_vector,label,current_theta,current_theta_0):

    

    if(np.dot(label,(np.dot(current_theta,feature_vector.transpose())+current_theta_0))<=0):

        

        current_theta=current_theta+np.dot(label,feature_vector)

        current_theta_0=current_theta_0+label

        

    return (current_theta,current_theta_0)
def perceptron(feature_matrix, labels, T):



    update=(np.zeros(feature_matrix.shape[1]),0)

    for t in range(T):

        for i in range(feature_matrix.shape[0]):



            update=perceptron_single_step_update(feature_matrix[i],labels[i],update[0],update[1])



    return(update)
def hinge_loss_single(feature_vector, label, theta, theta_0):



    y = np.dot(theta, feature_vector) + theta_0

    loss = max(0.0, 1 - y * label)

    return loss

    raise NotImplementedError

    

def hinge_loss_full(feature_matrix, labels, theta, theta_0):



    loss=0

    for vec, y in zip(feature_matrix, labels):

        loss=loss+hinge_loss_single(vec,y,theta,theta_0)

    

    return (loss/len(labels))

 
def plot_classifier(theta,theta_0): # to visualise our model

    good=toy_data[0]==1

    plt.scatter(toy_data[1][good],toy_data[2][good],c='blue',alpha=0.8,label='Classified True')

    plt.scatter(toy_data[1][~good],toy_data[2][~good],c='r',alpha=0.8,label='Classified False')



    xmin,xmax=plt.axis()[:2]

    

    x=np.linspace(xmin,xmax)

    y = -(theta[0]*x + theta_0) / (theta[1] + 1e-16)

    plt.plot(x,y,label='Classifier',lw=2)

    plt.xlabel("Values of x1")

    plt.ylabel("Values of x2")

    plt.title("Linear classifier for a 2-d feature vector")

    plt.legend()
toy_data=np.loadtxt("/kaggle/input/toy_data.tsv",delimiter='\t',unpack=True)

toy_label,toy_feature= toy_data[0],toy_data[[1,2]]
theta,theta_0=perceptron(toy_feature.transpose(),toy_label,15)

plot_classifier(theta,theta_0)

print(f"Average hinge loss for Perceptron algorithm is {hinge_loss_full(toy_feature.transpose(),toy_label,theta,theta_0):.4f}")
def get_order(n_samples):

    try:

        with open(str(n_samples) + '.txt') as fp:

            line = fp.readline()

            return list(map(int, line.split(',')))

    except FileNotFoundError:

        random.seed(1)

        indices = list(range(n_samples))

        random.shuffle(indices)

        return indices
def perceptron_stochaistic(feature_matrix, labels, T):



    update=(np.zeros(feature_matrix.shape[1]),0)

    for t in range(T):

        for i in get_order(feature_matrix.shape[0]):

            # 

            update=perceptron_single_step_update(feature_matrix[i],labels[i],update[0],update[1])



    return(update)
theta,theta_0=perceptron_stochaistic(toy_feature.transpose(),toy_label,15)

plot_classifier(theta,theta_0)

print(f"Average hinge loss for Perceptron algorithm is {hinge_loss_full(toy_feature.transpose(),toy_label,theta,theta_0):.4f}")
def average_perceptron(feature_matrix, labels, T):



    

    update=(np.zeros(feature_matrix.shape[1]),0) 

    theta_sum=np.array(update[0])

    theta_0_sum=update[1]

    count=0

    for t in range(T):

        for i in get_order(feature_matrix.shape[0]):

            update=perceptron_single_step_update(feature_matrix[i],labels[i],update[0],update[1])

            theta_sum=theta_sum+np.array(update[0])

            theta_0_sum=theta_0_sum+update[1]

            count=count+1

            

    avg_theta=theta_sum/count

    avg_theta_0=theta_0_sum/count

    

    return (avg_theta,avg_theta_0)
theta,theta_0=average_perceptron(toy_feature.transpose(),toy_label,15)

plot_classifier(theta,theta_0)

print(f"Average hinge loss for Average Perceptron algorithm is {hinge_loss_full(toy_feature.transpose(),toy_label,theta,theta_0):.4f}")
def pegasos_single_step_update(feature_vector,label,L,eta,current_theta,current_theta_0):

    

    if(np.dot(label,(np.dot(current_theta,feature_vector.transpose())+current_theta_0))<=1):

        current_theta=current_theta*(1-eta*L)+eta*np.dot(label,feature_vector)

        current_theta_0=current_theta_0+eta*label

    else:

        current_theta=current_theta*(1-eta*L)

    

    return (current_theta,current_theta_0)





def pegasos(feature_matrix, labels, T, L):



    update=(np.zeros(feature_matrix.shape[1]),0) #why not just theta and theta zero

    count=0

    for t in range(T):

        

        for i in get_order(feature_matrix.shape[0]):

            count=count+1

            eta=1/np.sqrt(count)

            

            update=pegasos_single_step_update(feature_matrix[i],labels[i],L,eta,update[0],update[1])        

    return update
theta,theta_0=pegasos(toy_feature.transpose(),toy_label,T=15,L=0.1)

plot_classifier(theta,theta_0)

print(f"Average hinge loss for Pegasos algorithm is {hinge_loss_full(toy_feature.transpose(),toy_label,theta,theta_0):.4f}")
train_data=pd.read_csv("/kaggle/input/reviews_train.tsv",delimiter='\t',engine ='python')

validation_data=pd.read_csv("/kaggle/input/reviews_val.tsv",delimiter='\t',engine ='python')

test_data=pd.read_csv("/kaggle/input/reviews_test.tsv",delimiter='\t',engine ='python')
train_data.head()
train_text=train_data['text'].values

train_label=train_data['sentiment'].values



val_text=validation_data['text'].values

val_label=validation_data['sentiment'].values



test_text=test_data['text'].values

test_label=test_data['sentiment'].values
def words_corpus(text_array):

    count=0

    dictionary={}

    for text in text_array:

        for c in punctuation + digits:

            text = text.replace(c, ' ' + c + ' ')

            

        

        for word in text.lower().split():

            if word not in dictionary.keys():

                dictionary[word]=len(dictionary)

    return dictionary
def text_to_feature(text_feature,dic):

    feature_matrix=np.zeros([len(text_feature),len(dic)])

   

    for i,text in enumerate(text_feature):

        for c in punctuation + digits:

            text = text.replace(c, ' ' + c + ' ')

        for word in text.lower().split():

            

            if word in dic:   ##For loop

                feature_matrix[i,dic[word]]=1

        

    return feature_matrix
dictionary=words_corpus(train_text)
train_feature_matrix=text_to_feature(train_text,dictionary)

val_feature_matrix=text_to_feature(val_text,dictionary)

test_feature_matrix=text_to_feature(test_text,dictionary)



print(f"Size of Training data: {train_feature_matrix.shape}")

print(f"Size of Validation data: {val_feature_matrix.shape}")

print(f"Size of Test data: {test_feature_matrix.shape}")
def classify(feature_matrix, theta, theta_0):

    

    predictions=[]

    for i in range(len(feature_matrix)):

        

       if(np.dot(theta,feature_matrix[i].transpose())+theta_0)>0:

           

           predictions.append(1)

       else:

           predictions.append(-1)

           

    return np.array(predictions)

def accuracy(preds, targets):



    return (preds == targets).mean()
def classifier_accuracy(classifier,train_feature_matrix,val_feature_matrix,train_labels,val_labels,**kwargs):



    thetas=classifier(train_feature_matrix,train_labels,**kwargs)

        

    predictions_train=classify(train_feature_matrix,thetas[0],thetas[1])

    predictions_val=classify(val_feature_matrix,thetas[0],thetas[1])

    

    train_acc=accuracy(predictions_train,train_labels)

    val_acc=accuracy(predictions_val,val_labels)

    

    return (train_acc,val_acc)
train_accuracy_perc, val_accuracy_perc = classifier_accuracy(perceptron_stochaistic,train_feature_matrix,val_feature_matrix,train_label,val_label,T=15)

print("{:35} {:.4f}".format("Training accuracy for perceptron:", train_accuracy_perc))

print("{:35} {:.4f}".format("Validation accuracy for perceptron:", val_accuracy_perc))
train_accuracy_avper, val_accuracy_avper = classifier_accuracy(average_perceptron,train_feature_matrix,val_feature_matrix,train_label,val_label,T=15)

print("{:35} {:.4f}".format("Training accuracy for Average perceptron:", train_accuracy_avper))

print("{:35} {:.4f}".format("Validation accuracy for Average perceptron:", val_accuracy_avper))
train_accuracy_peag, val_accuracy_peag = classifier_accuracy(pegasos,train_feature_matrix,val_feature_matrix,train_label,val_label,T=15,L=0.01)

print("{:35}{:.4f}".format("Training accuracy for Pegasos:", train_accuracy_peag))

print("{:35} {:.4f}".format("Validation accuracy for Pegasos:", val_accuracy_peag))
def tune_param (train_fn,param_vals,train_feature,train_label,val_feature,val_label,**kwargs):

    train_accuracy=[]

    val_accuracy=[]

    

    for t in param_vals:

        theta,theta_0=train_fn(train_feature,train_label,t,**kwargs)

        train_predict=classify(train_feature,theta,theta_0)

        t_acc=accuracy(train_predict,train_label)

        train_accuracy.append(t_acc)

        

        val_predict=classify(val_feature,theta,theta_0)

        v_acc=accuracy(val_predict,val_label)

        val_accuracy.append(v_acc)

        print(f"For T = {t} ========== Training Accuracy = {t_acc:.4f} ========== Validation Accuracy = {v_acc:.4f}")

        

    return train_accuracy,val_accuracy
def plot_accuracy(algo_name, param_name, param_vals, acc_train, acc_val):

   

    

    plt.subplots()

    plt.plot(param_vals, acc_train, '-o')

    plt.plot(param_vals, acc_val, '-o')



  

    algo_name = ' '.join((word.capitalize() for word in algo_name.split(' ')))

    param_name = param_name.capitalize()

    plt.suptitle('Classification Accuracy vs {} ({})'.format(param_name, algo_name))

    plt.legend(['train','val'], loc='upper right', title='Partition')

    plt.xlabel(param_name)

    #plt.ylim(0.5,1)

    plt.ylabel('Accuracy (%)')

    plt.show()
data = (train_feature_matrix, train_label, val_feature_matrix, val_label)

Ts = [5,10,15,20,25,30,40,50]
train_accuracy_per, val_accuracy_per=tune_param(perceptron_stochaistic,Ts,*data)

print(f"The best Parameter for Perceptron algorithm is T = {Ts[val_accuracy_per.index(max(val_accuracy_per))]} and maximum accuracy reached is {max(val_accuracy_per):.4f}")
plot_accuracy("Perceptron","T",Ts,train_accuracy_per,val_accuracy_per)
train_accuracy_avgp, val_accuracy_avgp=tune_param(average_perceptron,Ts,*data)

print(f"The best Parameter for Average Perceptron algorithm is T = {Ts[val_accuracy_avgp.index(max(val_accuracy_avgp))]} and maximum accuracy reached is {max(val_accuracy_avgp):.4f}")
plot_accuracy("Average Perceptron","T",Ts,train_accuracy_avgp,val_accuracy_avgp)
train_accuracy_peg, val_accuracy_peg=tune_param(pegasos,Ts,*data,L=0.01)

print(f"\nThe best Parameter for Pegasos algorithm is T = {Ts[val_accuracy_peg.index(max(val_accuracy_peg))]} and maximum accuracy reached is {max(val_accuracy_peg):.4f}")
plot_accuracy("Pegasos","T",Ts,train_accuracy_peg,val_accuracy_peg)
parameters=pegasos(train_feature_matrix,train_label,T=25,L=0.01)

pred=classify(test_feature_matrix,parameters[0],parameters[1])

acc=accuracy(pred,test_label)



print(f"The accuracy of the best fitted algorithm for Test data is : {acc:.4f}")
results=pd.DataFrame(pred,columns=['Predicted_Labels'])

results.to_csv("submission.csv", index=False)