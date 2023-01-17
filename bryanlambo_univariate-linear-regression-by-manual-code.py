import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



def import_file():

    data=pd.read_csv('../input/cracow-apartments-data-set/cracow_apartments.csv')

    m=len(data)

    return data,m



#Let us get only the price and size for now

def clean(data):

    y, X=data['price'].to_numpy(), data['size'].to_numpy()

    return y,X



#Get the cost value. Important to check the cost plot

def cost(prediction, y, m):

    vals=[(Ho-Y)**2 for Ho,Y in zip(prediction,y)]

    cost_value=sum(vals)/(2*m)

    return cost_value



#Calculate the predicted value given the parameters

def predict(w0,b,X):

    prediction=[w0*x+b for x in X]

    prediction=np.array(prediction)

    return prediction



#Train our model to get the correct parameters

def train(X,y,m,iterations,learning_rate=0.0002):

    accumulated_cost=[]

    w0,b=0,0

    

    for i in range(iterations):

        final_predictions=predict(w0,b,X)

        w0-=learning_rate*(sum(np.multiply((final_predictions - y),X)))/m

        b-=learning_rate*(sum(final_predictions - y))/m

        accumulated_cost.append(cost(final_predictions,y,m))

    print('The function is price={}*size + {} at {} iterations'.format(np.round(w0,2),np.round(b,2),iterations))

    return final_predictions,accumulated_cost



#Visualize the results

def visualize(X,y,final_predictions,iterations,accumulated_cost):

    #compare actual data to predicted data

    fig1,(ax1,ax2)=plt.subplots(1,2,sharey=True)

    fig1.suptitle('Housing prices prediction')

    ax1.scatter(X,y)

    ax1.set_title('Actual data')

    ax1.set_xlabel('Size of apartment')

    ax1.set_ylabel('Price')

    ax2.scatter(X,final_predictions)

    ax2.set_title('Predicted data')

    ax2.set_xlabel('Size of apartment')

    plt.show()

    

    #check the cost graph

    plt.figure(2)

    plt.plot(range(iterations), accumulated_cost)

    plt.title('Cost Function')

    plt.xlabel('Iterations')

    plt.ylabel('Cost')

    plt.xlim((0,len(X)))

    plt.show()



#Run the program. You can specify the number of iterations

def program(iterations=1000):

    data,m=import_file()

    y,X=clean(data)

    final_predictions,accumulated_cost=train(X,y,m,iterations)

    visualize(X,y,final_predictions,iterations,accumulated_cost)



#Execute

program()