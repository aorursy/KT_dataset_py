#Sample exercise on multivariate linear regression



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#First import the file from the specified directory. Do not forget to use r'filename' when file is not in the cw folder

def import_file():

    data=pd.read_csv('../input/cracow-apartments-data-set/cracow_apartments.csv')

    m=len(data)

    return data,m



#Get all the data

def clean(data):

    y, X1=data['price'].to_numpy(), data['size'].to_numpy()

    X2, X3=data['distance_to_city_center'].to_numpy(), data['rooms'].to_numpy()

    return y,X1,X2,X3



#Get the cost value. Important to check the cost plot

def cost(prediction, y, m):

    vals=[(Ho-Y)**2 for Ho,Y in zip(prediction,y)]

    cost_value=sum(vals)/(2*m)

    return cost_value



#Calculate the predicted value given the parameters

def predict(w1,w2,w3,b,X1,X2,X3):

    prediction=np.array([w1*x+b for x in X1])+np.array([w2*x for x in X2])+np.array([w3*x for x in X3])

    return prediction



#Train our model to get the correct parameters

def train(X1,X2,X3,y,m,iterations,learning_rate=0.0002):

    accumulated_cost=[]

    w1,w2,w3,b=0,0,0,0

    

    for i in range(iterations):

        final_predictions=predict(w1,w2,w3,b,X1,X2,X3)

        w1-=learning_rate*(sum(np.multiply((final_predictions - y),X1)))/m

        w2-=learning_rate*(sum(np.multiply((final_predictions - y),X2)))/m

        w3-=learning_rate*(sum(np.multiply((final_predictions - y),X3)))/m

        b-=learning_rate*(sum(final_predictions - y))/m

        accumulated_cost.append(cost(final_predictions,y,m))

    print('The function is price={}*size + {}*distance + {}*rooms + {} at {} iterations'.format(np.round(w1,2),np.round(w2,2),np.round(w3,2),np.round(b,2),iterations))

    return final_predictions,accumulated_cost



#Visualize the results

def visualize(X1,y,final_predictions,iterations,accumulated_cost):

    #compare actual data to predicted data

    fig1,(ax1,ax2)=plt.subplots(1,2,sharey=True)

    fig1.suptitle('Housing prices prediction')

    ax1.scatter(X1,y)

    ax1.set_title('Actual data')

    ax1.set_xlabel('Combined Factors')

    ax1.set_ylabel('Price')

    ax2.scatter(X1,final_predictions)

    ax2.set_title('Predicted data')

    ax2.set_xlabel('Combined Factors')

    plt.show()

    

    #check the cost graph

    plt.figure(2)

    plt.plot(range(iterations), accumulated_cost)

    plt.title('Cost Function')

    plt.xlabel('Iterations')

    plt.ylabel('Cost')

    plt.xlim((0,len(X1)))

    plt.show()



#Run the program. You can specify the number of iterations

def program(iterations=1000):

    data,m=import_file()

    y,X1,X2,X3=clean(data)

    final_predictions,accumulated_cost=train(X1,X2,X3,y,m,iterations)

    visualize(X1,y,final_predictions,iterations,accumulated_cost)

    

#Execute

program()

    