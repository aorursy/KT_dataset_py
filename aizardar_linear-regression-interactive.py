import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rcParams

from sklearn.model_selection import train_test_split

rcParams['figure.figsize'] = (10, 5)   # Change this if figures look ugly. from matplotlib import rcParams

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline







# IPython libraries



from ipywidgets import interactive

from IPython.display import display

import ipywidgets as widgets
training_points = 250    #  Number of training points

noise = 0.1   # Noise level





x_min = -5

x_max = 5



def generate_poly_data(training_points,x_min,x_max,noise):

    x1 = np.linspace(x_min,x_max,training_points*5)

    x = np.random.choice(x1,size=training_points)

    y = np.sin(x) + noise*np.random.normal(size=training_points)

    plt.scatter(x,y,edgecolors='k',c='red',s=60)

    plt.grid(True)

    plt.show()

    return (x,y)





x,y = generate_poly_data(training_points,x_min,x_max,noise)
def func_fit(test_size,degree):

    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=2020)

    

    t1=np.min(X_test)

    t2=np.max(X_test)

    t3=np.min(y_test)

    t4=np.max(y_test)

    

    t5=np.min(X_train)

    t6=np.max(X_train)

    t7=np.min(y_train)

    t8=np.max(y_train)

    

    posx_test=t1+(t2-t1)*0.7

    posx_train=t5+(t6-t5)*0.7

    posy_test=t3+(t4-t3)*0.2

    posy_train=t7+(t8-t7)*0.2

    

    model = make_pipeline(PolynomialFeatures(degree,interaction_only=False), 

                          LinearRegression(normalize=True))

    

    X_train=X_train.reshape(-1,1)

    X_test=X_test.reshape(-1,1)

    

    model.fit(X_train,y_train)

    

    train_pred = np.array(model.predict(X_train))

    train_score = model.score(X_train,y_train)

    

    test_pred = np.array(model.predict(X_test))

    test_score = model.score(X_test,y_test)

    

    RMSE_test=np.sqrt(np.mean(np.square(test_pred-y_test)))

    RMSE_train=np.sqrt(np.mean(np.square(train_pred-y_train)))

    

    print("Test score: {}, Training score: {}".format(test_score,train_score))

    

    print("RMSE Test: {}, RMSE train: {}".format(RMSE_test,RMSE_train))

    

    plt.figure(figsize=(12,4))

    

    plt.subplot(1,2,1)

    plt.title("Test set performance\n",fontsize=16)

    plt.xlabel("X-test",fontsize=13)

    plt.ylabel("y-test",fontsize=13)

    plt.scatter(X_test,y_test,edgecolors='k',c='red',s=60)

    plt.scatter(X_test,test_pred,edgecolors='k',c='yellow',s=60)

    plt.grid(True)

    plt.legend(['Actual test data','Predicted values'])

    plt.text(x=posx_test,y=posy_test,s='Test score: %.3f'%(test_score),fontsize=15)

    

    plt.subplot(1,2,2)

    plt.title("Training set performance\n",fontsize=16)

    plt.xlabel("X-train",fontsize=13)

    plt.ylabel("y-train",fontsize=13)

    plt.scatter(X_train,y_train,c='red')

    plt.scatter(X_train,train_pred,c='yellow')

    plt.grid(True)

    plt.legend(['Actual training data','Fitted values'])

    plt.text(x=posx_train,y=posy_train,s='Training score: %.3f'%(train_score),fontsize=15)

    

    plt.show()

       

    return (train_score,test_score)
style = {'description_width': 'initial'}

# Continuous_update = False for IntSlider control to stop continuous model evaluation while the slider is being dragged

m = interactive(func_fit,test_size=widgets.Dropdown(options={"10% data":0.1,"20% data":0.2, "30% data":0.3,

                                                    "40% data":0.4,"50% data":0.5},

                                          description="Test set size",style=style),

               degree=widgets.IntSlider(min=1,max=10,step=1,description= 'Polynomial degree',

                                       stye=style,continuous_update=False))



# Set the height of the control.children[-1] so that the output does not jump and flicker

output = m.children[-1]

output.layout.height = '350px'



# Display the control

display(m)