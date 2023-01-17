import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import clear_output

%matplotlib inline

pd.set_option('display.float_format', lambda x: '%.5f' % x)
df = pd.DataFrame({"x":np.arange(101)}) # Creates data frame with x ranging from 0 to 100
df["y"] = df["x"] * 11 # adding the target feature y, which is x*11 which makes it perfect for linear model
arr = df.values

x = arr[:,0]

y = arr[:,1]
# Initializing parameters to 0

theta0 = 0 #intercept

theta1 = 0
learning_rate = 0.0001
loss_df = pd.DataFrame() # data frame to track the loss and parameter values for each iteration
same = 0 # variables that check if the loss is same as its previous and ends the loop if the loss value is same for 10 times

i=0

while same < 10:

    loss_df_0 = pd.DataFrame()

    loss = (np.mean(((theta0 + theta1 * x) - y) ** 2)) # Mean squred error

    loss_df_0 = pd.DataFrame({"iter":pd.Series(i),"t0":pd.Series(theta0),"t1":pd.Series(theta1),"loss":pd.Series(loss)})

    if i == 0:

        loss_df = loss_df.append(loss_df_0)

    t0 = theta0 - (learning_rate * np.mean((theta0 + theta1 * x) - y)) #updating theta0 or the intercept

    t1 = theta1 - (learning_rate * np.mean(((theta0 + theta1 * x) - y) * x)) #updating theta1

    theta0 = t0

    theta1 = t1

    if np.round(loss,5) == np.round(loss_df.iloc[-1,3],5): #incrementing variable 'same' if the loss is same as it's previous

        same += 1

    if i > 0:

        loss_df = loss_df.append(loss_df_0)

    i += 1

    print(loss_df.tail())

    clear_output(wait = True)
loss_df.plot(x="iter",y="loss") # plotting the loss function for all the iterations

plt.title("Loss Per Iteration")

plt.xlabel("Iteration")

plt.ylabel("Loss")

plt.show()
loss_df.tail(10) # inspecting last few rows of loss data frame
final_t0 = np.round(loss_df.iloc[-1,1],2) # storing theta0

final_t1 = np.round(loss_df.iloc[-1,2],2) # storing theta1



final_t0, final_t1
df["Pred"] = final_t0 + final_t1 * df["x"] #making predictions on training data (in sample)

df["y"].plot(kind="kde",legend=True)

df["Pred"].plot(kind="kde",legend=True)

plt.show()
#MSE

np.mean(((df["y"]-df["Pred"])**2))
np.random.seed(11)

df = pd.DataFrame({"x1":np.arange(10,111),"x2":np.arange(100,201),"x3":np.arange(200,301),"x4":np.arange(800,901),"x5":np.linspace(-200,200,101),

                  "x6":np.random.randint(100,10000,101)}) #generating dataset for the problem
# generating the target feature y using some random linear computations which makes it perfect for linear regression

df["y"] = (df["x1"] * 111) + (df["x2"] * 20) - (0.002 * df["x3"]) - (1.002 * df["x4"]) + (1.5 * df["x5"]) - (19.5 * df["x6"])
learning_rate = 0.5 # initializing the learning rate, alpha
df.head()
arr = df.values
X = arr[:,:-1].copy() # storing input features in X
#Feature scaling helps the algorith converge faster, hence sacling the input features within range [0,1]

for i in range(X.shape[1]):

    X[:,i] = X[:,i] / np.max(X[:,i])
all_ones = np.ones_like(arr[:,0]).reshape(-1,1)

X = np.append(all_ones,X,axis=1) # adding a vector of ones before the matrix X
X[:5,:] 
y = arr[:,-1].reshape(-1,1) # storing the target feature in y
theta = np.zeros_like(X[0,:]).reshape(-1,1) # initializing the theta vector to 0s
loss_df = pd.DataFrame() # data frame to track losses per iteration
same = 0 # variable that tracks if the loss is same as previous loss

i=0

while same < 10:

    loss_df_0 = pd.DataFrame()

    loss = np.mean((np.dot(X,theta)-y) ** 2) #mean squared error.

    err = np.dot(X,theta)-y #error vector of mx1 dim

    theta = theta - learning_rate * (1/X.shape[0])*(np.dot(err.T,X)).reshape(-1,1) #updating theta in a single step using matrix operations

    loss_df_0.insert(0,"iter",pd.Series(i))

    for j in range(len(theta)):

        loss_df_0.insert((j+1),("t"+str(j)),pd.Series(theta[j]))

    loss_df_0.insert((j+2),"loss",pd.Series(loss))

    

    

    if i == 0:

        loss_df = loss_df.append(loss_df_0)

    if np.round(loss,5) == np.round(loss_df.iloc[-1,-1],5):

        same += 1 #increments if loss is same as revious loss and breaks the loop once it's 10

    if i > 0:

        loss_df = loss_df.append(loss_df_0)

    i += 1

    print(loss_df.tail())

    clear_output(wait = True)
loss_df.plot(x="iter",y="loss") # plotting loss per iteration

plt.show()
theta # the final coefficients which minimize the loss
df["Pred"] = np.dot(X,theta) #adding predictions to data frame
df["y"].plot(kind="kde",legend=True)

df["Pred"].plot(kind="kde",legend=True)

plt.show()
#MSE

np.mean(((df["y"]-df["Pred"])**2))
loss_df_re = loss_df.melt(id_vars=["iter","loss"]).copy()

sns.relplot(kind="line",x="value",y="loss",data=loss_df_re,col="variable",col_wrap=4) # plotting loss per parameter

plt.show()
df = pd.DataFrame({"x":np.arange(101)}) # generating data for the problem
np.random.seed(11)

df["y"] = (df["x"] ** 2) + np.linspace(1000,10000,len(df["x"])) # adding y feature to make it a binomial function
sns.regplot(x="x",y="y",data=df,marker="+") # the plot shows a non-linear relationship b/w input and target features

plt.show()
df.insert(1,"x_sq",df["x"]**2) # since the relationship in binomial function, adding x^2  term to input features
df.head()
#feature scaling speeds up the execution

df["x"] = df["x"] / df["x"].max()

df["x_sq"] = df["x_sq"] / df["x_sq"].max()
arr = df.values
x = arr[:,0]

xsq = arr[:,1] # storing input features in variables
y = arr[:,2] # target feature
theta0 = 0

theta1 = 0

thetasq = 0 # initializing theta variables ton 0
learning_rate = 1.415
loss_df = pd.DataFrame()
same = 0

i=0

while same < 10:

    loss_df_0 = pd.DataFrame()

    loss = (np.mean(((theta0 + theta1 * x + thetasq * xsq) - y) ** 2))

    loss_df_0 = pd.DataFrame({"iter":pd.Series(i),"t0":pd.Series(theta0),"t1":pd.Series(theta1),"t_sq":pd.Series(thetasq),"loss":pd.Series(loss)})

    if i == 0:

        loss_df = loss_df.append(loss_df_0)

    t0 = theta0 - (learning_rate * np.mean((theta0 + theta1 * x + thetasq * xsq) - y))

    t1 = theta1 - (learning_rate * np.mean(((theta0 + theta1 * x + thetasq * xsq) - y) * x))

    tsq = thetasq - (learning_rate * np.mean(((theta0 + theta1 * x + thetasq * xsq) - y) * xsq))

    theta0 = t0

    theta1 = t1

    thetasq = tsq

    if np.round(loss,5) == np.round(loss_df.iloc[-1,4],5):

        same += 1

    if i > 0:

        loss_df = loss_df.append(loss_df_0)

    i += 1

    print(loss_df.tail())

    clear_output(wait = True)
loss_df.plot(x="iter",y="loss") #plotting loss vs iteration

plt.show()
final_t0 = np.round(loss_df.iloc[-1,1],2)

final_t1 = np.round(loss_df.iloc[-1,2],2)

final_tsq = np.round(loss_df.iloc[-1,3],2)

final_t0, final_t1, final_tsq
df["Pred"] = final_t0 + final_t1 * df["x"] + final_tsq * df["x_sq"] #adding prediction to data frame
df["y"].plot(kind="kde",legend=True)

df["Pred"].plot(kind="kde",legend=True)

plt.show()
df[["y","Pred"]].plot() #actual y vs prediction

plt.show()
#MSE

np.mean(((df["y"]-df["Pred"])**2))
loss_df_re = loss_df.melt(id_vars = ["iter","loss"]).copy()

sns.relplot(kind="line",x="value",y="loss",col="variable",data=loss_df_re) # loss per parameter

plt.show()
df = pd.DataFrame({"x1":np.arange(1,101)}) #generating dataset of marks
df["y"] = 0

df.loc[df["x1"]>=50,"y"]=1 # adding target feature showing pass or fail. 1 is pass and 0 is fail
learning_rate = 11
df.head()
arr = df.values

arr = np.array(arr,dtype=np.float64)
X = arr[:,:-1].copy() # storing input features in X
# Feature Scaling

for i in range(X.shape[1]):

    X[:,i] = X[:,i] / np.max(X[:,i])
all_ones = np.ones_like(arr[:,0]).reshape(-1,1)

X = np.append(all_ones,X,axis=1) # add ones before the matrix X
X[:5,:]
y = arr[:,-1].reshape(-1,1) # storing target feature in y
theta = np.zeros_like(X[0,:]).reshape(-1,1) # initializing theta to 0s
loss_df = pd.DataFrame() #dataframe to track loo per iteration
same = 0

i=0

while same < 100:

    loss_df_0 = pd.DataFrame()

    hx = 1 / (1+np.exp(-np.dot(X,theta))) # h(x)=1/(1+e^-(theta0*x0+theta1*x1...+thetan*xn))

    

    loss = np.mean((y*np.log(hx)) + ((1-y)*np.log(1-hx)))*-1 #-1/m((y1*log(h(x)1) + ((1-y1)*log(1-h(x)1)))))

    err = hx-y #error vector of mx1 dim

    theta = theta - learning_rate * (1/X.shape[0])*(np.dot(err.T,X)).reshape(-1,1) #same as in linear regression

    loss_df_0.insert(0,"iter",pd.Series(i))

    for j in range(len(theta)):

        loss_df_0.insert((j+1),("t"+str(j)),pd.Series(theta[j]))

    loss_df_0.insert((j+2),"loss",pd.Series(loss))



    

    if i == 0:

        loss_df = loss_df.append(loss_df_0)

    if np.round(loss,5) == np.round(loss_df.iloc[-1,-1],5):

        same += 1

    if i > 0:

        loss_df = loss_df.append(loss_df_0)

    i += 1

    print(loss_df.tail())

    clear_output(wait = True)
loss_df.plot(x="iter",y="loss") #plotting loss vs iteration

plt.show()
theta #final parameters
df["Pred"] = 1 / (1+np.exp(-np.dot(X,theta))) #adding predictions on training set using sigmoid function to dataframe
df[["y","Pred"]].plot(figsize=(10,10)) #plotting prediction vs actual y

plt.axhline(y=0.5,xmin=0,xmax=100,color="red")

plt.text(x=0.1,y=0.51,s="0.5")

plt.show()
df.loc[df["Pred"]<0.5,"Pred"]=0

df.loc[df["Pred"]>=0.5,"Pred"]=1 # h(x)>=0.5 means y=1 else y=0
df[["y","Pred"]].plot(figsize=(10,10)) # plotting prediction vs actual y after converting it into binary output

plt.show()
loss_df_re = loss_df.melt(id_vars=["iter","loss"]).copy()

sns.relplot(kind="line",x="value",y="loss",data=loss_df_re,col="variable") # plotting loss per parameter

plt.show()
pd.crosstab(index=df["Pred"],columns=df["y"],values=df["y"],aggfunc="count").fillna(0) # confusion matrix