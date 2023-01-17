#let's work on Boston dataset using Linear Regression and compare the scores
#when executed without any alterration (vs) by adding new dependant columns to the dataset
from sklearn import datasets
boston=datasets.load_boston()
print(type(boston)) #it is in bunch, we must convert it
boston
#we had things like 'data' and 'target' in that bunch. (target denotes output)
x=boston.data #we're setting names for those
y=boston.target #-----------------(x is input and y is output)-------------
print(x,"\n",y) #still the data is not clear to us
x.shape #let's look at shape
#it says 506 rows and 13 columns
#to see the data clearly, we use pandas
import pandas as pd
df=pd.DataFrame(x) #now, it's more clear
df
#in that bunch, we saw something named 'feature_names'  #we suppose it to be column name
print(boston.feature_names)
#now let's set those feature names to column names
df.columns=boston.feature_names
df
#invest much time on analysing the bunch and "describe" to get good knowledge aboout the data
#only then, you must go ahead with the rest
df.describe()
#we had a thing called DESCR in that bunch (always look for description in the bunch) and analyse it
boston.DESCR
#--------------------skip this cell now. Come back after going through the rest-------------------------
#--------------------------------multivariable and gradient descendaing-----------------------------------------
#-------altering the dataset by adding new dependant features and checking the score-----------
df["INDUS**2"]=df["INDUS"]**2
df["RM**2"]=df["RM"]**2
df["AGE**2"]=df["AGE"]**2
df["DIS**2"]=df["DIS"]**2
x1=df.values
from sklearn import model_selection
x1_train,x1_test=model_selection.train_test_split(x1,random_state=0) 
y1_train,y1_test=model_selection.train_test_split(y,random_state=0)
from sklearn.linear_model import LinearRegression
alg2=LinearRegression()
alg2.fit(x1_train,y1_train)
y1_pred=alg2.predict(x1_test)
score_test1=alg2.score(x1_test,y1_test) 
print(score_test1)
score_train1=alg2.score(x1_train,y1_train)
print(score_train1)
#as you see, the score is high here compared to the normal linear regression  score (at the last cell)
#----------------let's train it------------------
#we've to divide the data into training data and testing data (probably 75% - 25%)
from sklearn import model_selection
x_train,x_test=model_selection.train_test_split(x,random_state=0) #where x is a numpy array which we already stored above
#similarly for y
y_train,y_test=model_selection.train_test_split(y,random_state=0)
#can be done in same line --------#x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y)
#let's look at the sizes
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape) #it's divided into 75& - 25%
#now let's use linear regression algorithm to train the model
from sklearn.linear_model import LinearRegression #simple linear regression algo is already there in sklearn.linear.model 
#let's try it for our dataset (sklearn.linear.model is a submudule of sklearn)
alg1=LinearRegression() #we set alg1 to that algorithm
#now, let's train our data using that algo
alg1.fit(x_train,y_train) #it trains the data
#now, let's predict the test data using that algo
y_pred=alg1.predict(x_test) #it predicts the output for the test data and stores it in y_pred
y_pred 
#now, lets compare y_pred with y_test
#let's use a graph to compare
import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred) #use test in x-axis and pred in y-axis
plt.axis([0,40,0,40])
plt.grid()
plt.show() #if the line showed a perfect diagnol, then it means the precition is correct
#co.eff of determination
score_test=alg1.score(x_test,y_test) #test data's input and output
print(score_test)
score_train=alg1.score(x_train,y_train) #train data's input and output
print(score_train)
#now, let's compare these scores to those scores which we got by adding new features to the dataset
#you can find it in the 10th cell (which you skipped)