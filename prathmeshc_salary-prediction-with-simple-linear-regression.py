import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from ipywidgets import interact

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/salary-dataset/Data_salary.csv') #read_csv() is used to read the csv file

df.head() #head() in pandas displays first five rows of the dataset by default
df.shape #By shape attribute we can check the shape of the dataset
df.isnull().sum() #Let us check for any NaN values in the dataset
df = df.rename(columns={'YearsExperience' : 'Experience'}) #Rename the column names in the dataset

df.head()
#Initialize the independent and dependent variables



inp = df.iloc[:, :1]  

outp = df.Salary.values
#Train the model



trainer = LinearRegression()

trainer.fit(inp, outp)
# 'm' and 'c' are the 'slope' and 'intercept' of the best fit line



m = trainer.coef_

c = trainer.intercept_

m,c
# Let us now do the prediction and store the predicted values in an array.



prediction = trainer.predict(inp)
prediction
# Plot the best fit line on the graph



plt.scatter(inp, outp) #plots the scatter plot 

plt.plot(inp, prediction, 'g') #plots the best fit line 
#Let us now see the accuracy of the model we just trained.



print(r2_score(outp, prediction))
def salary_predict(Experience):

    salary = trainer.predict([[Experience]])

    print("Salary should be : ",salary[0])
#Lets create an user-friendly scroller to predict the salary

#Note : If the scroller doesn't appear, try running the code in jupyter notebook



interact(salary_predict, Experience=(0,50))