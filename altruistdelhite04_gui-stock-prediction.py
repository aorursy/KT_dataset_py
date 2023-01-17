import pandas as pd

import os



os.listdir('../input')



Stock_Market = pd.read_csv('../input/Stock market data.csv')

df = pd.DataFrame(Stock_Market,columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])

print (df)
from pandas import DataFrame

import matplotlib.pyplot as plt

plt.scatter(df['Interest_Rate'], df['Stock_Index_Price'], color='red')

plt.title('Stock Index Price Vs Interest Rate', fontsize=14)

plt.xlabel('Interest Rate', fontsize=14)

plt.ylabel('Stock Index Price', fontsize=14)

plt.grid(True)

plt.show()

 

plt.scatter(df['Unemployment_Rate'], df['Stock_Index_Price'], color='green')

plt.title('Stock Index Price Vs Unemployment Rate', fontsize=14)

plt.xlabel('Unemployment Rate', fontsize=14)

plt.ylabel('Stock Index Price', fontsize=14)

plt.grid(True)

plt.show()  
from sklearn import linear_model

import statsmodels.api as sm

X = df[['Interest_Rate','Unemployment_Rate']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets

Y = df['Stock_Index_Price']

 

# with sklearn

regr = linear_model.LinearRegression()

regr.fit(X, Y)



print('Intercept: \n', regr.intercept_)

print('Coefficients: \n', regr.coef_)





# prediction with sklearn

New_Interest_Rate = 2.75

New_Unemployment_Rate = 5.3

print ('Predicted Stock Index Price: \n', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))



# with statsmodels

X = sm.add_constant(X) # adding a constant

 

model = sm.OLS(Y, X).fit()

predictions = model.predict(X) 

 

print_model = model.summary()

print(print_model)

import tkinter as tk 

import statsmodels.api as sm

X = df[['Interest_Rate','Unemployment_Rate']] # here we have 2 input variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets

Y = df['Stock_Index_Price'] # output variable (what we are trying to predict)



# with sklearn

regr = linear_model.LinearRegression()

regr.fit(X, Y)



print('Intercept: \n', regr.intercept_)

print('Coefficients: \n', regr.coef_)





# with statsmodels

X = sm.add_constant(X) # adding a constant

 

model = sm.OLS(Y, X).fit()

predictions = model.predict(X) 

 





# tkinter GUI

root= tk.Tk() 

 

canvas1 = tk.Canvas(root, width = 1200, height = 450)

canvas1.pack()



# with sklearn

Intercept_result = ('Intercept: ', regr.intercept_)

label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')

canvas1.create_window(260, 220, window=label_Intercept)



# with sklearn

Coefficients_result  = ('Coefficients: ', regr.coef_)

label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')

canvas1.create_window(260, 240, window=label_Coefficients)



# with statsmodels

print_model = model.summary()

label_model = tk.Label(root, text=print_model, justify = 'center', relief = 'solid', bg='LightSkyBlue1')

canvas1.create_window(800, 220, window=label_model)





# New_Interest_Rate label and input box

label1 = tk.Label(root, text='Type Interest Rate: ')

canvas1.create_window(100, 100, window=label1)



entry1 = tk.Entry (root) # create 1st entry box

canvas1.create_window(270, 100, window=entry1)



# New_Unemployment_Rate label and input box

label2 = tk.Label(root, text=' Type Unemployment Rate: ')

canvas1.create_window(120, 120, window=label2)



entry2 = tk.Entry (root) # create 2nd entry box

canvas1.create_window(270, 120, window=entry2)





def values(): 

    global New_Interest_Rate #our 1st input variable

    New_Interest_Rate = float(entry1.get()) 

    

    global New_Unemployment_Rate #our 2nd input variable

    New_Unemployment_Rate = float(entry2.get()) 

    

    Prediction_result  = ('Predicted Stock Index Price: ', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))

    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')

    canvas1.create_window(260, 280, window=label_Prediction)

    

button1 = tk.Button (root, text='Predict Stock Index Price',command=values, bg='orange') # button to call the 'values' command above 

canvas1.create_window(270, 150, window=button1)

 



root.mainloop()
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

X = df[['Interest_Rate','Unemployment_Rate']].astype(float) # here we have 2 input variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets

Y = df['Stock_Index_Price'].astype(float) # output variable (what we are trying to predict)



# with sklearn

regr = linear_model.LinearRegression()

regr.fit(X, Y)



print('Intercept: \n', regr.intercept_)

print('Coefficients: \n', regr.coef_)





# tkinter GUI

root= tk.Tk()



canvas1 = tk.Canvas(root, width = 500, height = 300)

canvas1.pack()



# with sklearn

Intercept_result = ('Intercept: ', regr.intercept_)

label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')

canvas1.create_window(260, 220, window=label_Intercept)



# with sklearn

Coefficients_result  = ('Coefficients: ', regr.coef_)

label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')

canvas1.create_window(260, 240, window=label_Coefficients)





# New_Interest_Rate label and input box

label1 = tk.Label(root, text='Type Interest Rate: ')

canvas1.create_window(100, 100, window=label1)



entry1 = tk.Entry (root) # create 1st entry box

canvas1.create_window(270, 100, window=entry1)



# New_Unemployment_Rate label and input box

label2 = tk.Label(root, text=' Type Unemployment Rate: ')

canvas1.create_window(120, 120, window=label2)



entry2 = tk.Entry (root) # create 2nd entry box

canvas1.create_window(270, 120, window=entry2)





def values(): 

    global New_Interest_Rate #our 1st input variable

    New_Interest_Rate = float(entry1.get()) 

    

    global New_Unemployment_Rate #our 2nd input variable

    New_Unemployment_Rate = float(entry2.get()) 

    

    Prediction_result  = ('Predicted Stock Index Price: ', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))

    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')

    canvas1.create_window(260, 280, window=label_Prediction)

    

button1 = tk.Button (root, text='Predict Stock Index Price',command=values, bg='orange') # button to call the 'values' command above 

canvas1.create_window(270, 150, window=button1)

 



#plot 1st scatter 

figure3 = plt.Figure(figsize=(5,4), dpi=100)

ax3 = figure3.add_subplot(111)

ax3.scatter(df['Interest_Rate'].astype(float),df['Stock_Index_Price'].astype(float), color = 'r')

scatter3 = FigureCanvasTkAgg(figure3, root) 

scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)

ax3.legend() 

ax3.set_xlabel('Interest Rate')

ax3.set_title('Interest Rate Vs. Stock Index Price')



#plot 2nd scatter 

figure4 = plt.Figure(figsize=(5,4), dpi=100)

ax4 = figure4.add_subplot(111)

ax4.scatter(df['Unemployment_Rate'].astype(float),df['Stock_Index_Price'].astype(float), color = 'g')

scatter4 = FigureCanvasTkAgg(figure4, root) 

scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)

ax4.legend() 

ax4.set_xlabel('Unemployment_Rate')

ax4.set_title('Unemployment_Rate Vs. Stock Index Price')



root.mainloop()