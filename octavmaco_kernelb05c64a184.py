# Importul bibliotecilor pe care le vom utiliza

import pandas as pd

from sklearn import linear_model

import tkinter as tk 

import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



PIB = {'An': [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019],

                'PIB': [178.8,193.1,206.2,222,247.1,265.7,296,324.7,358.4,339.8,334.3,348.1,358.9,377.6,396.2,416.4,442,481.5,514.5,545.5],

                'Rata_inflatiei': [45.7,34.5,22.2,15.3,11.9,9.0,6.6,4.8,7.8,5.6,6.1,5.8,3.3,4.0,1.1,-0.6,-1.6,1.3,1.3,1.3],

                'Rata_somajului': [7.6,7.3,8.3,7.8,8.0,7.1,7.2,6.3,5.5,6.3,7.0,7.2,6.8,7.1,6.8,6.8,5.9,5.0,4.0,3.4],

                'Datoria_publica': [17.6,16.1,16.0,14.8,10.5,8.0,3.8,5.1,8.1,15.4,22.9,27.3,28.9,29.5,29.7,29.7,27.9,28.3,28.3,28.3]        

                }



df = pd.DataFrame(PIB,columns=['An','PIB','Rata_inflatiei','Rata_somajului','Datoria_publica']) 



X = df[['Rata_inflatiei','Rata_somajului','Datoria_publica']].astype(float) # Variabilele predictive (independente)

Y = df['PIB'].astype(float) # variabila rezultat (dependentă)



# Utilizăm biblioteca scikit-learn

regr = linear_model.LinearRegression()

regr.fit(X, Y)



print('Termenul liber (interceptul): \n', regr.intercept_)

print('Coeficienții: \n', regr.coef_)





# Inițializarea Tkinter GUI

root= tk.Tk()



canvas1 = tk.Canvas(root, width = 600, height = 340)

canvas1.pack()



# Utilizăm scikit-learn

Intercept_result = ('Termenul liber (interceptul): ', regr.intercept_)

label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')

canvas1.create_window(260, 220, window=label_Intercept)



# Utilizăm scikit-learn

Coefficients_result  = ('Coeficienții: ', regr.coef_)

label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')

canvas1.create_window(260, 240, window=label_Coefficients)





# Introducerea noii valori pentru rata inflației

label1 = tk.Label(root, text='Introduceți rata inflației: ')

canvas1.create_window(100, 100, window=label1)



entry1 = tk.Entry (root) # crearea primului câmp de introducere al datelor

canvas1.create_window(270, 100, window=entry1)



# Introducerea noii valori pentru rata șomajului

label2 = tk.Label(root, text=' Introduceți rata șomajului: ')

canvas1.create_window(100, 120, window=label2)



entry2 = tk.Entry (root) # crearea celui de-al doilea câmp de introducere al datelor

canvas1.create_window(270, 120, window=entry2)



# Introducerea noii valori pentru datoria publică

label3 = tk.Label(root, text=' Introduceți datoria publică: ')

canvas1.create_window(100, 140, window=label3)



entry3 = tk.Entry (root) # crearea celui de-al treilea câmp de introducere al datelor

canvas1.create_window(270, 140, window=entry3)





def values(): 

    global Nou_Rata_inflatiei #Prima variabilă predictor

    Nou_Rata_inflatiei = float(entry1.get()) 

    

    global Nou_Rata_somajului #A doua variabilă predictor

    Nou_Rata_somajului = float(entry2.get()) 

    

    global Nou_Datoria_publica #A treia variabilă predictor

    Nou_Datoria_publica = float(entry3.get()) 

    

    Prediction_result  = ('PIB previzionat (mld. USD): ', regr.predict([[Nou_Rata_inflatiei ,Nou_Rata_somajului, Nou_Datoria_publica]]))

    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')

    canvas1.create_window(260, 280, window=label_Prediction)



# Crearea unui buton pentru aplicarea modelului antrenat prin funcția values() pentru noile valori introduse

button1 = tk.Button (root, text='Previzionează PIB-ul României',command=values, bg='orange') 

canvas1.create_window(270, 180, window=button1)

 



# Reprezentarea grafică a primului nor de puncte

figure3 = plt.Figure(figsize=(5,4), dpi=100)

ax3 = figure3.add_subplot(111)

ax3.scatter(df['Rata_inflatiei'].astype(float),df['PIB'].astype(float), color = 'r')

scatter3 = FigureCanvasTkAgg(figure3, root) 

scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)

ax3.legend() 

ax3.set_xlabel('Rata inflației')

ax3.set_title('Rata inflației vs. PIB')



#  Reprezentarea grafică a celui de-al doilea nor de puncte

figure4 = plt.Figure(figsize=(5,4), dpi=100)

ax4 = figure4.add_subplot(111)

ax4.scatter(df['Rata_somajului'].astype(float),df['PIB'].astype(float), color = 'g')

scatter4 = FigureCanvasTkAgg(figure4, root) 

scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)

ax4.legend() 

ax4.set_xlabel('Rata șomajului')

ax4.set_title('Rata șomajului vs. PIB')



# Reprezentarea grafică a celui de-al treilea nor de puncte

figure5 = plt.Figure(figsize=(5,4), dpi=100)

ax5 = figure5.add_subplot(111)

ax5.scatter(df['Datoria_publica'].astype(float),df['PIB'].astype(float), color = 'b')

scatter5 = FigureCanvasTkAgg(figure5, root) 

scatter5.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)

ax5.legend() 

ax5.set_xlabel('Datoria publica')

ax5.set_title('Datoria publica vs. PIB')



root.mainloop()