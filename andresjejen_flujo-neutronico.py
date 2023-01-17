import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
def pond(miu):

    """

    Calcular el factor de ponderación para cada dirección.

    """

    ponderaciones = []

    for i in range(len(miu)-1):

        ponderaciones.append(miu[i]-miu[i+1])

    ponderaciones.append(miu[-1])

    ponderaciones = np.array(ponderaciones)

    

    return ponderaciones
def simulacion(dx = 0.001, dt= 0.025, x_0 = 0.0, t_0 = 0.05, dir = 20,v = 1, St = 1, Ss=0.5, x_f= 1,PhiAn=None,ponderaciones = False):

    """

    

    """

    

    #Aignación de valores

    

    #Nodos y tiempos

    x = x_0;

    t = t_0;

    

    #Direcciones

    dd = 1/dir;

    miu = -np.sort(-np.arange(dd,1+dd,dd));   #Discretización del dominio angular

    

    #B

    B = 1/(v*dt);

    

    # Phi Nodo Anterior    

    # Condiciones Iniciales

    # matriz de un solo nodo con filas de tiempos y columnas de direcciones

    PhiA = PhiAn;

    

    #Temporal Nodos y tiempo Nodo Actual

    TempTN = np.concatenate((np.array([[0,0]]),np.array([[0,dt]]),np.array([[0,2*dt]]),np.array([[0,3*dt]])),axis=0);

    

    # -------  |  Nodo  |   tiempo 

    #               0        0     

    #               0        0.025 

    #               0        2*dt

    #               0        3*t

    

    PhiA = np.concatenate((TempTN,PhiA),axis=1);

    

    # -------  |  Nodo  |   tiempo   |   Phi 1/20  |  Phi 2/20  |  .... | Phi 1 

    #               0        0             0                0               0      #transanterior

    #               0        0.025         0/1            0/1               1      #Anterir           #trananterior

    #               0        2*dt            0            0                 0      #Anterior

    #               0        3*t              0            0                0

    

    

    # Condiciones iniciales

    # Resultados finales de la simulación

    Phi = PhiA[0:2];  # Guarda 2 primeras filas en vector final



    # Valores actuales resultado de cada simulación

    # Valores del nodo actual

    PhiN = np.zeros([1,dir+2]);



    #Variable para navegar en el nodo anterior

    cont = 0;



    #Variables de ayuda

    val1 = [];     #Valores del nodo anterior tiempo anterior

    val2 = [];     #Valores del nodo anterior tiempo anterior anterior



    while (x <= x_f):

        val1 = PhiA[cont+1,2:];

        val2 = PhiA[cont,2:];

        

        phic = np.array([val1 + (dx)*( ( (B+St)*( ( val2 / (1+v*St*dt)) - val1 ) ) + ( (Ss/2)* np.sum(val1)*dd ) )/miu]);



        # Verificar si el resultado es 0

        if (phic == np.zeros([1,dir])).all():  #Terminínó el nodo

            

            Phi = np.concatenate((Phi,PhiN[1:]),axis=0);  #Push nuevos nodos

            

            PhiA = np.concatenate((np.zeros([1,dir+2]),PhiN,np.zeros([1,dir+2]),np.zeros([1,dir+2])),axis = 0);         #Nodo anterior

            

            PhiN = np.zeros([1,dir+2]);

            

            x = x + dx;    # Pasar al siguiente nodo

            t = 25*x+dt;   # Paso al tiempo válido del nodo

            

            cont = 1;    

            

        else:      # No he terminado el nodo

            PhiN = np.concatenate((PhiN,np.concatenate((np.array([[x,t]]),phic),axis = 1)),axis=0);    #Push nuevo calculo

            t = t+dt;

            cont = cont +1;

    

    #Crear el Dataframe

    columnas = ["Nodo","Tiempo"]

    for i,k in enumerate(miu):

        columnas.append("Phi "+ '{:.5}'.format(str(k)))



    df = pd.DataFrame(Phi[1:],columns=columnas);

    

    if (ponderaciones):

        ponde = pond(miu)

        return (df,ponde,columnas);

    else:

        return (df,columnas);
Ss = np.array([0.1,0.5,0.7,0.9,0.99999]);



Totales = pd.DataFrame();

Primeros = pd.DataFrame();

TotalCol = [];



fl_o = 1;

dir = 20;

CondIni = np.concatenate((np.zeros([1,dir]),np.ones([1,dir]),np.zeros([1,dir]),np.zeros([1,dir])),axis = 0);

v = 1000000



for i,k in enumerate(Ss):

    

    df,ponderaciones,columnas = simulacion(Ss = k, v = v,dir = dir,x_f = 3,PhiAn = CondIni,ponderaciones=True);

    

    Primeros = df.groupby('Nodo')[columnas[2:]].nth(0);

    

    Totales['Total Ss = ' + str(k)] = np.sum(np.multiply(Primeros[columnas[2:]],ponderaciones),axis=1)    ## Cálculo de flujo total ponderado

    TotalCol.append('Total Ss = ' + str(k))

    

    # Gráficas Flujos Dispersos y Ponderado

    plt.figure(figsize=(15,15))

    for w,a in enumerate(columnas[2:]):

        plt.plot(Primeros.index,Primeros[a],label = '{:.8}'.format(str(a)))



    plt.scatter(Primeros.index,Totales['Total Ss = ' + str(k)],c='red')   #Pintar ponderada

    plt.grid(True)

    plt.title("Flujos Dispersos y ponderado con Ss = " + str(k))

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("Flujo");

    plt.legend()

    plt.show()

    

#graficar fliujos ponderados

plt.figure(figsize=(15,15))    

for i ,k in enumerate(TotalCol):

    plt.plot(Primeros.index,Totales[k],label = k)

    plt.grid(True)

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("Flujos Totales");

plt.legend()

plt.show()


v = np.array([100,1000,10000,1000000,10000000]);



Totales = pd.DataFrame();

Primeros = pd.DataFrame();

TotalCol = [];

Ss = 0.7



fl_o = 1;

dir = 20;

CondIni = np.concatenate((np.zeros([1,dir]),np.ones([1,dir]),np.zeros([1,dir]),np.zeros([1,dir])),axis = 0);



for i,k in enumerate(v):

    

    df,ponderaciones,columnas = simulacion(Ss = Ss, v = k,dir = dir,x_f = 3,PhiAn = CondIni,ponderaciones=True);

    

    Primeros = df.groupby('Nodo')[columnas[2:]].nth(0);

    

    Totales['Total v = ' + str(k)] = np.sum(np.multiply(Primeros[columnas[2:]],ponderaciones),axis=1)

    TotalCol.append('Total v = ' + str(k))

    

    # Gráficas Flujos Dispersos y Ponderado

    plt.figure(figsize=(15,15))

    for w,a in enumerate(columnas[2:]):

        plt.plot(Primeros.index,Primeros[a],label = '{:.8}'.format(str(a)))



    plt.scatter(Primeros.index,Totales['Total v = ' + str(k)],c='red')

    plt.grid(True)

    plt.title("Flujos Dispersos y ponderado con v = " + str(k))

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("Flujo");

    plt.legend()

    plt.show()

    

plt.figure(figsize=(15,15))    



for i ,k in enumerate(TotalCol):

    plt.plot(Primeros.index,Totales[k],label = k)

    plt.grid(True)

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("Flujos Totales");

plt.legend()

plt.show()
#Valores de las simulaciones



Ss = np.array([0.4140003,0.3553593,0.3537399,0.4637000,0.0859787]);

St = np.array([0.4140398,0.3553656,0.3537402,0.4637032,0.1141542]);

v = np.array([0.00021876,0.0013835,0.043751,1.3835,4.3751])*np.power(10,9);

ev = np.array([0.0253,1,1000,1000000,10000000]);

fn = np.array([2.501181,2.055437,2043.516,2020819,6404553])*(1/np.power(10,15));



gen = np.array([Ss,St,v,ev,fn]).T



Totales = pd.DataFrame();

Primeros = pd.DataFrame();

TotalCol = [];



fl_o = 1;

dir = 20;

CondIni = np.concatenate((np.zeros([1,dir]),np.ones([1,dir]),np.zeros([1,dir]),np.zeros([1,dir])),axis = 0);





for i,k in enumerate(gen):

    

    df,ponderaciones,columnas = simulacion(Ss = k[0],St = k[1], v = k[2],dir = dir,x_f = 3,PhiAn = CondIni,ponderaciones=True);

    

    Primeros = df.groupby('Nodo')[columnas[2:]].nth(0);

    

    Totales['Total Ss = ' + str(k[0]) + ', St =  ' + str(k[1])  + ', v =  ' + str(k[2])+ ', ev =  ' + str(k[3])+ ', fn =  ' + str(k[-1])] = np.power(10,12)*np.sum(np.multiply(Primeros[columnas[2:]],ponderaciones),axis=1)

    TotalCol.append('Total Ss = ' + str(k[0]) + ', St =  ' + str(k[1])  + ', v =  ' + str(k[2])+ ', ev =  ' + str(k[3]) + ', fn =  ' + str(k[-1]))

    

############ MULTIPLICAR POR F_0 10^12

    

# Gráficas Flujos Totales

plt.figure(figsize=(15,15))    

for i ,k in enumerate(TotalCol):

    plt.plot(Primeros.index,Totales[k],label = k)

    plt.grid(True)

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("neutrones/(cm^2*s)");

plt.legend()

plt.show()





Dosis = pd.DataFrame();



for i,k in enumerate(TotalCol):

    Dosis[k] = gen[i,-1]*Totales[k]



# Gráficas Dosis    

plt.figure(figsize=(15,15))       ## ^-15

for i ,k in enumerate(TotalCol[0:2]):

    ymax = Dosis[k].max()

    plt.plot(Primeros.index,Dosis[k],label = "Dosis "+ k + " , Dosis_Max ={:1.4f}".format(ymax))    

    plt.grid(True)

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("TASA DE DOSIS (cGy/s)");

plt.legend()

plt.show()



plt.figure(figsize=(15,15))     ## ^-12

ymax = Dosis[TotalCol[2]].max()

plt.plot(Primeros.index,Dosis[TotalCol[2]],label = "Dosis "+ TotalCol[2] + " , Dosis_Max ={:1.4f}".format(ymax))   

plt.grid(True)

plt.xlabel("Nodo x (cm)");

plt.ylabel("TASA DE DOSIS (cGy/s)");

plt.legend()

plt.show()



plt.figure(figsize=(15,15))    

for i ,k in enumerate(TotalCol[3:]):   ## ^-9

    ymax = Dosis[k].max()

    plt.plot(Primeros.index,Dosis[k],label = "Dosis "+ k + " , Dosis_Max ={:1.4f}".format(ymax))   

    plt.grid(True)

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("TASA DE DOSIS (cGy/s)");

plt.legend()

plt.show()





Ss = np.array([0.1,0.5,0.7,0.9,0.99999]);

Totales = pd.DataFrame();

Primeros = pd.DataFrame();

TotalCol = [];



fl_o = 1;

dir = 20;

CondIni = np.concatenate((np.zeros([1,dir]),np.array([np.append(fl_o,np.zeros([1,dir-1]))]),np.zeros([1,dir]),np.zeros([1,dir])),axis = 0);



for i,k in enumerate(Ss):

    

    df,columnas = simulacion(Ss = k, v = 1000000,dir = dir,x_f = 5,PhiAn = CondIni);

    

    Primeros = df.groupby('Nodo')[columnas[2:]].nth(0);

    

    Totales['Total Ss = ' + str(k)] = np.sum(Primeros[columnas[2:]],axis=1)

    TotalCol.append('Total Ss = ' + str(k))

    

    # Gráficas Flujos dispersos

    plt.figure(figsize=(15,15))

    for w,a in enumerate(columnas[3:]):

        plt.plot(Primeros.index,Primeros[a],label = '{:.8}'.format(str(a)))



    plt.grid(True)

    plt.title("Flujos Dispersos con Ss = " + str(k))

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("Flujo");

    plt.legend()

    plt.show()

    

    plt.figure(figsize=(15,15))

    plt.plot(Primeros.index,Primeros[columnas[2]],label = columnas[2])

    plt.plot(Primeros.index,Totales['Total Ss = ' + str(k)],c='red',label = 'Flujo Total')

    plt.grid(True)

    plt.title("Flujo incidente y Total con Ss = " + str(k))

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("Flujo");

    plt.legend()

    plt.show()



plt.figure(figsize=(15,15))    



for i ,k in enumerate(TotalCol):

    xmax = np.argmax(Totales[k]);

    ymax = Totales[k].max()

    plt.plot(Primeros.index,Totales[k],label = k + " x_Max (cm)= {:1.4f}, Flujo_Total_Max ={:1.4f}".format(xmax, ymax))

    plt.grid(True)

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("Flujos Totales");

plt.legend()

plt.show()
v = np.array([100,1000,10000,1000000,10000000]);



Totales = pd.DataFrame();

Primeros = pd.DataFrame();

TotalCol = [];

Ss = 0.7



fl_o = 1;

dir = 20;

CondIni = np.concatenate((np.zeros([1,dir]),np.array([np.append(fl_o,np.zeros([1,dir-1]))]),np.zeros([1,dir]),np.zeros([1,dir])),axis = 0);



for i,k in enumerate(v):

    

    df,columnas = simulacion(Ss = Ss, v = k,dir = dir,x_f = 3,PhiAn = CondIni);

    

    Primeros = df.groupby('Nodo')[columnas[2:]].nth(0);

    

    Totales['Total v = ' + str(k)] = np.sum(Primeros[columnas[2:]],axis=1)

    TotalCol.append('Total v = ' + str(k))

    

    # Gráficas Flujos dispersos

    plt.figure(figsize=(15,15))

    for w,a in enumerate(columnas[3:]):

        plt.plot(Primeros.index,Primeros[a],label = '{:.8}'.format(str(a)))



    plt.grid(True)

    plt.title("Flujos Dispersos con v = " + str(k))

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("Flujo");

    plt.legend()

    plt.show()

    

    #Gráfica de Flujo incidente y Total

    plt.figure(figsize=(15,15))

    plt.plot(Primeros.index,Primeros[columnas[2]],label = columnas[2])

    plt.plot(Primeros.index,Totales['Total v = ' + str(k)],c='red',label = 'Flujo Total')

    plt.grid(True)

    plt.title("Flujo incidente y Total con v = " + str(k))

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("Flujo");

    plt.legend()

    plt.show()



plt.figure(figsize=(15,15))    



for i ,k in enumerate(TotalCol):

    xmax = np.argmax(Totales[k]);

    ymax = Totales[k].max()

    plt.plot(Primeros.index,Totales[k],label = k + " x_Max (cm)= {:1.4f}, Flujo_Total_Max ={:1.4f}".format(xmax, ymax))

    plt.grid(True)

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("Flujos Totales");

plt.legend()

plt.show()
#Valores de las simulaciones



Ss = np.array([0.4140003,0.3553593,0.3537399,0.4637000,0.0859787]);

St = np.array([0.4140398,0.3553656,0.3537402,0.4637032,0.1141542]);

v = np.array([0.00021876,0.0013835,0.043751,1.3835,4.3751])*np.power(10,9);

ev = np.array([0.0253,1,1000,1000000,10000000]);

fn = np.array([2.501181,2.055437,2043.516,2020819,6404553])*(1/np.power(10,15));



gen = np.array([Ss,St,v,ev,fn]).T



Totales = pd.DataFrame();

Primeros = pd.DataFrame();

TotalCol = [];



fl_o = 1;

dir = 20;

CondIni = np.concatenate((np.zeros([1,dir]),np.array([np.append(fl_o,np.zeros([1,dir-1]))]),np.zeros([1,dir]),np.zeros([1,dir])),axis = 0);



for i,k in enumerate(gen):

    

    df,columnas = simulacion(Ss = k[0],St = k[1], v = k[2],dir = dir,x_f = 5,PhiAn = CondIni);

    

    Primeros = df.groupby('Nodo')[columnas[2:]].nth(0);

    

    Totales['Total Ss = ' + str(k[0]) + ', St =  ' + str(k[1])  + ', v =  ' + str(k[2])+ ', ev =  ' + str(k[3])+ ', fn =  ' + str(k[-1])] = np.power(10,12)*np.sum(Primeros[columnas[2:]],axis=1)

    TotalCol.append('Total Ss = ' + str(k[0]) + ', St =  ' + str(k[1])  + ', v =  ' + str(k[2])+ ', ev =  ' + str(k[3]) + ', fn =  ' + str(k[-1])) 

# Gráficas Flujos Totales

plt.figure(figsize=(18,18))    

for i ,k in enumerate(TotalCol):

    xmax = np.argmax(Totales[k]);

    ymax = Totales[k].max()

    plt.plot(Primeros.index,Totales[k],label = k + " x_Max (cm)= {:1.4f}, Flujo_Total_Max ={:1.4f}".format(xmax, ymax))   

    plt.grid(True)

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("neutrones/(cm^2*s)");

plt.legend()

plt.show()



Dosis = pd.DataFrame();



for i,k in enumerate(TotalCol):

    Dosis[k] = gen[i,-1]*Totales[k]



# Gráficas Dosis    

plt.figure(figsize=(18,18))       ## ^-15

for i ,k in enumerate(TotalCol[0:2]):

    xmax = np.argmax(Dosis[k]);

    ymax = Dosis[k].max()

    plt.plot(Primeros.index,Dosis[k],label = "Dosis " + k + " x_Max (cm)= {:1.4f}, Dosis_Max ={:1.4f}".format(xmax, ymax))   

    plt.grid(True)

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("TASA DE DOSIS (cGy/s)");

plt.legend()

plt.show()



plt.figure(figsize=(18,18))     ## ^-12

xmax = np.argmax(Dosis[TotalCol[2]]);

ymax = Dosis[TotalCol[2]].max()

plt.plot(Primeros.index,Dosis[TotalCol[2]],label = "Dosis " + TotalCol[2] + " x_max (cm)= {:1.4f}, Dosis_Max ={:1.4f}".format(xmax, ymax))   

plt.grid(True)

plt.xlabel("Nodo x (cm)");

plt.ylabel("TASA DE DOSIS (cGy/s)");

plt.legend()

plt.show()



plt.figure(figsize=(18,18))    

for i ,k in enumerate(TotalCol[3:]):   ## ^-9

    xmax = np.argmax(Dosis[k]);

    ymax = Dosis[k].max()

    plt.plot(Primeros.index,Dosis[k],label = "Dosis "+ k + " x_max(cm)= {:1.4f}, Dosis_Max ={:1.4f}".format(xmax, ymax))   

    plt.grid(True)

    plt.xlabel("Nodo x (cm)");

    plt.ylabel("TASA DE DOSIS (cGy/s)");

plt.legend()

plt.show()