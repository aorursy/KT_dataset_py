def Ack():  # Acknowledgement about the code
    print("\nData science program")
    print("Pokemon Dataset , source : https://www.kaggle.com/abcsds/pokemon/data")

def Opn(file):  # Opens csv file in read mode
    import pandas as pnd
    return pnd.read_csv(file)

def PriDa(ofil):  # Finds mean median mode of 'ofil'
    import matplotlib.pyplot as plo
    print("\n\n Mean of dataset \n")
    print(ofil.mean())  #Displays mean of dataset
    print("\n\n Median of dataset \n")
    print(ofil.median())  #Displays median of dataset
    print("\n\n Mode of dataset")
    print(ofil.mode())  #Displays mode of dataset

def PriGph(ofil):  # Plots different graphs on 'ofil'
    import matplotlib.pyplot as plo
    print("Graph plots will be shown \n")
    print("Figure 1 - General plotting of data on graph")
    ofil.plot()  # Plots all columns against index graph
    print("Figure 2 - Scatter graph with x axis as Attack and y axis as Defence")
    ofil.plot(kind='scatter',x='Attack',y='Defense') # Plots scatter graph
    print("Figure 3 - Density graph")
    ofil.plot(kind='density')  # Plots estimate density function graph
    print("Figure 4 - Histogram graph")
    ofil.plot(kind='hist') # Plots histogram graph
    print("Figure 5 - Box graph")
    ofil.plot(kind='box') # plots Box graph
    plo.show() #visualization

def Menu():  # Calls all the functions 
    Ack()
    ofile = Opn("../input/Pokemon.csv")
    PriDa(ofile)
    PriGph(ofile)

Menu()
