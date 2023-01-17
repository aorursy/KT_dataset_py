
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from matplotlib import animation
from itertools import product
import plotly as py
import plotly.graph_objs as go
import ipywidgets as widgets
class game:

    dataframelist = []   

    def __init__(self,n,t):
        self.n= n
        self.t= t
              
    def iniz(self):
        del self.dataframelist[:] #azzerare i valori del dataframe precedente,sennò continua ad appendere 
        x_position= np.zeros((self.n,self.t))     
        y_position= np.zeros((self.n,self.t))
        x_values_initial = []
        y_values_initial= []
        positions = [[]]
        distances = np.array(0)
        alive = np.ones((self.n,self.t+1),dtype=bool)
        for i in range(self.n):
            x_values_initial.append(random.randint(1,50))
            y_values_initial.append(random.randint(1,50))
        x_position = np.insert(x_position,0,x_values_initial)
        x_position = np.reshape(x_position,(self.n,self.t+1), order='F')
        y_position = np.insert(y_position,0,y_values_initial)
        y_position = np.reshape(y_position,(self.n,self.t+1), order='F')
        for j in range(1,(self.t+1)):
            for i in range(self.n):
                x_position[i][j] = x_position[i][j-1] + random.choice((-1,0,+1))
                y_position[i][j] = y_position[i][j-1] + random.choice((-1,0,+1))
        for i in range(self.n):
             for j in (range(self.t+1)):
                positions.append([x_position[i][j],y_position[i][j]])
        positions_array = np.array(positions)
        positions_array = positions_array[1:]
        positions_array = np.reshape(positions_array,(self.n,(self.t+1)))
        for h in range(self.t+1):
            for i in range(self.n):
                for j in range(self.n):
                    distances = np.append(distances,(distance.euclidean(positions_array[i][h],positions_array[j][h])))
        distances = distances[1:]
        distances = np.reshape(distances,(self.t+1,self.n,self.n))
        
        num_bias = np.count_nonzero(distances<7,axis=1)
        num_corr = num_bias - 1
        num_corr = np.transpose(num_corr)

        position_neighborhood = []
        for i in range(self.n):
             position_neighborhood.append([list(a) for a in zip(positions_array[i],num_corr[i])])
        for i in range(self.n):
            self.dataframelist.append([list(a) for a in zip(position_neighborhood[i],alive[i])])
        dataframe = pd.DataFrame(self.dataframelist)
        for h in range(self.t):
            for i in range(self.n):
                if self.dataframelist[i][h][1] == True:                #se vero
                    if self.dataframelist[i][h][0][1] < 2:                #se quelli vicino meno di 2
                        self.dataframelist[i][h+1][1] = False               #allora falso
                    if self.dataframelist[i][h][0][1] > 5:                #se quelli vicino maggiori di 5
                        self.dataframelist[i][h+1][1] = False               #allora falso
                    if self.dataframelist[i][h][0][1] == 5:               #se quelli vicino uguali a 5
                        self.dataframelist[i][h+1][1] = True                #allora vero
                    if self.dataframelist[i][h][0][1] == 4:               #se quelli vicino uguali a 4
                        self.dataframelist[i][h+1][1] = True                #allora vero
                    if self.dataframelist[i][h][0][1] == 3:               #se quelli vicino uguali a 3
                        self.dataframelist[i][h+1][1] = True                #allora vero
                    if self.dataframelist[i][h][0][1] == 2 :              #se quelli vicino uguali a 2
                        self.dataframelist[i][h+1][1] = True                #allora vero
                        
                if self.dataframelist[i][h][1] == False:                #se falso
                    self.dataframelist[i][h+1][1] = False                 #allora quello dopo falso                    
                if self.dataframelist[i][h][1] == False:                #se falso
                    if self.dataframelist[i][h][0][1] == 3:               #se quelli vicino uguale a 3
                        self.dataframelist[i][h+1][1] = True                #allora vero
                    if self.dataframelist[i][h][0][1] == 4:               #se quelli vicino uguale a 4
                        self.dataframelist[i][h+1][1] = True                #allora vero         
                    if self.dataframelist[i][h][0][1] == 5:               #se quelli vicino uguale a 5
                        self.dataframelist[i][h+1][1] = True                #allora vero         
       
    
    # return (dataframe)
    def dataframe(self):
        '''return the dataframe with all the data positions,the state and the number of neighbours'''
        return (pd.DataFrame(self.dataframelist))
    
    
    def prepare_plot(self):
        '''not necessary to call'''
        #signals= widgets.SelectMultiple(options=range((self.t+1)), value=[0,], description = 'm')
        dataframe = pd.DataFrame(self.dataframelist)
        x = []
        y = []
        for h in np.arange(self.t+1):
            for i in np.arange(self.n):
                if dataframe[h][i][1] == True:
                    x.append(dataframe[h][i][0][0][0])
                    y.append(dataframe[h][i][0][0][1])
                else:
                    x.append(-999)
                    y.append(-999)
        x=np.asarray(x)
        y=np.asarray(y)
        x=np.reshape(x,(self.t+1,self.n))
        y=np.reshape(y, (self.t+1,self.n))  
        return(x,y)
values = game(100,8)
game.iniz(values)
#game.dataframe(values) #se si vogliono vedere i dati
layout = go.Layout(
    title= 'GAME',
    yaxis=dict(range=[0,55]),
    xaxis=dict(range=[0,55])
)

def update_plot(signals):
    datax = []
    for i in signals:
        trace1 = go.Scatter(
            x = game.prepare_plot(values)[0][i],
            y = game.prepare_plot(values)[1][i],
            mode='markers'
        )
        datax.append(trace1)
    
    fig = go.Figure(data=datax,layout=layout)
    py.offline.iplot(fig)
    
signals= widgets.SelectMultiple(options=range((len(game.prepare_plot(values)[1]))), value=[0,], description = 'time')
widgets.interactive(update_plot,signals=signals)
