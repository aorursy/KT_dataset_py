import pandas as pd

import numpy as np



import matplotlib 

from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import matplotlib as mpl

from mpl_toolkits.basemap import Basemap

from matplotlib import animation

from IPython.display import HTML

from IPython.display import Image







class Dataset:

    

    COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

    COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

    covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv").dropna(axis = 0)

    time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

    time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

    time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

    

    def gen(dicty, i, df):

        for x,y,z,w in zip(df.iloc[:,i],df.iloc[:,5],df.iloc[:,6],df.iloc[:,7]):

            dicty[x] = [0,0,0,0,0,0]

        for x,y,z,w in zip(df.iloc[:,i],df.iloc[:,5],df.iloc[:,6],df.iloc[:,7]):

            dicty[x][0] += 1

            dicty[x][1] += y

            dicty[x][2] += z

            dicty[x][3] += w

        for x in list(set(df.iloc[:,i])):

            if(dicty[x][1] != 0):

                dicty[x][4] = round(100.0*dicty[x][2]/dicty[x][1],2) 

                dicty[x][5] = round(100.0*dicty[x][3]/dicty[x][1],2)

            else:

                dicty[x][4] = 0 

                dicty[x][5] = 0

    states = {}

    countries = {}

    gen(states, 2, covid_19_data)

    gen(countries, 3, covid_19_data)

    Data = {"states": states, "countries" : countries}

    

    def frecuencia(self, j):

        fig = plt.figure(figsize=(30, 10))

        A = ['countries', 'states']

        for i in range(1,3):

            plt.subplot(2, 1, i)

            data = pd.DataFrame.from_dict(self.Data[A[i-1]])

            data.T.reset_index()

            if j==1:

                u = "Frecuencia total " + str(data.T.reset_index().iloc[:,j].sum())

                UU = "Frecuecia_total"

            elif j==2:

                u = "Casos confirmados " + str(data.T.reset_index().iloc[:,j].sum())

                UU = "Casos_confirmados"

            elif j == 3:

                u = "casos de muerte " + str(data.T.reset_index().iloc[:,j].sum())

                UU = "Casos_muerte"

            elif j== 4:

                u= "casos de curados " + str(data.T.reset_index().iloc[:,j].sum())

                UU = "Casos_curados"

            elif j== 5:

                u= "tasa de muertes, tasa media = "  + str(data.T.reset_index().iloc[:,j].mean())

                UU = "Tasa_muerte"

            else:

                u= "tasa de curados, tasa media = " + str(data.T.reset_index().iloc[:,j].mean())

                UU = "Tasa_curado"

                

            if i==1:

                matplotlib.rc('xtick', labelsize=5)

                plt.title(u + ' por paises')

                plt.xlabel('País')

            if i==2:

                matplotlib.rc('xtick', labelsize=10)

                plt.title(u + ' por ciudades')

                plt.xlabel('Ciudad')

            matplotlib.rc('ytick', labelsize=15) 

            plt.bar(data.T.reset_index().iloc[:,0], data.T.reset_index().iloc[:,j], align = 'center' , label= u+ ' para '+A[i-1])

            plt.plot(data.T.reset_index().iloc[:,0], data.T.reset_index().iloc[:,j],'k--', linewidth=4, label='')

            plt.tick_params(axis='x', rotation=90)

            plt.xlim(0,len(data.T.reset_index()))

            plt.subplots_adjust(hspace=0.7, wspace = 0.4)

            plt.legend(loc='upper right')

            #plt.yscale('log')

            plt.ylabel('counts')

        fig.savefig(UU+".png")

        plt.show()

        

    def mappingData(self,j):

        fig = plt.figure(figsize=(30, 20))

        m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)

        m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)

        m.fillcontinents(color='grey', alpha=0.3)

        m.drawcoastlines(linewidth=0.1, color="white")

        ax = plt.gca

        if j==1:

            data = self.time_series_covid_19_confirmed

            title = ' contagios confirmados hasta el '

            giff = "contagios"

        if j==2:

            data = self.time_series_covid_19_deaths

            title = ' muertos confirmados hasta el '

            giff = "muertos"

        if j==3:

            data = self.time_series_covid_19_deaths

            title = ' curados confirmados hasta el '

            giff = "curados"

            

        points = m.scatter(data['Long'], data['Lat'], s=data.iloc[:,4], alpha=0.4, cmap="Set1", c =data.iloc[:,4])

        def animate(i):

            plt.title(str(data.iloc[:,4+i].sum())+title+ data.columns[4+i], fontsize=20)

            points = m.scatter(data['Long'], data['Lat'], s=data.iloc[:,4+i], alpha=0.4, cmap="Set1", c =data.iloc[:,4+i])

            return points,

        anim = animation.FuncAnimation(fig,animate,interval=5,blit=True,frames=np.arange(2, 51, 1), repeat = True)

        %time anim.save(giff+'.gif', writer='imagemagick', fps=5)

        #%time anim.save('aqis2.gif', writer='pillow', fps=30)

        plt.close()

    def timeseries(self,j):

        if j==1:

            df = self.time_series_covid_19_confirmed.iloc[:,4::].T

        elif j==2:

            df = self.time_series_covid_19_deaths.iloc[:,4::].T

        elif j==3:

            df = self.time_series_covid_19_recovered.iloc[:,4::].T

        df.columns = list(dataset.time_series_covid_19_confirmed['Country/Region'])

        dicty ={'dates': df.index}

        for x in df.columns:

            dicty[x] = np.zeros(len(df), dtype=int)

        for x in list(df.columns):

            if len(df[x].to_numpy().shape) == 1:

                dicty[x] += df[x].to_numpy()

            if len(df[x].to_numpy().shape) > 1:

                dicty[x] += np.sum(df[x].to_numpy(), axis=1)

        data = pd.DataFrame.from_dict(dicty)

        return data

    

    def timeseriesContagios(self):

        data = self.timeseries(1)

        fig = plt.figure(figsize=(30, 85))

        A = [93000000,10000,3000,1000,500,200,110,60]

        B = [10000,3000,1000,500,200,110,60,30]

        for j in range(1,9):

            plt.subplot(8, 1, j)

            for x in data.columns[1::]:

                if (A[j-1]>data[x].sum()>=B[j-1]):

                    plt.plot(data['dates'],data[x], 'o--')

                    plt.title(str(A[j-1])+' - '+str(B[j-1]), fontsize=20)

            plt.tick_params(axis='x', rotation=90)

            plt.yscale('log')

            plt.legend(loc='upper left', prop={"size":20})

            matplotlib.rc('ytick', labelsize=20)

            matplotlib.rc('xtick', labelsize=20)

        plt.subplots_adjust(hspace=0.3)

        fig.savefig("contagios.png")

            

    def timeseriesDeathsRecovered(self):

        fig = plt.figure(figsize=(30, 40))

        for j in range(2,4):

            plt.subplot(2, 1, j-1)

            if j==2:

                plt.title('Death cases', fontsize=20)

            elif j==3:

                plt.title('Recovered cases', fontsize=20)

            data = self.timeseries(j)

            for x in data.columns[1::]:

                if (data[x].sum()>=50):

                    plt.plot(data['dates'],data[x], 'o--')

                plt.tick_params(axis='x', rotation=90)

                plt.yscale('log')

                plt.legend(loc='upper left', prop={"size":20})

                matplotlib.rc('ytick', labelsize=20)

                matplotlib.rc('xtick', labelsize=20)

        fig.savefig("t.png")

dataset = Dataset()

dataset.covid_19_data.head()
dataset.frecuencia(1)
dataset.frecuencia(2)
dataset.frecuencia(3)
dataset.frecuencia(4)
dataset.frecuencia(5)
dataset.frecuencia(6)
dataset.time_series_covid_19_confirmed.head()
dataset.mappingData(1)

dataset.mappingData(2)

dataset.mappingData(3)
Image("../working/muertos.gif")
Image("../working/contagios.gif")
Image("../working/curados.gif")
dataset.timeseriesContagios()
dataset.timeseriesDeathsRecovered()