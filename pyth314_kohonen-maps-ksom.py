import numpy as np

import pandas as pd

# Import images from web 

import requests

from io import BytesIO
# Function to manage the input of Data

import glob 

def data_path_from_name(name):

    L=glob.glob(f'../**/{name}',recursive=True)

    if len(L)>1 : 

        print(f'All path for {name} :')

        print(L)

        print(f'Data path return {L[0]}')

    return(L[0])

    pass

data_path_from_name('house.png')
dist=lambda a,b : np.linalg.norm(a-b) # Grid metric 

neighborhood = lambda k,r=1,bound=[0,5] : [k-i for i in range(1,r+1) if k-i >=0]+[k]+[k+i for i in range (1,r+1) if k+i<=5]

phi=lambda k,i,r=1,bound=[0,5] : 1 if i in neighborhood(k,r,bound) else 0 # Lateral interaction function 
class Kmap:

    def __init__(self,l_w,r=0):

        self.w=np.array(l_w,dtype='float64')

        self.r=r

    def input_feature_space(self,list_):

        self.input_feature_space=[]

        for l in list_:

            self.input_feature_space.append(np.vstack(np.array(l)))

    def _winner_talk(self,x,dist=dist):

        ar_dist=np.apply_along_axis(lambda w : dist(w,x.T),arr=self.w,axis=1)

        #print('dist : ',ar_dist)

        winner=ar_dist.argmin()

        min_dist=ar_dist[winner]

        return(winner,min_dist)

    def update_weight(self,x,alpha,dist=dist,phi=phi):

        phi_=lambda k,i : phi(k,i,r=self.r,bound=[0,self.w.shape[0]-1])

        winner,min_dist=self._winner_talk(x,dist)

        for i,w in enumerate(self.w): # Itterate over the row 

            #print('i :',i,'(x.T-w[:])',(x.T-w[:])[0])

            self.w[i]+=alpha*phi_(winner,i)*(x.T-w[:])[0]

        return(self.w)
# The right way to design complex and adjustable function is to design a class and implement __call__(self) method

class ConditionalMap_fitness:

    """ The such a purpose is to define how well is a map against some conditions : 

    so this function will evaluate each move of point against the map to say if that move tends more 

    to the center of tile that is similare to an other (good to have a few number of tile shape) 

    and tyle that don't overlap with others"""

    

    def __init(self,input_space=None):

        self.input_space=input_space

    

    def _adjustment_of_param(self):

        """Desgin some adjustment paramater technic such as simulated anealing stuff or like this"""

        pass

    

    def overlapingSquared(self,map_):

        pass

    def similaritySquared(self,map_):

        pass

    

    def __call__(self,map_):

        return(self.similaritySquared(map_) - overlapingSquared(map_))
generate_feature_space=lambda size,n=3: [np.random.rand(1,n)[0].tolist() for _ in range(size)]

feature_space=generate_feature_space(100)
import pandas as pd



class Kmap:

    

    def __init__(self,l_w,r=0,dist=dist,phi_=phi):

        self.w=np.array(l_w,dtype='float64') # Initialize the weigth of neurones

        self.r=r # radius of the interconnection function

        self.dist=dist # metric of the kohonen grid

        self.phi=lambda k,i : phi_(k,i,r=self.r,bound=[0,self.w.shape[0]-1]) # interconnection function

        

    def input_feature_space(self,list_,weight_init=True):

        self.feature_space=[]

        for l in list_:

            self.feature_space.append(np.vstack(np.array(l)))

        self.barycencter=sum(self.feature_space)/len(self.feature_space)

        if weight_init:

            self.w=np.array([np.random.normal(n_kmap.barycencter)[:,0].tolist() for _ in range(len(self.w))],dtype='float64') # Initialize the weigth of neurones

            





    def _winner_talk(self,x):

        dist=self.dist

        ar_dist=np.apply_along_axis(lambda w : dist(w,x.T),arr=self.w,axis=1)

        #print('dist : ',ar_dist)

        winner=ar_dist.argmin()

        min_dist=ar_dist[winner]

        return(winner,min_dist)

    

    def update_weight(self,x,alpha):

        

        dist,phi=self.dist,self.phi

        

        winner,min_dist=self._winner_talk(x,dist)

        for i,w in enumerate(self.w): # Itterate over the row 

            #print('i :',i,'(x.T-w[:])',(x.T-w[:])[0])

            self.w[i]+=alpha*phi(winner,i)*(x.T-w[:])[0]

        return(self.w)

    

    def batch_update(self,alpha,conditionalMapFitness=None):

        dist,phi=self.dist,self.phi

        

        l_data=[]

        for x in self.feature_space :

            winner,min_dist=self._winner_talk(x)

            for i,w in enumerate(self.w): # Itterate over the row 

                if phi(winner,i)>0:

                    d={'neurone':i,'move':alpha*phi(winner,i)*(x.T-w[:])[0]}

                    l_data.append(d)

        data=pd.DataFrame(l_data,columns=['neurone','move'])

        

        if conditionalMapFitness is not None : 

            data['fitness']=data.apply(func=lambda x : conditionalMapFitness(x.neurone,x.move), axis=1)

        else :

            data['fitness']=1

        

        # Renvoie l ajustment accumuler en donnant sense a la fitness (Normalisation de la fitness par neurones)

        accumulate_move=lambda i :(data[data.neurone==i].move*\

                                   data[data.neurone==i].fitness/(data[data.neurone==i].fitness).sum()\

                                  ).sum() 

        

        for i in data.neurone.unique().tolist(): # Pour tout les neurones qui ont besoin d etre updater 

            self.w[i]+=accumulate_move(i)

            

        return(self.w)



n_kmap=Kmap([[3,-1,2],[-3,5,1]],r=0)

n_kmap.input_feature_space(feature_space)
n_kmap.w
for _ in range(100):

    n_kmap.batch_update(alpha=0.1)

n_kmap.w
# Exemple of generator

def infinite_sequence():

    num = 0

    while True:

        yield num

        num += 1

inf=infinite_sequence()
for _ in range(10):

    print(next(inf))
# Overide the function iter & next of a class to itterate over a certain object

from PIL import Image

import random



class FeatureSpaceFromImage: 

    

    def __init__(self,image="luigi.jpg",batch=10000):

        try :

            self.im = Image.open(data_path_from_name(image))

        except :

            response = requests.get(image)

            self.im = Image.open(BytesIO(response.content))

            

                

        self.batch = batch

        

        norm=np.linalg.norm(np.array(list(self.im.size)))

        self.figsize=(10*self.im.size[0]/norm,10*self.im.size[1]/norm)

    

    def __iter__(self):

        self.n = 0

        return self

    

    def __next__(self):

        if self.n<self.batch :

            x,y=random.randint(0,self.im.size[0]-1),random.randint(0,self.im.size[1]-1)

            while self.im.getpixel((x,y))==self.im.getpixel((0,0)):

                x,y=random.randint(0,self.im.size[0]-1),random.randint(0,self.im.size[1]-1)

            self.n += 1

            return(np.vstack(np.array([x,self.im.size[1]-1-y],dtype='float64')),self.im.getpixel((x,y)))

        else:

            raise StopIteration()
feature_space=FeatureSpaceFromImage()

feature_space_gen=iter(feature_space)
feature_space.im.size[0]
#%matplotlib notebook

import matplotlib.pyplot as plt

%matplotlib inline







plt.rcParams['figure.figsize'] = feature_space.figsize

l_points=list(feature_space_gen)

x,y=[p[0,0] for p,_ in l_points],[p[1,0] for p,_ in l_points]

rgb=[tuple([c_/255 for c_ in list(c)]) for _,c in l_points]
plt.scatter(x, y, alpha=1,marker='.')

plt.show()
plt.scatter(x, y, c=rgb, alpha=1,marker='.')

plt.show()
feature_space.im
dist=lambda a,b : np.linalg.norm(a-b) # Grid metric 

neighborhood = lambda k,r=1,bound=[0,5] : [k-i for i in range(1,r+1) if k-i >=0]+[k]+[k+i for i in range (1,r+1) if k+i<=5]

phi=lambda k,i,r=1,bound=[0,5] : 1 if i in neighborhood(k,r,bound) else 0 # Lateral interaction function 

class Kmap:

    

    def __init__(self,num_neurons,feature_space,r=0,dist=dist,phi_=phi):

        self.num_neurons=num_neurons

        self.r=r # radius of the interconnection function

        self.dist=dist # metric of the kohonen grid

        self.phi=lambda k,i : phi_(k,i,r=self.r,bound=[0,self.w.shape[0]-1]) # interconnection function

        self.feature_space=feature_space

        feature_space_gen=iter(feature_space)

        self.w=np.array([next(feature_space_gen)[0].T[0].tolist() for _ in range(num_neurons)])

        

        

    def input_feature_space(self,list_,weight_init=True):

        self.feature_space=[]

        for l in list_:

            self.feature_space.append(np.vstack(np.array(l)))

        self.barycencter=sum(self.feature_space)/len(self.feature_space)

        if weight_init:

            self.w=np.array([np.random.normal(n_kmap.barycencter)[:,0].tolist() for _ in range(len(self.w))],dtype='float64') # Initialize the weigth of neurones

            





    def _winner_talk(self,x):

        dist=self.dist

        ar_dist=np.apply_along_axis(lambda w : dist(w,x.T),arr=self.w,axis=1)

        #print('dist : ',ar_dist)

        winner=ar_dist.argmin()

        min_dist=ar_dist[winner]

        return(winner,min_dist)

    

    def update_weight(self,x,alpha):

        

        dist,phi=self.dist,self.phi

        

        winner,min_dist=self._winner_talk(x,dist)

        for i,w in enumerate(self.w): # Itterate over the row 

            #print('i :',i,'(x.T-w[:])',(x.T-w[:])[0])

            self.w[i]+=alpha*phi(winner,i)*(x.T-w[:])[0]

        return(self.w)

    

    def batch_update(self,alpha,max_bash=100,conditionalMapFitness=None):

        dist,phi=self.dist,self.phi

        

        l_data=[]

        i_stop=0

        feature_space_gen=iter(self.feature_space)

        for x,_ in  feature_space_gen:

            i_stop+=1

            if i_stop>max_bash: 

                break

            winner,min_dist=self._winner_talk(x)

            for i,w in enumerate(self.w): # Itterate over the row 

                if phi(winner,i)>0:

                    d={'neurone':i,'move':alpha*phi(winner,i)*(x.T-w[:])[0]}

                    l_data.append(d)

        data=pd.DataFrame(l_data,columns=['neurone','move'])

        

        if conditionalMapFitness is not None : 

            data['fitness']=data.apply(func=lambda x : conditionalMapFitness(x.neurone,x.move), axis=1)

        else :

            data['fitness']=1

        

        # Renvoie l ajustment accumuler en donnant sense a la fitness (Normalisation de la fitness par neurones)

        accumulate_move=lambda i :(data[data.neurone==i].move*\

                                   data[data.neurone==i].fitness/(data[data.neurone==i].fitness).sum()\

                                  ).sum() 

        

        for i in data.neurone.unique().tolist(): # Pour tout les neurones qui ont besoin d etre updater 

            self.w[i]+=accumulate_move(i)

            

        return(self.w)
n_kmap=Kmap(num_neurons=90,feature_space=feature_space,r=1)
#n_kmap.w
for _ in range(100):

    n_kmap.batch_update(alpha=0.1)
plt.scatter(x, y, c=rgb, alpha=0.5,marker='.')

plt.scatter(n_kmap.w[:,0],n_kmap.w[:,1],marker='o',alpha=1)

plt.show()
#plt.scatter(x, y, c=rgb, alpha=0.5,marker='.')

plt.plot(n_kmap.w[:,0],n_kmap.w[:,1],marker='o')

plt.show()
plt.scatter(n_kmap.w[:,0],n_kmap.w[:,1],marker='o')

plt.show()