import numpy as np

def MORITA_MODEL(G,x,y,theta,delta,k_avg):

    a=0

    popravek=0

    raz=0

    z=0

    i=0

    j=0

    

    sort=np.zeros([N],int)    

    d=np.zeros([N,N],float)

    raz=np.zeros([N,N],float)# razdalja med i-tim in j-tim delcem

    pf=np.zeros([N,N],float)# produkkt fi times fj, simetricna matrika!!!

    korelacijska=np.zeros([N,N],float)

    stpovezav=np.zeros([N],float)



    #x=np.zeros([N],float)#x lega i-tega delca

    #y=np.zeros([N],float)#y lega i-tega delcaf=zeros([N],float)#fitness value

    ime=np.zeros([N,N],float)

    f=np.zeros([N],float)

    lege=np.zeros([N,3],float)

    temp_k=np.zeros([N],float)

    b=2.5



    for i in range(N):  

        f[i]=pow((float(i)+1)/float(N),1/(1-float(b)))

    

    for i in range(N):

        x[i]=random.uniform(0,1)

        y[i]=random.uniform(0,1)

        G.add_node(i,pos=(x[i],y[i]))

        

    for i in range(N):

        for j in range(i+1,N,1):



            raz[i,j]=np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)

            raz[j,i]=raz[i,j]

            pf[i,j]=f[i]*f[j]

            ime[i,j]=pf[i][j]/pow(raz[i,j],delta)

            ime[j,i]=ime[i,j]

    while z<2000:

        popravek=0

        a=0

        for i in range(N):

            stpovezav[i]=0

        for i in range(N):

            for j in range(i+1,N,1):

                if i==j:

                    korelacijska[i][j]=0

                    

                if i!=j:

                    if ime[i][j]>theta:

                        korelacijska[i][j]=1

                        korelacijska[j][i]=1

                        stpovezav[i]+=1

                        stpovezav[j]+=1

                    else:

                        korelacijska[i][j]=0

                        korelacijska[j][i]=0

                        

        for i in range(N):

            popravek+=stpovezav[i]



        if float(popravek)/float(N)<k_avg+.05:

            theta=theta/pow(1.1,0.5)

        if float(popravek)/float(N)>k_avg-.05:

            theta=theta*pow(1.12,0.5)

        raz+=1

        z+=1

        if float(popravek)/float(N)<(k_avg+0.05) and float(popravek)/float(N)>(k_avg-0.05):

            z+=20000

    for i in range(N):

        for j in range(i+1,N,1):

            if korelacijska[i][j]==1:

                d[i][j]=1

                G.add_edge(i,j)

                G.add_edge(j,i)
import networkx as nx

import matplotlib.pyplot as plt

import random



N=500

delta=0.1

theta=5

k_avg=5.0



x=np.zeros(N,float)

y=np.zeros(N,float)

for i in range(N):

    x[i]=random.uniform(0,1)

    y[i]=random.uniform(0,1)



while delta<5.2:

    op=np.zeros(N,float)

    G=nx.Graph()

    MORITA_MODEL(G,x,y,theta,delta,k_avg)

    pos=nx.get_node_attributes(G,'pos')

    ss=[]

    for i in G:

        ss.append(G.degree(i)*3)

    nx.draw(G,pos,node_size=ss,node_color='blue')

    plt.title("delta = %.1f"%(delta))

    plt.show()

    delta+=1.0