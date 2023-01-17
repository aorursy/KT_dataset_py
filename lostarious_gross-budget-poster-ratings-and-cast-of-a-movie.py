#This is just some practice in order to get used to kernels for me 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline



from subprocess import check_output

data = pd.read_csv("../input/movie_metadata.csv",sep=",",usecols=["facenumber_in_poster","gross","budget","imdb_score","cast_total_facebook_likes"])

gross= np.array(data["gross"],dtype='float64')

budget = np.array(data["budget"],dtype='float64')

imdb = np.array(data["imdb_score"],dtype='float64')

fnum = np.array(data["facenumber_in_poster"],dtype='float64')

likes = np.array(data["cast_total_facebook_likes"])

plt.xlim((0,150000))

plt.ylim((0,500000000))

plt.plot(likes,gross,"*")

plt.xlabel("Total Facebook likes of the cast")

plt.ylabel("Gross of the movie")

plt.show()



plt.plot(budget,gross,".")

plt.xlabel("Budget of the movie")

plt.ylabel("Gross of the movie")

plt.xlim((0,260000000))

plt.ylim((0,400000000))

plt.show()

fiveOL = []

fiveSeven = []

sevenTen = []

counter = 0

counts1 = [0,0,0,0,0]

counts2 = [0,0,0,0,0]

counts3 = [0,0,0,0,0]

for x in np.nditer(imdb):

    

    if(x<=5.):

        fiveOL.append(fnum[counter])

        if(fnum[counter]==0):

            counts1[0]+=1

        if(fnum[counter]==1):

            counts1[1]+=1

        if(fnum[counter]==2):

            counts1[2]+=1

        if(fnum[counter]==3):

            counts1[3]+=1

        if(fnum[counter]>=4):

            counts1[4]+=1

            

            

    if(5<x<=7):

        fiveSeven.append(fnum[counter])

        

        if(fnum[counter]==0):

            counts2[0]+=1

        if(fnum[counter]==1):

            counts2[1]+=1

        if(fnum[counter]==2):

            counts2[2]+=1

        if(fnum[counter]==3):

            counts2[3]+=1

        if(fnum[counter]>=4):

            counts2[4]+=1

    if(7<x):

        sevenTen.append(fnum[counter])

        if(fnum[counter]==0):

            counts3[0]+=1

        if(fnum[counter]==1):

            counts3[1]+=1

        if(fnum[counter]==2):

            counts3[2]+=1

        if(fnum[counter]==3):

            counts3[3]+=1

        if(fnum[counter]>=4):

            counts3[4]+=1

    counter+=1

        

        

plt.axis('Equal')

plt.title('Number of faces in the poster of movies rated 0-5 in IMDB')

plt.pie(counts1,labels=["Zero","One","Two","Three","Four+"],shadow = True, autopct='%1.1f%%',startangle = 90)

plt.show()



plt.axis('Equal')

plt.title('Number of faces in the poster of movies rated 5-7 in IMDB')

plt.pie(counts2,labels=["Zero","One","Two","Three","Four+"],shadow = True, autopct='%1.1f%%',startangle = 90)

plt.show()



plt.axis('Equal')

plt.title('Number of faces in the poster of movies rated 7-10 in IMDB')

plt.pie(counts3,labels=["Zero","One","Two","Three","Four+"],shadow = True, autopct='%1.1f%%',startangle = 90)

plt.show()

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.