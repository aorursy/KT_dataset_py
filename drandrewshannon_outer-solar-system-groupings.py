#This first cell will just load and clean the data, so you can play with single clustering algorithms in other cells



#My goal with this sheet is really just to test out automatic grouping algorithms on outer solar system bodies, to see what they do on it.

#Frankly, I'm not optimistic they can do better than humans already have, except the extended vs detached thing, perhaps





#Choice of algorithms to test out follows from https://machinelearningmastery.com/clustering-algorithms-with-python/

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy # linear algebra

import math #Regular, god fearing math

import pandas # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pyplot #Iffen I wants to plt, aye.





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



ossos = pandas.read_table('../input/outer-solar-system-origins-survey/t3char.dat', header=None, delim_whitespace=True,

                          names=['Class', 'Subclass', 'j', 'k', 'Status', 'ID', 'Magnitude', 'Error Magnitude',

                                'Filter', 'H', 'Distance', 'Error Distance', 'Number of Observations', 'Length of Arc', 

                                'Mean Residuals, RA', 'Mean Residuals, DEC', 'Maximum Residuals, RA', 'Maximum Residuals, DEC', 

                                'a', 'sigma a', 'e', 'sigma e', 'i', 'sigma i', 'Omega', 'Error Omega', 'small omega', 

                                'Error small omega', 'time of pericentre', 'Error tile of pericentre', 'RA discovery', 

                                'DEC discovery', 'JD discovery', 'n', 'MPC name'])

#print(type(ossos))

#for column in ossos.columns:

#    print(column)

#print(ossos.head())

semi = ossos['a'].tolist()

ecc = ossos['e'].tolist()

inc = ossos['i'].tolist()





STANDARD_A = 1.0 #My default choice is 50, but I don't know a choice that makes it focus on resonant pops

aeiarray = numpy.dstack((numpy.array([axis/STANDARD_A for axis in semi]), numpy.array(ecc), numpy.array([math.radians(i) for i in inc])))



aeiarray = aeiarray.reshape(840,3)



aearray = numpy.dstack((numpy.array(semi), numpy.array(ecc)))

                       

print(type(aearray))

print(aearray.shape)



aearray = aearray.reshape(840,2)



#ossos.plot.scatter('a', 'e', figsize =(8,8))









#Okay, let's just try something, see what happens

#Machine learning models are typically about experimentation, and knowing the typical results people find

from sklearn.cluster import AffinityPropagation

model = AffinityPropagation(damping=0.8, max_iter=10000)



model.fit(aeiarray)



pops = model.predict(aeiarray)

clusters = numpy.unique(pops)





pyplot.xlim(28,58)

for cluster in clusters:

    underlying = numpy.where(pops == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 1])

pyplot.show()



pyplot.xlim(28,58)

for cluster in clusters:

    underlying = numpy.where(pops == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 2])

pyplot.show()
from sklearn.cluster import AgglomerativeClustering

model_AC = AgglomerativeClustering(n_clusters=45)

pops_AC = model_AC.fit_predict(aeiarray)



clusters_AC = numpy.unique(pops_AC)

pyplot.xlim(28,58)

for cluster in clusters_AC:

    underlying = numpy.where(pops_AC == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 1])

pyplot.show()



pyplot.xlim(28,58)

for cluster in clusters_AC:

    underlying = numpy.where(pops_AC == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 2])

pyplot.show()
from sklearn.cluster import Birch

model_B = Birch(threshold=0.1, n_clusters=20)

model_B.fit(aeiarray)

pops_B = model_B.predict(aeiarray)



clusters_B = numpy.unique(pops_B)

pyplot.xlim(28,58)

for cluster in clusters_B:

    underlying = numpy.where(pops_B == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 1])

pyplot.show()



pyplot.xlim(28,58)

for cluster in clusters_B:

    underlying = numpy.where(pops_B == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 2])

pyplot.show()
from sklearn.cluster import DBSCAN

model_DB = DBSCAN(eps=0.03, min_samples=12)

#0.03, 12 pulls out colds 

#0.02, 12 - the kernel ?

pops_DB = model_DB.fit_predict(aeiarray)



clusters_DB = numpy.unique(pops_DB)

pyplot.xlim(28,58)

for cluster in clusters_DB:

    underlying = numpy.where(pops_DB == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 1])

pyplot.show()



pyplot.xlim(28,58)

for cluster in clusters_DB:

    underlying = numpy.where(pops_DB == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 2])

pyplot.show()
from sklearn.cluster import KMeans



model_KM = KMeans(n_clusters=10)

model_KM.fit(aeiarray)



pops_KM = model_KM.predict(aeiarray)



clusters_KM = numpy.unique(pops_KM)

pyplot.xlim(28,58)

for cluster in clusters_KM:

    underlying = numpy.where(pops_KM == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 1])

pyplot.show()



pyplot.xlim(28,58)

for cluster in clusters_KM:

    underlying = numpy.where(pops_KM == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 2])

pyplot.show()
from sklearn.cluster import MiniBatchKMeans

model_MBKM = MiniBatchKMeans(n_clusters=13)

model_MBKM.fit(aeiarray)



#4 sees scattered

#8 sees colds, but otherwise weird

#9 sorta sees 2:1 and 5:2





pops_MBKM = model_MBKM.predict(aeiarray)



clusters_MBKM = numpy.unique(pops_MBKM)

pyplot.xlim(28,58)

for cluster in clusters_MBKM:

    underlying = numpy.where(pops_MBKM == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 1])

pyplot.show()



pyplot.xlim(28,58)

for cluster in clusters_MBKM:

    underlying = numpy.where(pops_MBKM == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 2])

pyplot.show()
from sklearn.cluster import MeanShift

model_MS = MeanShift()

pops_MS = model_MS.fit_predict(aeiarray)



#Dunno, can't get much out of this?



clusters_MS = numpy.unique(pops_MS)

pyplot.xlim(28,58)

for cluster in clusters_MS:

    underlying = numpy.where(pops_MS == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 1])

pyplot.show()



pyplot.xlim(28,58)

for cluster in clusters_MS:

    underlying = numpy.where(pops_MS == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 2])

pyplot.show()
from sklearn.cluster import OPTICS



model_OPT = OPTICS(eps=0.7, min_samples=8)

pops_OPT = model_OPT.fit_predict(aeiarray)



#I can never make this do much - it wants really tight groupings ... 

#Collisional families?  But they're too big.  Kernel and ... other kernels ?



clusters_OPT = numpy.unique(pops_OPT)

pyplot.xlim(28,58)

for cluster in clusters_OPT:

    underlying = numpy.where(pops_OPT == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 1])

pyplot.show()



pyplot.xlim(28,58)

for cluster in clusters_OPT:

    underlying = numpy.where(pops_OPT == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 2])

pyplot.show()
from sklearn.cluster import SpectralClustering

model_SC = SpectralClustering(n_clusters=10)

pops_SC = model_SC.fit_predict(aeiarray)



#Need at least 16 to see anything, this method really likes to make high a bodies the non-belongers



clusters_SC = numpy.unique(pops_SC)

pyplot.xlim(28,58)

for cluster in clusters_SC:

    underlying = numpy.where(pops_SC == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 1])

pyplot.show()



pyplot.xlim(28,58)

for cluster in clusters_SC:

    underlying = numpy.where(pops_SC == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 2])

pyplot.show()

from sklearn.mixture import GaussianMixture



model_G = GaussianMixture(n_components=20)

#4 is an interesting choice, seems to find Cold Classicals

#6 finds cold classicals, maybe hots from scattered a bit ?

#8 finds colds, hots, though plutinos and 4/3, 5:4 are in hots too



model_G.fit(aeiarray)

# assign a cluster to each example

pops_G = model_G.predict(aeiarray)



clusters_G = numpy.unique(pops_G)

pyplot.xlim(28,58)

for cluster in clusters_G:

    underlying = numpy.where(pops_G == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 1])

pyplot.show()



pyplot.xlim(28,58)

for cluster in clusters_G:

    underlying = numpy.where(pops_G == cluster)

    pyplot.scatter(aeiarray[underlying, 0]*STANDARD_A, aeiarray[underlying, 2])

pyplot.show()