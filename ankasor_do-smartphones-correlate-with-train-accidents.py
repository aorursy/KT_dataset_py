import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import math



import statsmodels.api as sm

from statsmodels.sandbox.tools import pca



data = {

 1991: (1495, float('nan')),

 1992: (1533, float('nan')),

 1993: (1394, float('nan')),

 1994: (1113, float('nan')),

 1995: (1243, float('nan')),

 1996: (1220, float('nan')),

 1997: (1013, float('nan')),

 1998: (996, float('nan')),

 1999: (946, float('nan')),

 2000: (836, 29.8),

 2001: (790,55.7),

 2002: (771,69.8),

 2003: (813,73.0),

 2004: (564, 72.1),

 2005: (569,76.4),

 2006: (622,80.6),

 2007: (602,81.8),

 2008: (540, float('nan')),

 2009: (524,86.7),

 2010: (566,88.9),

 2011: (474,90.0),

 2012: (509,90.3),

 2013: (479, float('nan')),

 2014: (492,93.6),

 2015: (472,93.5)

 }



sc_dataset_unfall = []

sc_dataset_smartphone = []



for i in data.items():

    if not math.isnan(i[1][1]):

        sc_dataset_unfall.append(i[1][0])

        sc_dataset_smartphone.append(i[1][1])



plt.scatter(sc_dataset_smartphone, sc_dataset_unfall)



plt.title ("Smartphones and rail traffic accidents\nGermany 2000-2015")



plt.xlabel("Households with mobile phone (%)")

plt.ylabel("Accidents with injury in rail traffic")



plt.show()



tl_dataset_unfall_y = []

tl_dataset_unfall = []



tl_dataset_smartphone_y = []

tl_dataset_smartphone = []



for i in data.items():

    if not math.isnan(i[1][1]):

        tl_dataset_smartphone.append(i[1][1])

        tl_dataset_smartphone_y.append(i[0])

        

    tl_dataset_unfall.append(i[1][0])

    tl_dataset_unfall_y.append(i[0])

    





fig, ax1 = plt.subplots()

ax2 = ax1.twinx()



ax1.plot(tl_dataset_unfall_y, tl_dataset_unfall, 'r')

ax1.set_xlabel("Year")

ax1.set_ylabel("Accidents with injury in rail traffic")



ax2.plot(tl_dataset_smartphone_y, tl_dataset_smartphone, 'b')

ax2.set_ylabel("Households with mobile phone (%)")

plt.legend(handles=[mpatches.Patch(color='r', label='Accidents'), mpatches.Patch(color='b', label='Phones')], loc=3)

plt.title ("Timeline of rail traffic accidents and mobile phone ownership\nGermany 1991-2015")

plt.show()



lr_dataset_x = [] 

lr_dataset_y = []



for i in data.items():

    if not math.isnan(i[1][1]):

        lr_dataset_x.append([i[0], i[1][1]])

        lr_dataset_y.append(i[1][0])

    

lr_dataset_x = np.array(lr_dataset_x)

lr_dataset_y = np.array(lr_dataset_y)



lr_dataset_x_c = sm.add_constant(lr_dataset_x)



model = sm.OLS(lr_dataset_y, lr_dataset_x_c)

results = model.fit()

print(results.summary())



xred, fact, eva, eve  = pca(lr_dataset_x, keepdim=0)



print("==============================================================================")

print("Eigenvalues:", eva)

print ("Factor 1 Eigenvector:", eve[0])

print ("Factor 2 Eigenvector:", eve[1])

print("==============================================================================")



fact_c = sm.add_constant(fact)



model = sm.OLS(lr_dataset_y, fact_c)

results = model.fit()

print(results.summary())
