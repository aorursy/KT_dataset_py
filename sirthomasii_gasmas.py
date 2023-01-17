%matplotlib inline



import pandas as pd 

import numpy as np

from scipy.misc import electrocardiogram

from scipy.signal import find_peaks

from matplotlib import pyplot as plt



gasmas_df = pd.read_csv("../input/gasmas_pos1") 



#----------GET THE PEAKS VIA THE SCIPY SIGNAL PACKAGE---------



peaks, _ = find_peaks(gasmas_df.iloc[:,0], distance=30)

peaks_df=[]

peaks_df_cleaned=[]



for i, row in gasmas_df.iterrows():  

    

#-------- CREATE ONE ARRAY WITH ALL PEAKS - ABSORPTION REGION



    if (((i in peaks) and (i < 18000) ) or ((i in peaks) and (i > 32500)and (i < 40000))):

        peaks_df_cleaned.append(row)

        

#------- CREATE ONE ARRAY WITH ALL PEAKS + ABSORPTION REGION

        

    if (((i in peaks) and (i < 40000) )):

        peaks_df.append(row)

        

peaks_df_cleaned = pd.DataFrame(peaks_df_cleaned)

peaks_df = pd.DataFrame(peaks_df)





#----------POLYFIT PEAKS DATA - ABSORPTION REGION---------



peaks_df_index = list(range(0,(len(peaks_df_cleaned))))

peaks_df_index = peaks_df_cleaned.axes[0].tolist()

c1,c0 = np.polyfit(peaks_df_index,list(peaks_df_cleaned.iloc[:,0]), 1)



#----------POLYFIT / PEAKS DATA ---------



polyfit_vs_data = []

for i,r in peaks_df.iterrows():    

    polyfit_vs_data.append((c1*i + c0)/r)



#----------TOTAL POLYFIT ARRAY ---------



polyfit = []

for i in range(0,40000):

    polyfit.append(c1*i + c0)



polyfit = pd.DataFrame(polyfit)

polyfit_vs_data = pd.DataFrame(polyfit_vs_data)





#----------PLOTS---------



ax = gasmas_df.plot(style="g.", ms=1)

peaks_df.plot(ax=ax,style="r.")

peaks_df_cleaned.plot(ax=ax,style="b.")

ax.legend(["GASMAS Data","Peaks","Peaks for polyfit"]);

plt.show()



ax = gasmas_df.plot(style="m.", ms=1)

peaks_df.plot(ax=ax,style="x")

plt.xlim(20000,21000)

ax.legend(["GASMAS Data","Peaks"]);

plt.show()



ax = peaks_df.plot(style="m.")

polyfit.plot(ax=ax,style="r.")

ax.legend(["Peaks", "Polyfit"]);



plt.show()



ax = polyfit_vs_data.plot(style="b.")

ax.legend(["Polyfit / Data"]);



plt.show()
