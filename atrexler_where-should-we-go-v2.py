import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import pca

import pandas as pd

import numpy as np

import seaborn as sea

import math

from sklearn import neighbors 

from functools import partial

from sklearn.neighbors import NearestNeighbors as knn

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler





#read the data

oec=pd.read_csv('../input/oec.csv')

#give earth the average surface temp of 12C.

oec.loc[384,'SurfaceTempK']=283



#function to convert to decimal degrees from the polar coords provided.

def dec_degrees(ra,dec):

    

    ra_conv=[15,0.25,0.004166]

    ra_l=[float(x) for x in ra.split(' ')]

    ra_dd=np.sum([a*b for a,b in zip(ra_conv,ra_l)])

    dec_conv=[1,0.016667,0.00027778]

    dec_l=[float(x) for x in dec.split(' ')]

    dec_dd=np.sum([c*d for c,d in zip(dec_conv,dec_l)])

        

    return ra_dd,dec_dd



#iterate over the df and create decimal degree values

for idx,x in oec.iterrows():

    ra=oec.loc[idx,'RightAscension']

    dec=oec.loc[idx,'Declination']

    if (str(ra)!='nan') & (str(dec)!='nan'):

        ra_dd,dec_dd=dec_degrees(ra,dec)

    else:

        ra_dd='nan'

        dec_dd='nan'

    oec.loc[idx,'RA_dd']=ra_dd

    oec.loc[idx,'Dec_dd']=dec_dd



#plot to show all planets.  color increments over all datapoints and size shows relative distance away

#fill missing distance values with an average value of all dists.

#size is normalized to useful scale for visualization s=1000/dists

csp=plt.cm.RdBu(np.linspace(0,1,len(oec)))

fig,ax=plt.subplots()

dists=oec['DistFromSunParsec']

dists.fillna(value=np.mean(dists),inplace=True)

ax=plt.scatter(x=oec['RA_dd'],y=oec['Dec_dd'],c=csp,s=1000/dists)

plt.xlabel('right ascension (decimal degrees)')

plt.ylabel('declination (decimal degrees)')

plt.title('sky map of the exoplanets!',size=20)
sys_cols=['PlanetaryMassJpt','RadiusJpt','PeriodDays','SurfaceTempK','HostStarMassSlrMass','HostStarMetallicity','HostStarTempK','HostStarAgeGyr']

orbit_cols=['PeriodDays','TypeFlag','SemiMajorAxisAU','Eccentricity','PeriastronDeg','LongitudeDeg','AscendingNodeDeg','InclinationDeg']



all_cols=['PlanetIdentifier', 'TypeFlag', 'PlanetaryMassJpt', 'RadiusJpt',

       'PeriodDays', 'SemiMajorAxisAU', 'Eccentricity', 'PeriastronDeg',

       'LongitudeDeg', 'AscendingNodeDeg', 'InclinationDeg',

       'SurfaceTempK', 'AgeGyr', 'DiscoveryMethod', 'DiscoveryYear',

       'LastUpdated', 'RightAscension', 'Declination', 'DistFromSunParsec',

       'HostStarMassSlrMass', 'HostStarRadiusSlrRad',

       'HostStarMetallicity', 'HostStarTempK', 'HostStarAgeGyr', 'Dec_dd',

       'RA_dd']



#use sklearn kerneldensity.sample to compute random samples from the KDE

oec_sys=oec[sys_cols]

oec_orbits=oec[orbit_cols]



#define function to calc kde for a dataset and then generate random new values

def newvals(x):

    x=x[pd.notnull(x)]

    xres=x.reshape(-1,1)

    kde=neighbors.KernelDensity(kernel='gaussian').fit(xres)

    new_series=kde.sample(10000)

    new=[float(x) for x in new_series]

    return new



#build new df with et of replacement values

replacement_vals=pd.DataFrame(columns=sys_cols)

replacement_vals_o=pd.DataFrame(columns=orbit_cols)

for c,d in zip(oec_sys,oec_orbits):

    replacement_vals[c]=newvals(oec_sys[c])

    replacement_vals_o[d]=newvals(oec_orbits[d])

    

#function to return a random value from replacement values.

def r_val(x):

    if pd.isnull(x):

        new=np.random.choice(v)

        while new<=0:

            new=np.random.choice(v)

        return new

    else:

        return x



filled_oec_sys=pd.DataFrame(columns=sys_cols)

filled_oec_orbit=pd.DataFrame(columns=orbit_cols)



for e,f in zip(oec_sys,oec_orbits):

    v=replacement_vals[e]

    v_orbit=replacement_vals_o[f]

    filled_oec_sys[e]=oec_sys[e].apply(r_val) 

    filled_oec_orbit[f]=oec_orbits[f].apply(r_val)



filled_oec=pd.concat([filled_oec_sys,filled_oec_orbit],axis=1)
#scale first for proper PCA

scaled_sys=StandardScaler().fit_transform(filled_oec_sys)

pca_obj=pca.PCA(n_components=2).fit(scaled_sys)

oec_sys_pca=pca_obj.fit_transform(scaled_sys)

plt.scatter(x=oec_sys_pca[:,0],y=oec_sys_pca[:,1],s=5)

plt.xlim([-5,5])

plt.ylim([-5,5])

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.title('star system dataset')

plt.show()



scaled_orbits=StandardScaler().fit_transform(filled_oec_orbit)

pca_orbit=pca.PCA(n_components=2).fit(scaled_orbits)

oec_orbit_pca=pca_orbit.fit_transform(scaled_orbits)

plt.scatter(x=oec_orbit_pca[:,0],y=oec_orbit_pca[:,1],s=2)

plt.xlim([-5,5])

plt.ylim([-5,5])

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.title('orbital parameters')

plt.show()
#look at good systems with hab zones and see where they are in starchart.  where should we look?

#define habscore below by just summing the scores I assign to each parameter we checj.

#where its unclear what the value should be, use Sol as the reference.

def make_hb(exoplanets):

    habscore=pd.DataFrame(index=exoplanets.index,columns=['StarTemp','StarMetal','PlanetMass','Ecc','water'])

    earthref=exoplanets.loc[exoplanets['PlanetIdentifier']=='Earth']

    #want 7000K to 4000K star temp. 'StarTemp' parameter.

    habscore.loc[(exoplanets['HostStarTempK'] > 4000) & (exoplanets['HostStarTempK']<7000),'StarTemp']=1

    habscore['StarTemp'].replace(to_replace='NaN',value=0,inplace=True)

    

    #want high star metallicity.  use Sol as reference point.  'StarMetal' parameter.

    metal_sol=earthref['HostStarMetallicity'].values

    habscore.loc[(exoplanets['HostStarMetallicity'].values>=metal_sol),'StarMetal']=1

    habscore['StarMetal'].replace(to_replace='NaN',value=0,inplace=True)

    

    #want generally higher planetary mass.  lower mass means less gravity so molecules can escape more readily.

    #lower mass planets also tend to be smaller diameter so have higher SA to vol ratios, and will thusly 

    #lose heat more readily from their formation.

    #larger planets also more likely to have massive atmosphere and iron core (so thusly mag field to protect)

    #'PlanetMass' parameter.

    earthmass=earthref['PlanetaryMassJpt'].values

    habscore.loc[((exoplanets['PlanetaryMassJpt'].values>=earthmass) & (exoplanets['PlanetaryMassJpt'].values<(5*earthmass))),'PlanetMass']=1

    habscore['PlanetMass'].replace(to_replace='NaN',value=0,inplace=True)

    

    #want lower eccentricity of orbit.

    habscore.loc[(exoplanets['Eccentricity'] < 0.1),'Ecc']=1

    habscore['Ecc'].replace(to_replace='NaN',value=0,inplace=True)

    

    #finally the temperature needs to be reasonably in a range for liquid water.  complicated by the fact 

    #that its an average temperature.

    habscore.loc[(exoplanets['SurfaceTempK'] >=248) & (exoplanets['SurfaceTempK']<=398),'water']=1

    habscore['water'].replace(to_replace='NaN',value=0,inplace=True)

    

    return habscore.sum(axis=1) 



oec['habscore']=make_hb(oec)

plt.scatter(x=oec['habscore'],y=oec['DistFromSunParsec'])

plt.xlabel('habitability score')

plt.ylabel('distance from sun in parsecs')



#define destinations, exoplanets with high habscores.  plot all.

destinations=oec.loc[oec['habscore']==4]

f2=plt.figure()

plt.title('high habscore exoplanets (>=4)',size=15,y=1.05)

plt.scatter(x=destinations['DistFromSunParsec'],y=destinations['SurfaceTempK'])

plt.ylabel('Surface Temp (K)')

plt.xlabel('Distance from Sun (parsec)')

plt.show()
#next we plot the high habscores overlaid on the skymap, where should we be looking?

f3=plt.figure()

plt.scatter(x=oec['RA_dd'],y=oec['Dec_dd'],c='blue')

plt.scatter(x=destinations['RA_dd'],y=destinations['Dec_dd'],c='orange',s=50)

plt.xlabel('right ascension (decimal degrees)')

plt.ylabel('declination (decimal degrees)')

plt.title('where should we be looking?',size=20,y=1.05)
scaled_orbits=StandardScaler().fit_transform(filled_oec_orbit)

pca_orbit=pca.PCA(n_components=2).fit(scaled_orbits)

oec_orbit_pca=pca_orbit.fit_transform(scaled_orbits)

plt.scatter(x=oec_orbit_pca[:,0],y=oec_orbit_pca[:,1],s=2)

plt.scatter(oec_orbit_pca[destinations.index.values,0],oec_orbit_pca[destinations.index.values,1],c='r')

plt.xlim([-5,5])

plt.ylim([-5,5])
#next focus on stuff close to use with useful temperature range

f=plt.figure(figsize=[5,5])

plt.title('likely destinations for interstellar exploration',size=15,y=1.05)

xvals=destinations.loc[(destinations['DistFromSunParsec']<200) & (destinations['SurfaceTempK']<500),'DistFromSunParsec'].values

yvals=destinations.loc[(destinations['DistFromSunParsec']<200) & (destinations['SurfaceTempK']<500),'SurfaceTempK'].values

labels=destinations.loc[(destinations['DistFromSunParsec']<200) & (destinations['SurfaceTempK']<500),'PlanetIdentifier'].values

plt.scatter(x=xvals,y=yvals)

plt.xlim([-50,200])

plt.ylim([200,450])

plt.ylabel('Surface Temp (K)')

plt.xlabel('Distance from Sun (parsec)')

for xv,yv,l in zip(xvals,yvals,labels):

    plt.annotate(s=l,xy=(xv,yv))

plt.show()