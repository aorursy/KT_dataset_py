import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.cluster import KMeans

import matplotlib.colors

%matplotlib inline
pd.set_option('display.max_columns', None)  

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)
df=pd.read_csv("/kaggle/input/open-exoplanet-catalogue/oec.csv")
df.head()
df.describe()
df.groupby(['DiscoveryMethod', 'DiscoveryYear']).size().unstack().T.fillna(0).astype(int).sort_values(by=['DiscoveryYear'])
numericCols=['PlanetaryMassJpt', 'RadiusJpt','PeriodDays', 'SemiMajorAxisAU', 'Eccentricity',

                 'SurfaceTempK','HostStarMassSlrMass', 'HostStarRadiusSlrRad',

                 'HostStarMetallicity','HostStarTempK','HostStarAgeGyr']

fig, ax = plt.subplots(figsize=(10,10))

ax=sns.heatmap(df[numericCols].corr(), annot=True, linewidths=.5,square=True)

print(ax.get_ylim())

ax.set_ylim(10.0, 0) #Due to few cropping issues in seaborn
df.isna().mean().sort_values(ascending=False).head(15)*100
fig, axes = plt.subplots(2, 2,figsize=(18,12))

axes[0,0].hist(df["PlanetaryMassJpt"],bins=100)

axes[0,1].hist(df["RadiusJpt"],bins=100)

axes[1,0].hist(df["PeriodDays"],bins=100)

axes[1,1].hist(df["SemiMajorAxisAU"],bins=100)

plt.show()
df=df.dropna(subset=['PlanetaryMassJpt', 'RadiusJpt'])#Remove null values

plt.scatter(df["RadiusJpt"],df["PlanetaryMassJpt"])
df["RadiusJpt"].quantile(0.999)
dfremoved=df[df["RadiusJpt"]<df["RadiusJpt"].quantile(0.999)]

dfremoved=dfremoved[dfremoved["PlanetaryMassJpt"]<dfremoved["RadiusJpt"].quantile(0.999)]

plt.scatter(dfremoved["RadiusJpt"],dfremoved["PlanetaryMassJpt"])
df["PlanetaryMassJptlog"]=df["PlanetaryMassJpt"].apply(np.log)

df["RadiusJptlog"]=df["RadiusJpt"].apply(np.log)



plt.scatter(df["RadiusJptlog"],df["PlanetaryMassJptlog"])
linearRegressor = LinearRegression()
xTrain=df[["RadiusJptlog"]]

yTrain=df["PlanetaryMassJptlog"]
linearRegressor.fit(xTrain,yTrain)
plt.scatter(xTrain, yTrain, color = 'red')

plt.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')

plt.title('Radius vs Mass of planets')

plt.xlabel('log(Radius)')

plt.ylabel('log(Mass)')

plt.show()
linearRegressor.score(xTrain,yTrain)
linearRegressor.coef_
linearRegressor.intercept_
km = KMeans(

    n_clusters=2, init='random',

    n_init=10, max_iter=300, 

    tol=1e-04, random_state=0

)
X=np.array(df[["RadiusJptlog","PlanetaryMassJptlog"]])

X=X.reshape(-1,2)
km.fit(X)

y_km = km.fit_predict(X)

plt.scatter(

    X[y_km == 0, 0], X[y_km == 0, 1],

    s=50, c='lightgreen',

    edgecolor='black',

    label='gas giants'

)



plt.scatter(

    X[y_km == 1, 0], X[y_km == 1, 1],

    s=50, c='orange',

    edgecolor='black',

    label='rocky planets'

)

plt.title('Radius vs Mass of planets')

plt.xlabel('log(Radius)')

plt.ylabel('log(Mass)')



plt.legend(scatterpoints=1)

plt.show()
df["cluster"]=y_km
df["MassRadiusLogRatio"]=df["PlanetaryMassJptlog"]/df["RadiusJptlog"]
df.groupby("cluster").agg({"MassRadiusLogRatio":"mean"})
df["MassRadiusCubeRatio"]=df["PlanetaryMassJpt"]/df["RadiusJpt"]**3
df["MassRadiusCubeRatio"].hist()
df.groupby("cluster").agg({"MassRadiusCubeRatio":"mean"})
plt.boxplot([df[df["cluster"]==0]["MassRadiusCubeRatio"],df[df["cluster"]==1]["MassRadiusCubeRatio"]])

positions = (1, 2)

labels = ("Rocky planets", "Gas giants")

plt.ylabel('Density')

plt.xticks(positions, labels)

plt.show()
df=pd.read_csv("/kaggle/input/open-exoplanet-catalogue/oec.csv")
plt.scatter(df["SemiMajorAxisAU"],df["PeriodDays"])
df=df.dropna(subset=['SemiMajorAxisAU', 'PeriodDays']) #Removing null values
df["SemiMajorAxisAUlog"]=df["SemiMajorAxisAU"].apply(np.log)

df["PeriodDayslog"]=df["PeriodDays"].apply(np.log)

plt.scatter(df["SemiMajorAxisAUlog"],df["PeriodDayslog"])
linearRegressor = LinearRegression()
xTrain=df[["SemiMajorAxisAUlog"]]

yTrain=df["PeriodDayslog"]
yTrain.shape
xTrain.describe()
yTrain.describe()
linearRegressor.fit(xTrain,yTrain)
plt.scatter(xTrain, yTrain, color = 'blue')

plt.plot(xTrain, linearRegressor.predict(xTrain), color = 'red',linewidth=5.0)

plt.title('Semi major axis vs time of orbit')

plt.xlabel('log(Semimajor axis)')

plt.ylabel('log(Time of orbit)')

plt.show()
linearRegressor.score(xTrain,yTrain)
linearRegressor.coef_
linearRegressor.intercept_
df.head()
def linreg(x,y):

    x=np.array(x).reshape(-1,1)

    y=np.array(y)

    linearRegressor = LinearRegression()

    linearRegressor.fit(x,y)

    return len(y),linearRegressor.score(x,y),linearRegressor.coef_[0],linearRegressor.intercept_

    
ra=""

dec=""

x=[]

y=[]

counts=[]

scores=[]

slopes=[]

intercepts=[]

for row in df.itertuples(index=True, name='Pandas'):

    if(ra==getattr(row, "RightAscension") and dec==getattr(row, "Declination")):

        x.append(getattr(row, "SemiMajorAxisAUlog"))

        y.append(getattr(row, "PeriodDayslog"))

    else:

        if(len(x)>=2):

            count,score,slope,intercept=linreg(x,y)

            counts.append(count)

            scores.append(score)

            slopes.append(slope)

            intercepts.append(intercept)



        ra=getattr(row, "RightAscension")

        dec=getattr(row, "Declination")

        x=[getattr(row, "SemiMajorAxisAUlog")]

        y=[getattr(row, "PeriodDayslog")]

        

linreg_results=pd.DataFrame({"Count":counts,"Score":scores,"Slope":slopes,"Intercept":intercepts})  
linreg_results.head()
linreg_results.describe()
plt.boxplot(linreg_results["Slope"])

plt.show()
linreg_results=linreg_results[linreg_results["Slope"]<6]

linreg_results.describe()
plt.hist(linreg_results["Slope"])

plt.show()
df["Luminosity"]=df["HostStarRadiusSlrRad"]**2*df["HostStarTempK"]**4

df["Luminositylog"]=np.log(df["Luminosity"])
cmap = matplotlib.colors.ListedColormap(["blue","blue","blue","blue","yellow","darkorange","red","red","red"][::-1])
fig = plt.figure(figsize=(5, 5))

plt.scatter(df["HostStarTempK"],df["Luminositylog"],c=df["HostStarTempK"],s=100*df["HostStarRadiusSlrRad"], cmap="coolwarm_r",edgecolor='black', linewidth=0.2)

plt.gca().invert_xaxis()

plt.title('HR Diagram')

plt.xlabel('HostStarTempK')

plt.ylabel('log(luminosity)')

plt.show()
df=df[df["HostStarTempK"]<25000]
fig = plt.figure(figsize=(18, 12))

points=plt.scatter(df["HostStarTempK"],df["Luminositylog"],c=df["HostStarTempK"],s=100*df["HostStarRadiusSlrRad"], cmap=cmap,edgecolor='black', linewidth=0.5)

plt.colorbar(points)

plt.gca().invert_xaxis()

plt.title('HR Diagram')

plt.xlabel('HostStarTempK')

plt.ylabel('log(luminosity)')

plt.show()
df['Luminosity'] = df['HostStarRadiusSlrRad']**2  * (df['HostStarTempK']/5777)**4
#add habitable zone boundaries

df['HabZoneOut'] = np.sqrt(df['Luminosity']/0.53)

df['HabZoneIn'] = np.sqrt(df['Luminosity']/1.1)
habitable_zone=df[(df["SemiMajorAxisAU"]>df["HabZoneIn"]) & (df["SemiMajorAxisAU"]<df["HabZoneOut"])]

habitable_zone
habitable_zone=habitable_zone[(habitable_zone["PlanetaryMassJpt"]>0.0015) & (habitable_zone["PlanetaryMassJpt"]<0.03)]

habitable_zone
habitable_zone[["PlanetIdentifier","DistFromSunParsec"]].sort_values(by=["DistFromSunParsec"])
df[df["ListsPlanetIsOn"]=="Solar System"]