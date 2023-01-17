
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from datetime import timedelta
plt.style.use("ggplot")
# archives
caract=pd.read_csv("../input/caracteristics.csv", encoding="latin1")
holiday=pd.read_csv("../input/holidays.csv", encoding="latin1")
places=pd.read_csv("../input/places.csv", encoding="latin1")
users=pd.read_csv("../input/users.csv", encoding="latin1")
vehicles=pd.read_csv("../input/vehicles.csv", encoding="latin1")

# Date of the accidents
dia=[]
for i in tqdm(range(0,len(caract))):
    dia.append(pd.datetime((caract.iloc[i,1]+2000),caract.iloc[i,2],caract.iloc[i,3]))
caract["Data"]=dia
# Period of the accident
def periodo(x):
    if x< 600:
        return "Dawn"
    elif x<1600:
        return "Day Time"
    elif x<1900:
        return "Twilight"
    else:
        return "Night"
caract["period"]=caract["hrmn"].apply(lambda x: periodo(x))
# Selecting just a few columns
caract=caract.loc[:,["Num_Acc", "lum","agg", "int", "atm", "col", "Data", "period"]]
# Finding weekday
weekend=[]
for i in tqdm(range(0,len(caract))):
    dia=caract.iloc[i,6]
    if int(dia.weekday()) in [5,6]:
        weekend.append("weekend")
    else:
        weekend.append("work day")
caract["weekend"]=weekend
# Puting holidays proximity into the DF Characteristics
# Here we want to identify accidents that happened at the limits of 1 day (after or before) a holiday
holiday.ds=holiday.ds.apply(lambda x: pd.to_datetime(x))
listadias=[]
for i in tqdm(range(0, len(holiday))):
    listadias.append(holiday.iloc[i,0]-timedelta(days=1))
    listadias.append(holiday.iloc[i,0])
    listadias.append(holiday.iloc[i,0]+timedelta(days=1))

caract["feriado"]=caract["Data"].apply(lambda x: "holiday" if x in listadias else "non holiday")
    
# Places: choosing arbitrary some variables
places=places.loc[:,["Num_Acc","catr", "circ", "plan", "surf"]]

# Merging in "caract DataFrame"
caract=pd.merge(caract, places,on="Num_Acc")
# Let´s identify accidents with severe victims
users=users[users["grav"].isin([2,3])]
acidentes_graves=users.Num_Acc.unique()
caract["grave"]=caract.Num_Acc.apply(lambda x: "Serious Victims" if x in acidentes_graves else "No Victims")
# Let´s put out the "NaN" rows
caract=caract.dropna()
# Putting names in "lum"
key={1:"Full day", 2: "Twilight or dawn", 3:"Night without public lighting",
        4:"Night with public lighting", 5:"Night with public lighting on"}
caract.lum=caract.lum.apply(lambda x: key[x])
# Putting names in "agg"
key={1:"Out of agglomeration",2:"Building areas"}
caract["agg"]=caract["agg"].apply(lambda x: key[x])
# Putting names in "int"
caract["int"]=caract["int"].apply(lambda x: 9 if x==0 else x)
key={9:'Other intersection', 8:'Level crossing', 7:'Place', 6:'Giratory', 
     5:'Intersection with more than 4 branches', 4:'Intersection in Y', 
     3:'Intersection in T', 2:'Intersection in X', 1:'Out of intersection'}
caract["int"]=caract["int"].apply(lambda x: key[x])
# Putting names in "atm"
key={9:'Other', 8:'Cloudy weather', 7:'Dazzling weather', 6:'Strong wind - storm', 
     5:'Fog - smoke', 4:'Snow - hail', 3:'Heavy rain', 
     2:'Light rain', 1:'Normal'}
caract["atm"]=caract["atm"].apply(lambda x: key[x])
# Putting names in "col"
key={7:'Without collision', 6:'Other collision', 5:'Three or more vehicles - multiple collisions', 
     4:'Three vehicles and more - in chain', 3:'Two vehicles - by the side', 
     2:'Two vehicles - from the rear', 1:'Two vehicles - frontal'}
caract["col"]=caract["col"].apply(lambda x: key[x])
# Putting names in "catr"
key={9:'other', 6:'Parking lot open to public traffic', 5:'Off public network', 
     4:'Communal Way', 3:'Departmental Road', 2:'National Road', 1:'Highway'}
caract["catr"]=caract["catr"].apply(lambda x: key[x])
# Putting names in "circ"
key={4:'With variable assignment channels', 3:'Separated carriageways', 2:'Bidirectional', 1:'One way', 0:"Other"}
caract["circ"]=caract["circ"].apply(lambda x: key[x])
# Putting names in "plan"
key={4:'In "S"', 3:'Curved right', 2:'Curved on the left', 1:'Straight part', 0:"ignored"}
caract["plan"]=caract["plan"].apply(lambda x: key[x])
# Putting names in "surf"
key={9:'other', 8:'fat - oil', 7:'icy', 6:'mud', 5:'snow', 4:'flooded', 3:'puddles', 2:'wet', 1:'normal', 0:"Ignored"}
caract["surf"]=caract["surf"].apply(lambda x: key[x])
# Let´s change columns names for more comprehensible ones
new_names={"lum":"luminousity", "agg":"agglomeration", "int":"type Intersection", "atm":"atmosphere",
          "feriado":"holiday", "catr":"type of road", "circ":"circulation", "plan":"shape", "surf":"surface", "grave":"severity"}
caract=caract.rename(columns=new_names)
# We´ll use just a sample for clustering... For practical computational issues.
sample=caract.sample(n=30000)

# Let´s drop the columns "Num_Acc", "Data", "severity", calling the new DF by "X"
X=sample.drop(columns=["Num_Acc", "Data", "severity"])
# We´ll use the "cost" statistics to choose the best number of clusters
from kmodes.kmodes import KModes

clusters=[]
costs=[]

for i in tqdm(range(1,31)):
    km=KModes(n_clusters=i)
    km=km.fit(X)
    
    clusters.append(i)
    costs.append(km.cost_)

# Plotting graph
plt.figure()
plt.plot(clusters, costs)
plt.title("Costs (Inertia) of Clustering")
plt.ylabel("Costs")
plt.xlabel("Number of Clusters")
plt.xticks(np.arange(1,30,2))
plt.show()
# Modeling the clusters with the sample (n=30.000)
km=KModes(n_clusters=12)
clusters=km.fit_predict(X)
sample["Cluster"]=clusters
freq=sample.Cluster.value_counts(normalize=True, ascending=True)
freq.plot(kind="barh", color="Blue")
plt.title("Relative Frequencies of the Clusters")
plt.xlabel("Relative Frequencies")
plt.ylabel("Cluster Label")
plt.show()
# Selecting just the accidents with severe victims
severe_accidents=sample[sample["severity"]=="Serious Victims"]
freq=severe_accidents.Cluster.value_counts(ascending=True)
freq.plot(kind="barh", color="Red")
plt.title("Absolute Number of Accidents With Severe Victims")
plt.xlabel("Total Accidents")
plt.ylabel("Cluster Label")
plt.show()
clusters=np.arange(0,12,1)
total=[]
total_injured=[]
risk=[]


for i in clusters:
    parcial=sample[sample["Cluster"]==i]
    total.append(len(parcial))
    total_injured.append(len(parcial[parcial["severity"]=="Serious Victims"]))
    risk.append(len(parcial[parcial["severity"]=="Serious Victims"])/len(parcial))

df_risk=pd.DataFrame({"Cluster":clusters, "Total Accidents":total, "Total w/ Serious Victims":total_injured,
                     "Risk (%)":risk})
df_risk=df_risk.set_index("Cluster")
df_risk=df_risk.sort_values("Total w/ Serious Victims", ascending=False)
df_risk
# Selecting the variables of interest
cluster_selected=2
interest=["luminousity", "agglomeration","type Intersection", "atmosphere", "col", 
          "period", "weekend", "holiday", "type of road", "circulation", "shape"]

# Spliting two databases: cluster and non-cluster
db_cluster=sample[sample["Cluster"]==cluster_selected]
db_noncluster=sample[sample["Cluster"]!=cluster_selected]

# Creating a DataFrame to compare cluster and non-cluster in each variable
for i in interest:
    categories=sample[i].unique()
    perc_cluster=[]
    perc_noncluster=[]
    
    for x in categories:
        perc_cluster.append(len(db_cluster[db_cluster[i]==x])/len(db_cluster))
        perc_noncluster.append(len(db_noncluster[db_noncluster[i]==x])/len(db_noncluster))
    
    # Creating a DataFrame for chart ploting:
    df_graphic=pd.DataFrame({"Categories":categories, "Cluster {}".format(cluster_selected):perc_cluster, 
                             "Others":perc_noncluster})
    df_graphic=df_graphic.set_index("Categories")
    
    #Chart
    plt.figure()
    df_graphic.plot(kind="barh")
    plt.title("Variable {}, in Cluster {}".format(i,cluster_selected))
    plt.xlabel("Relative Frequency")
    plt.show()    
        
