import pandas as pd
data = pd.read_csv("../input/FastFoodRestaurants.csv")
data.head(10)
data["name"].value_counts().head(10), len(data["name"].unique())
data = data[["city", "latitude", "longitude", "name", "province"]]

data["name"] = data["name"].str.upper()
data["name"] = data["name"].str.replace(" ", "")
data["name"] = data["name"].str.replace("'", "")
data["name"] = data["name"].str.replace("’", "")
data["name"] = data["name"].str.replace("-", "")
data["name"] = data["name"].str.replace(".", "")
data["name"] = data["name"].str.replace("/", "")
data["name"] = data["name"].str.replace("!", "")
data["name"] = data["name"].str.replace("AND", "&")
#data["name"] = data["name"].str.replace("&", "")
#sorted(data["name"].unique())

prev = " "
majornames = []
for i in sorted(data["name"]):
    if prev in i:
        majornames.append(prev)
        continue
    prev = i

majornames = list(sorted(set(majornames)))

def replace(origin):
    for major in majornames:
        if major in origin:
            return major
    return origin
    
data["name"] = data.apply(lambda row: replace(row["name"]), axis=1)

data.head(10)
data["name"].value_counts().head(10), len(data["name"].unique())
import matplotlib.pyplot as plt
%matplotlib inline

data["name"].value_counts().plot(kind="box", figsize=(5,5))

plt.title("Number of stores")
data["name"].value_counts()[data["name"].value_counts()>10].plot(kind="bar", figsize=(20, 5), color="cornflowerblue")
plt.ylabel("number of stores (>10)")
data["province"].value_counts().plot(kind="bar", figsize=(20, 5), color="cornflowerblue")
plt.ylabel("number of stores")
data2 = data.copy()
tmp = data["name"].value_counts()[data["name"].value_counts()>100].index.tolist()

data2["name"] = data["name"].apply(lambda elm: elm if elm in tmp else "")
data2 = data2[data2["name"]!=""]
feature = pd.get_dummies(data2.drop("city", axis=1))

feature["latitude"] = feature["latitude"]/feature["latitude"].max()
feature["longitude"] = feature["longitude"]/feature["longitude"].min()

from sklearn.manifold import Isomap

iso = Isomap(n_neighbors=30, n_components=2)
pos = iso.fit_transform(feature.T)
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(20, 20))

#plt.scatter(pos[:, 0], pos[:, 1])

for x, y, t in zip(pos[:, 0], pos[:, 1], feature.columns.tolist()):
    if "name_" in t:
        plt.scatter(x, y)
        plt.text(x=x, y=y, s=t)
    
#    if "province_" in t:
#        plt.scatter(x, y)
#        plt.text(x=x, y=y, s=t)
        
#    if "tude" in t:
#        continue    
#    plt.scatter(x, y)
#    plt.text(x=x, y=y, s=t)

plt.title("Isomap (store)")
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(20, 20))

#plt.scatter(pos[:, 0], pos[:, 1])

for x, y, t in zip(pos[:, 0], pos[:, 1], feature.columns.tolist()):
#    if "name_" in t:
#        plt.scatter(x, y)
#        plt.text(x=x, y=y, s=t)
    
    if "province_" in t:
        plt.scatter(x, y)
        plt.text(x=x, y=y, s=t)
        
#    if "tude" in t:
#        continue    
#    plt.scatter(x, y)
#    plt.text(x=x, y=y, s=t)

plt.title("Isomap (province)")
