import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px



import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
data.shape
data.info()
data.head()
def variable(variable):
    var = data[variable]
    varcount = var.value_counts()
    
    plt.Figure(figsize = (1,1))
    sns.barplot(x = varcount.index , y = varcount.values)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.show()

for i in data.columns:
    variable(i)
capshape = data["cap-shape"].unique()

poisson_state_top = []
edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in capshape:
    edibles = len(data[(data["cap-shape"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["cap-shape"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["cap-shape"] == i ) & (data["class"] == "e")]) + len(data[(data["cap-shape"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

capshape_poisson_list = pd.DataFrame({"cap-shape" : capshape , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(capshape))
bar_width = 0.35

rects1 = plt.bar(index, capshape_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, capshape_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel('Cap Shape')
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('x', 'b', 's', 'f', 'k', 'c'))
plt.legend()

plt.tight_layout()
plt.show()
capshape_poisson_list
data["cap-shape"].value_counts()
capsurface = data["cap-surface"].unique()

poisson_state_top = []
edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in capsurface:
    edibles = len(data[(data["cap-surface"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["cap-surface"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["cap-surface"] == i ) & (data["class"] == "e")]) + len(data[(data["cap-surface"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

capsurfac_poisson_list = pd.DataFrame({"cap-surface" : capsurface , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(capsurface))
bar_width = 0.35

rects1 = plt.bar(index, capsurfac_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, capsurfac_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel('Cap Surface')
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('s', 'y', 'f', 'g'))
plt.legend()

plt.tight_layout()
plt.show()
capsurfac_poisson_list
data['cap-surface'].value_counts()
capcolor = data["cap-color"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in capcolor:
    edibles = len(data[(data["cap-color"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["cap-color"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["cap-color"] == i ) & (data["class"] == "e")]) + len(data[(data["cap-color"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

capcolor_poisson_list = pd.DataFrame({"cap-color" : capcolor , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(capcolor))
bar_width = 0.35

rects1 = plt.bar(index, capcolor_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, capcolor_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("cap-color")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('n', 'y', 'w', 'g', 'e', 'p', 'b', 'u', 'c', 'r'))
plt.legend()

plt.tight_layout()
plt.show()
capcolor_poisson_list
data["cap-color"].value_counts()
print("gill-attachment Variable Unique Values: {}".format(data["gill-attachment"].unique()))
print("gill-spacing Variable Unique Values: {}".format(data["gill-spacing"].unique()))
print("gill-size Variable Unique Values: {}".format(data["gill-size"].unique()))
print("gill-color Variable Unique Values: {}".format(data["gill-color"].unique()))
gillattachment = data["gill-attachment"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in gillattachment:
    edibles = len(data[(data["gill-attachment"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["gill-attachment"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["gill-attachment"] == i ) & (data["class"] == "e")]) + len(data[(data["gill-attachment"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

gillattachment_poisson_list = pd.DataFrame({"gill-attachment" : gillattachment , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(gillattachment))
bar_width = 0.35

rects1 = plt.bar(index, gillattachment_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, gillattachment_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Gill Attachment")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('f','a'))
plt.legend()

plt.tight_layout()
plt.show()
gillattachment_poisson_list
gillspacing = data["gill-spacing"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in gillspacing:
    edibles = len(data[(data["gill-spacing"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["gill-spacing"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["gill-spacing"] == i ) & (data["class"] == "e")]) + len(data[(data["gill-spacing"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

gillspacing_poisson_list = pd.DataFrame({"gill-spacing" : gillspacing , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(gillspacing))
bar_width = 0.35

rects1 = plt.bar(index, gillspacing_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, gillspacing_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Gill Spacing")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('c','w'))
plt.legend()

plt.tight_layout()
plt.show()
gillspacing_poisson_list
gillsize = data["gill-size"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in gillsize:
    edibles = len(data[(data["gill-size"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["gill-size"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["gill-size"] == i ) & (data["class"] == "e")]) + len(data[(data["gill-size"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

gillsize_poisson_list = pd.DataFrame({"gill-size" : gillsize , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(gillsize))
bar_width = 0.35

rects1 = plt.bar(index, gillsize_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, gillsize_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Gill Size")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('n','b'))
plt.legend()

plt.tight_layout()
plt.show()
gillsize_poisson_list
data["gill-size"].value_counts()
gillcolor = data["gill-color"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in gillcolor:
    edibles = len(data[(data["gill-color"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["gill-color"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["gill-color"] == i ) & (data["class"] == "e")]) + len(data[(data["gill-color"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

gillcolor_poisson_list = pd.DataFrame({"gill-color" : gillcolor , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(gillcolor))
bar_width = 0.35

rects1 = plt.bar(index, gillcolor_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, gillcolor_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Gill Color")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('k' 'n' 'g' 'p' 'w' 'h' 'u' 'e' 'b' 'r' 'y' 'o'))
plt.legend()

plt.tight_layout()
plt.show()
gillcolor_poisson_list
data["gill-color"].value_counts()
stalk_list = ["stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring"]
print("stalk-shape Variable Unique Values: {}".format(data["stalk-shape"].unique()))
print("stalk-root Variable Unique Values: {}".format(data["stalk-root"].unique()))
print("stalk-surface-above-ring Variable Unique Values: {}".format(data["stalk-surface-above-ring"].unique()))
print("stalk-surface-below-ring Variable Unique Values: {}".format(data["stalk-surface-below-ring"].unique()))
print("stalk-color-above-ring Variable Unique Values: {}".format(data["stalk-color-above-ring"].unique()))
print("stalk-color-below-ring Variable Unique Values: {}".format(data["stalk-color-below-ring"].unique()))
stalkshape = data["stalk-shape"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in stalkshape:
    edibles = len(data[(data["stalk-shape"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["stalk-shape"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["stalk-shape"] == i ) & (data["class"] == "e")]) + len(data[(data["stalk-shape"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

stalkshape_poisson_list = pd.DataFrame({"stalk-shape" : stalkshape , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(stalkshape))
bar_width = 0.35

rects1 = plt.bar(index, stalkshape_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, stalkshape_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Stalk Shape")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('e' 't'))
plt.legend()

plt.tight_layout()
plt.show()
stalkshape_poisson_list
stalkroot = data["stalk-root"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in stalkroot:
    edibles = len(data[(data["stalk-root"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["stalk-root"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["stalk-root"] == i ) & (data["class"] == "e")]) + len(data[(data["stalk-root"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

stalkroot_poisson_list = pd.DataFrame({"stalk-root" : stalkroot , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(stalkroot))
bar_width = 0.35

rects1 = plt.bar(index, stalkroot_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, stalkroot_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Stalk Root")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('e' 'c' 'b' 'r' '?'))
plt.legend()

plt.tight_layout()
plt.show()
stalkroot_poisson_list
stalksurabri = data["stalk-surface-above-ring"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in stalksurabri:
    edibles = len(data[(data["stalk-surface-above-ring"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["stalk-surface-above-ring"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["stalk-surface-above-ring"] == i ) & (data["class"] == "e")]) + len(data[(data["stalk-surface-above-ring"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

stalksurabri_poisson_list = pd.DataFrame({"stalk-surface-above-ring" : stalksurabri , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(stalksurabri))
bar_width = 0.35

rects1 = plt.bar(index, stalksurabri_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, stalksurabri_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Stalk Surface Above Ring")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('s' 'f' 'k' 'y'))
plt.legend()

plt.tight_layout()
plt.show()
stalksurabri_poisson_list
stalksurberi = data["stalk-surface-below-ring"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in stalksurberi:
    edibles = len(data[(data["stalk-surface-below-ring"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["stalk-surface-below-ring"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["stalk-surface-below-ring"] == i ) & (data["class"] == "e")]) + len(data[(data["stalk-surface-below-ring"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

stalksurberi_poisson_list = pd.DataFrame({"stalk-surface-below-ring" : stalksurberi , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(stalksurberi))
bar_width = 0.35

rects1 = plt.bar(index, stalksurberi_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, stalksurberi_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Stalk Surface Below Ring")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('s' 'f' 'k' 'y'))
plt.legend()

plt.tight_layout()
plt.show()
stalksurberi_poisson_list
stalkcoabri = data["stalk-color-above-ring"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in stalkcoabri:
    edibles = len(data[(data["stalk-color-above-ring"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["stalk-color-above-ring"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["stalk-color-above-ring"] == i ) & (data["class"] == "e")]) + len(data[(data["stalk-color-above-ring"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

stalkcoabri_poisson_list = pd.DataFrame({"stalk-color-above-ring" : stalkcoabri , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(stalkcoabri))
bar_width = 0.35

rects1 = plt.bar(index, stalkcoabri_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, stalkcoabri_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Stalk Color Above Ring")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('w' 'g' 'p' 'n' 'b' 'e' 'o' 'c' 'y'))
plt.legend()

plt.tight_layout()
plt.show()
stalkcoabri_poisson_list
stalkcoberi = data["stalk-color-below-ring"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in stalkcoberi:
    edibles = len(data[(data["stalk-color-below-ring"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["stalk-color-below-ring"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["stalk-color-below-ring"] == i ) & (data["class"] == "e")]) + len(data[(data["stalk-color-below-ring"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

stalkcoberi_poisson_list = pd.DataFrame({"stalk-color-below-ring" : stalkcoberi , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(stalkcoberi))
bar_width = 0.35

rects1 = plt.bar(index, stalkcoberi_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, stalkcoberi_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Stalk Color Below Ring")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('w' 'p' 'g' 'b' 'n' 'e' 'y' 'o' 'c'))
plt.legend()

plt.tight_layout()
plt.show()
stalkcoberi_poisson_list
veil_list = ["veil-type" ,"veil-color"]

print("veil-type Variable Unique Values: {}".format(data["veil-type"].unique()))
print("veil-color Variable Unique Values: {}".format(data["veil-color"].unique()))
veiltype = data["veil-type"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in veiltype:
    edibles = len(data[(data["veil-type"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["veil-type"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["veil-type"] == i ) & (data["class"] == "e")]) + len(data[(data["veil-type"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

veiltype_poisson_list = pd.DataFrame({"veil-type" : veiltype , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(veiltype))
bar_width = 0.35

rects1 = plt.bar(index, veiltype_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, veiltype_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Veil Type")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('p'))
plt.legend()

plt.tight_layout()
plt.show()
veiltype_poisson_list
veilcolor = data["veil-color"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in veilcolor:
    edibles = len(data[(data["veil-color"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["veil-color"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["veil-color"] == i ) & (data["class"] == "e")]) + len(data[(data["veil-color"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

veilcolor_poisson_list = pd.DataFrame({"veil-color" : veilcolor , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(veilcolor))
bar_width = 0.35

rects1 = plt.bar(index, veilcolor_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, veilcolor_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Veil Color")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('w' 'n' 'o' 'y'))
plt.legend()

plt.tight_layout()
plt.show()
veilcolor_poisson_list
ring_list = ["ring-number","ring-type"]

print("ring-number Variable Unique Values: {}".format(data["ring-number"].unique()))
print("ring-type Variable Unique Values: {}".format(data["ring-type"].unique()))
ringnumber = data["ring-number"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in ringnumber:
    edibles = len(data[(data["ring-number"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["ring-number"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["ring-number"] == i ) & (data["class"] == "e")]) + len(data[(data["ring-number"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

ringnum_poisson_list = pd.DataFrame({"ring-number" : ringnumber , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(ringnumber))
bar_width = 0.35

rects1 = plt.bar(index, ringnum_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, ringnum_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Ring Number")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('o' 't' 'n'))
plt.legend()

plt.tight_layout()
plt.show()
ringnum_poisson_list
ringtype = data["ring-type"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in ringtype:
    edibles = len(data[(data["ring-type"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["ring-type"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["ring-type"] == i ) & (data["class"] == "e")]) + len(data[(data["ring-type"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

ringtype_poisson_list = pd.DataFrame({"ring-type" : ringtype , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(ringtype))
bar_width = 0.35

rects1 = plt.bar(index, ringtype_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, ringtype_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Ring Type")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('p' 'e' 'l' 'f' 'n'))
plt.legend()

plt.tight_layout()
plt.show()
ringtype_poisson_list
other_list =  ["bruises","odor" , "spore-print-color" ,"population","habitat"]

print("bruises Variable Unique Values: {}".format(data["bruises"].unique()))
print("odor Variable Unique Values: {}".format(data["odor"].unique()))
print("spore-print-color Variable Unique Values: {}".format(data["spore-print-color"].unique()))
print("population Variable Unique Values: {}".format(data["population"].unique()))
print("habitat Variable Unique Values: {}".format(data["habitat"].unique()))

bruises = data["bruises"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in bruises:
    edibles = len(data[(data["bruises"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["bruises"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["bruises"] == i ) & (data["class"] == "e")]) + len(data[(data["bruises"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

bruises_poisson_list = pd.DataFrame({"bruises" : bruises , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(bruises))
bar_width = 0.35

rects1 = plt.bar(index, bruises_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, bruises_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Bruises")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('t' 'f'))
plt.legend()

plt.tight_layout()
plt.show()
bruises_poisson_list
odor = data["odor"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in odor:
    edibles = len(data[(data["odor"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["odor"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["odor"] == i ) & (data["class"] == "e")]) + len(data[(data["odor"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

odor_poisson_list = pd.DataFrame({"odor" : odor , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(odor))
bar_width = 0.35

rects1 = plt.bar(index, odor_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, odor_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Odor")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('p' 'a' 'l' 'n' 'f' 'c' 'y' 's' 'm'))
plt.legend()

plt.tight_layout()
plt.show()
odor_poisson_list
sporeprintcolor = data["spore-print-color"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in sporeprintcolor:
    edibles = len(data[(data["spore-print-color"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["spore-print-color"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["spore-print-color"] == i ) & (data["class"] == "e")]) + len(data[(data["spore-print-color"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

sporeprintcolor_poisson_list = pd.DataFrame({"spore-print-color" : sporeprintcolor , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(sporeprintcolor))
bar_width = 0.35

rects1 = plt.bar(index, sporeprintcolor_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, sporeprintcolor_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Spore Print Color")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('k' 'n' 'u' 'h' 'w' 'r' 'o' 'y' 'b'))
plt.legend()

plt.tight_layout()
plt.show()
sporeprintcolor_poisson_list
population  = data["population"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in population :
    edibles = len(data[(data["population"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["population"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["population"] == i ) & (data["class"] == "e")]) + len(data[(data["population"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

population_poisson_list = pd.DataFrame({"population" : population  , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(population ))
bar_width = 0.35

rects1 = plt.bar(index, population_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, population_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Population")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('s' 'n' 'a' 'v' 'y' 'c'))
plt.legend()

plt.tight_layout()
plt.show()
population_poisson_list
habitat  = data["habitat"].unique()

edible = []
poisson = []
ort = []
edible_ort = []
poisson_ort = []

for i in habitat :
    edibles = len(data[(data["habitat"] == i ) & (data["class"] == "e")])
    poissons = len(data[(data["habitat"] == i ) & (data["class"] == "p")])
    ort_pers =  len(data[(data["habitat"] == i ) & (data["class"] == "e")]) + len(data[(data["habitat"] == i ) & (data["class"] == "p")]) 
    edible_ort_pers = (edibles / ort_pers)*100
    poisson_ort_pers = (poissons / ort_pers)*100

    edible.append(edibles)
    poisson.append(poissons)
    edible_ort.append(edible_ort_pers)
    poisson_ort.append(poisson_ort_pers)
    

habitat_poisson_list = pd.DataFrame({"habitat" : habitat  , "edible_count" : edible ,"edible_ort" : edible_ort , "poisson_count" : poisson, "poisson_ort" : poisson_ort})

fig, ax = plt.subplots()
index = np.arange(len(habitat))
bar_width = 0.35

rects1 = plt.bar(index, habitat_poisson_list["edible_count"], bar_width,
color='b',
label='Edible')

rects2 = plt.bar(index + bar_width, habitat_poisson_list["poisson_count"], bar_width,
color='g',
label='Poisson')


plt.xlabel("Habitat")
plt.ylabel('Poisson State')
plt.xticks(index + bar_width, ('u' 'g' 'm' 'd' 'p' 'w' 'l'))
plt.legend()

plt.tight_layout()
plt.show()
habitat_poisson_list
data.isnull().sum()
data["stalk-root"].unique()
stalkroot_poisson_list
data["stalk-root"].replace("?" ,"stalk_other" ,inplace = True)
data["stalk-root"].unique()
data["cap-shape"].replace(["s" , "c"] , "b" , inplace = True)
data["cap-shape"].unique()
data["cap-surface"].replace("g" , "y" , inplace = True)
data["cap-surface"].unique()
data["cap-color"].replace(["p" ,"b" , "u" ,"c" , "r"] , "cap_other" , inplace = True)
data["cap-color"].unique()
data = pd.concat([data , pd.get_dummies(data["cap-shape"])] , axis =1)
data = pd.concat([data , pd.get_dummies(data["cap-surface"])] , axis =1)
data = pd.concat([data , pd.get_dummies(data["cap-color"])] , axis =1)

data.drop("cap-shape" , axis = 1 , inplace = True)
data.drop("cap-surface" , axis = 1 , inplace = True)
data.drop("cap-color" , axis = 1 , inplace = True)
data.drop("gill-attachment" , axis = 1 , inplace = True)
data["gill-color"] = data["gill-color"].replace(["e" ,"b" , "r" ,"y" , "o"] , "gill_other")
data["gill-color"].unique()
data = pd.concat([data , pd.get_dummies(data["gill-spacing"])] , axis =1)
data = pd.concat([data , pd.get_dummies(data["gill-size"])] , axis =1)
data = pd.concat([data , pd.get_dummies(data["gill-color"])] , axis =1)

data.drop("gill-spacing" , axis = 1 , inplace = True)
data.drop("gill-size" , axis = 1 , inplace = True)
data.drop("gill-color" , axis = 1 , inplace = True)
data["stalk-surface-above-ring"] = data["stalk-surface-above-ring"].replace("y" ,"s")
data["stalk-surface-above-ring"].unique()
data["stalk-color-above-ring"] = data["stalk-color-above-ring"].replace(["b","c","y", "e"] ,"stalk-color-above-other")
data["stalk-color-above-ring"].unique()
data["stalk-color-below-ring"] = data["stalk-color-below-ring"].replace(["c","y", "e"] ,"stalk-color-below-other")
data["stalk-color-below-ring"].unique()
data = pd.concat([data , pd.get_dummies(data["stalk-shape"])] , axis =1)
data = pd.concat([data , pd.get_dummies(data["stalk-root"])] , axis =1)
data = pd.concat([data , pd.get_dummies(data["stalk-surface-below-ring"])] , axis =1)
data = pd.concat([data , pd.get_dummies(data["stalk-surface-above-ring"])] , axis =1)
data = pd.concat([data , pd.get_dummies(data["stalk-color-above-ring"])] , axis =1)
data = pd.concat([data , pd.get_dummies(data["stalk-color-below-ring"])] , axis =1)


data.drop("stalk-shape" , axis = 1 , inplace = True)
data.drop("stalk-root" , axis = 1 , inplace = True)
data.drop("stalk-surface-below-ring" , axis = 1 , inplace = True)
data.drop("stalk-surface-above-ring" , axis = 1 , inplace = True)
data.drop("stalk-color-above-ring" , axis = 1 , inplace = True)
data.drop("stalk-color-below-ring" , axis = 1 , inplace = True)
data["veil-color"] = data["veil-color"].replace(["n" , "o" , "y"] , "veil-color")
data["veil-color"].unique()
data = pd.concat([data , pd.get_dummies(data["veil-type"])] , axis =1)
data = pd.concat([data , pd.get_dummies(data["veil-color"])] , axis =1)


data.drop("veil-type" , axis = 1 , inplace = True)
data.drop("veil-color" , axis = 1 , inplace = True)
data["ring-number"] = data["ring-number"].replace("n" , "o")
data["ring-number"].unique()

data = pd.concat([data , pd.get_dummies(data["ring-number"])] , axis =1)
data.drop("ring-number" , axis = 1 , inplace = True)

data["ring-type"] = data["ring-type"].replace(["f" , "n"] , "p")
data["ring-type"].unique()

data = pd.concat([data , pd.get_dummies(data["ring-type"])] , axis =1)
data.drop("ring-type" , axis = 1 , inplace = True)

data["odor"] = data["odor"].replace("m" , "n")
data["odor"].unique()
data["spore-print-color"] = data["spore-print-color"].replace(["u","r" ,"o", "y" ,"b"] , "spor-other")
data["spore-print-color"].unique()
data = pd.concat([data , pd.get_dummies(data["odor"])] , axis =1)
data.drop("odor" , axis = 1 , inplace = True)

data = pd.concat([data , pd.get_dummies(data["bruises"])] , axis =1)
data.drop("bruises" , axis = 1 , inplace = True)

data = pd.concat([data , pd.get_dummies(data["spore-print-color"])] , axis =1)
data.drop("spore-print-color" , axis = 1 , inplace = True)

data = pd.concat([data , pd.get_dummies(data["population"])] , axis =1)
data.drop("population" , axis = 1 , inplace = True)

data = pd.concat([data , pd.get_dummies(data["habitat"])] , axis =1)
data.drop("habitat" , axis = 1 , inplace = True)
data["class"] = data["class"].replace("p" ,0)
data["class"] = data["class"].replace("e" ,1)

data["class"].unique()
data
from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV
x = data.drop("class" , axis = 1)
y = data["class"]
x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.33 , random_state = 42)
print("x_train: {}".format(x_train.shape))
print("x_test: {}".format(x_test.shape))
print("y_train: {}".format(y_train.shape))
print("y_test: {}".format(y_test.shape))
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train , y_train)

print("Decision Tree Test Accuracy: {}".format(dt.score(x_test, y_test)))

score_list = []
for i in range(1,100):
    rf = RandomForestClassifier(n_estimators = i , random_state = 42)
    rf.fit(x_train , y_train)
    score = rf.score(x_train , y_train)
    score_list.append(score)
    
    
plt.plot(range(1,100) , score_list)
plt.show()
rf = RandomForestClassifier(n_estimators = 3 , random_state = 42)
rf.fit(x_train , y_train)

print("Random Forest Test Accuracy {}".format(rf.score(x_test , y_test)))
lr = LogisticRegression()
lr.fit(x_train , y_train)
print("Logistic Regression Test Accuracy {}".format(lr.score(x_test , y_test)))
svm = SVC(random_state = 42)
svm.fit(x_train , y_train)
print("Support Vector Machine Test Accuracy {}".format(svm.score(x_test , y_test)))
nb = GaussianNB()
nb.fit(x_train , y_train)
print("Naive Bayes Test Accuracy {}".format(nb.score(x_test , y_test)))