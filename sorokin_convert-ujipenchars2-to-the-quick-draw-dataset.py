import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def stack_it(drawing):
    # unwrap the list
    in_strokes = [(xi,yi,i) for i,(x,y) in enumerate(drawing) for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
    c_strokes[:,2] += 1 # since 0 is no stroke
    return c_strokes

print(os.listdir("../input"))
df = pd.DataFrame(columns=("subset", "site", "writer", "character", "repetitions", "numstrokes", "numpoints", "drawing"))
df = df.set_index(["subset", "site", "writer", "character", "repetitions"])
with open("../input/ujipenchars2.txt", "r") as file:
    for line in file:
        words = line.strip().split()
        if words[0] == "//":
            continue
        elif words[0] == "WORD":
            info = words[2].split("_")
            session = info[2].split("-")
            index = (info[0], info[1], session[0], words[1], int(session[1]))
            df.loc[index] = 0
        elif words[0] == "NUMSTROKES":
            df.loc[index]["numstrokes"] = int(words[1])
            df.loc[index]["numpoints"] = 0
            df.loc[index]["drawing"] = []
        elif words[0] == "POINTS":
            numpoints = int(words[1])
            points = words[3:]
            x = [int(x) for x in points[0::2]]
            y = [int(y) for y in points[1::2]]
            assert len(x) == numpoints, (len(x), numpoints)
            assert len(y) == numpoints, (len(y), numpoints)
            df.loc[index]["numpoints"] += numpoints
            df.loc[index]["drawing"].append([x, y])
df1 = df[df.index.get_level_values("character").isin(["a", "r", "m"])]
print(len(df), len(df1))
print(df1["drawing"].head())
fig, m_axs = plt.subplots(3,3, figsize = (16, 16))
rand_idxs = np.random.choice(range(len(df1)), size = 9)
for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    image = stack_it(df1.iloc[c_id]["drawing"])
    lab_idx = np.cumsum(image[:,2]-1)
    for i in np.unique(lab_idx):
        c_ax.plot(image[lab_idx==i,0], np.max(image[:,1])-image[lab_idx==i,1], '.-')
    c_ax.axis('off')
    c_ax.set_title(df1.index.values[c_id])