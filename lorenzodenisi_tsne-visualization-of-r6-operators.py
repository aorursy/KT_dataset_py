import csv

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image

import os

import numpy as np

import pandas as pd

from sklearn.manifold import TSNE

from sklearn.preprocessing import Normalizer

from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from sklearn.decomposition import TruncatedSVD
def load_data(path):

    data = {}

    with open(path) as f:

        header = True

        headers = []

        for cols in csv.reader(f):

            if header:

                header = False

                headers = cols.copy()

                for h in headers: data[h] = []

                continue

            for n, cols in enumerate(cols):

                data[headers[n]].append(cols)

    return data



def load_icons(path):

    icons = {}

    files = os.listdir(path)

    for file in files:

        icons[(file[:-4]).upper()] = (Image.open(path + "/" + file)).resize((100, 100))



    return icons
operators = load_data("../input/tom-clancys-r6-operators-and-weapons/operators.csv")

weapons = load_data("../input/tom-clancys-r6-operators-and-weapons/weapons.csv")

icons = load_icons("../input/r6-operator-icons")
weapons_df = pd.DataFrame(weapons).drop(["Operator", "Organization", "Damage Suppressed", "Class"], axis=1)

weapons_df = pd.get_dummies(weapons_df, columns=["Type", "Role", "Suppressor", "ACOG", "Range"])
def get_weapon_info(weapon_name, weapons):

    res = weapons_df[weapons_df.Name==weapon_name].drop("Name", axis=1).values.tolist()

    if len(res)>0:

        return res[0]

    else:

        return [0]*(len(weapons_df.columns)-1)



# used when an operator has less than 3 primaries or less than two secondaries (average)

def merge_weapon_features(f1, f2):

    return ((np.array(f1, dtype=float) + np.array(f2, dtype=float)) / 2).tolist()

# fixing rate of fire

weapons_df.ROF.replace("Semi-Auto", 1, inplace=True)

weapons_df.ROF.replace("Pump", 1, inplace=True)

pt_features = []

for i in range(len(operators["Name"])):

    features = []

    features.append(operators["Armor"][i])

    features.append(operators["Speed"][i])

    features.append(operators["Difficulty"][i])



    features.extend(get_weapon_info(operators["Primary1"][i], weapons))

    if operators["Primary2"][i] == "":

        features.extend(get_weapon_info(operators["Primary1"][i], weapons))

        features.extend(get_weapon_info(operators["Primary1"][i], weapons))

    else:

        features.extend(get_weapon_info(operators["Primary2"][i], weapons))

        if operators["Primary3"][i] == "":

            f1 = get_weapon_info(operators["Primary1"][i], weapons)

            f2 = get_weapon_info(operators["Primary2"][i], weapons)

            features.extend(merge_weapon_features(f1, f2))

        else:

            features.extend(get_weapon_info(operators["Primary3"][i], weapons))



    features.extend(get_weapon_info(operators["Secondary1"][i], weapons))

    if operators["Secondary2"][i] == "":

        features.extend(get_weapon_info(operators["Secondary1"][i], weapons))

    else:

        features.extend(get_weapon_info(operators["Secondary2"][i], weapons))



    pt_features.append(np.array(features, dtype=float))
tsne = TSNE(n_components=2, perplexity=10, verbose=1, init='random', random_state=42)

pts = tsne.fit_transform(pt_features)
fig, ax = plt.subplots(figsize=(15, 15))

ax.scatter(pts[:, 0], pts[:, 1])



for n, pt in enumerate(pts):

    img = icons[operators["Name"][n]]

    ab = AnnotationBbox(OffsetImage(img, zoom=0.6), (pt[0], pt[1]), frameon=False)

    ax.add_artist(ab)



plt.grid()

plt.show()

with open("results.csv", mode='w', newline='') as file:

    writer = csv.writer(file, delimiter=',')

    if len(pts[0]) == 2:

        writer.writerow(["Name", 'X', 'Y'])

        for label, pt in zip(operators, pts):

            writer.writerow([label, pt[0], pt[1]])

            

    if len(pts[0]) == 3:

        writer.writerow(["Name", 'X', 'Y', 'Z'])

        for label, pt in zip(operators, pts):

            writer.writerow([label, pt[0], pt[1], pt[2]])