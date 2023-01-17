import graphviz

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from pathlib import Path
DATA_DIR = Path("../input/stanford-covid-vaccine/")

BPPS_DIR = DATA_DIR / "bpps"



train = pd.read_json(DATA_DIR / "train.json", lines=True)

test = pd.read_json(DATA_DIR / "test.json", lines=True)



bppm_paths = list(BPPS_DIR.glob("*.npy"))
len(train) + len(test) == len(bppm_paths)
def get_bppm(id_):

    return np.load(BPPS_DIR / f"{id_}.npy")





def draw_structure(structure: str):

    pm = np.zeros((len(structure), len(structure)))

    start_token_indices = []

    for i, token in enumerate(structure):

        if token == "(":

            start_token_indices.append(i)

        elif token == ")":

            j = start_token_indices.pop()

            pm[i, j] = 1.0

            pm[j, i] = 1.0

    return pm





def plot_structures(bppm: np.ndarray, pm: np.ndarray):

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    axes[0].imshow(bppm)

    axes[0].set_title("BPPM")

    axes[1].imshow(pm)

    axes[1].set_title("structure")

    plt.show()
idx = 0

sample = train.loc[idx]



bppm = get_bppm(sample.id)

pm = draw_structure(sample.structure)

plot_structures(bppm, pm)
idx = 1

sample = train.loc[idx]



bppm = get_bppm(sample.id)

pm = draw_structure(sample.structure)

plot_structures(bppm, pm)
idx = 2

sample = train.loc[idx]



bppm = get_bppm(sample.id)

pm = draw_structure(sample.structure)

plot_structures(bppm, pm)
idx = 3

sample = train.loc[idx]



bppm = get_bppm(sample.id)

pm = draw_structure(sample.structure)

plot_structures(bppm, pm)
idx = 4

sample = train.loc[idx]



bppm = get_bppm(sample.id)

pm = draw_structure(sample.structure)

plot_structures(bppm, pm)
idx = 5

sample = train.loc[idx]



bppm = get_bppm(sample.id)

pm = draw_structure(sample.structure)

plot_structures(bppm, pm)
def visualize_graph(bppm: np.ndarray, sequence: str, threshold=0.1):

    indices = np.where(bppm > threshold)

    edges = list(zip(indices[0], indices[1], bppm[indices]))

    

    g = graphviz.Graph(format="png")

    for from_, to, coef in edges:

        if from_ > to:

            g.edge(sequence[from_] + f"({from_})",

                   sequence[to] + f"({to})",

                   label=f"{coef:.2f}",

                   penwidth=f"{int(max(1, abs(coef * 20)))}")

    g.render("./graph")

    return g
idx = 0

sample = train.loc[idx]



bppm = get_bppm(sample.id)

visualize_graph(bppm, sample.sequence, threshold=0.05)
idx = 1

sample = train.loc[idx]



bppm = get_bppm(sample.id)

visualize_graph(bppm, sample.sequence)
idx = 2

sample = train.loc[idx]



bppm = get_bppm(sample.id)

visualize_graph(bppm, sample.sequence)
idx = 3

sample = train.loc[idx]



bppm = get_bppm(sample.id)

visualize_graph(bppm, sample.sequence)
idx = 4

sample = train.loc[idx]



bppm = get_bppm(sample.id)

visualize_graph(bppm, sample.sequence)
idx = 5

sample = train.loc[idx]



bppm = get_bppm(sample.id)

visualize_graph(bppm, sample.sequence)