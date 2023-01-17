import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



attacks = pd.read_csv("../input/attacks.csv", encoding="latin1")

(attacks.groupby("Country").count().iloc[:,0]

        .to_frame().reset_index(level=0).sort_values(by="Case Number", ascending=False))