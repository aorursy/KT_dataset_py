import numpy as np
import pandas as pd

pokemon = pd.read_csv("../input/pokemon.csv")
types = pd.read_csv("../input/type-chart.csv")

type_names = list(types.columns.values)[2:]
query = str.join(" & ", [type_name + " <= 1" for type_name in type_names])
invincible_types = types.query(query)

print("Woohoo!" if len(invincible_types) == 0 else "Aww man!")

vulnerabilities = {
    type_1: {
        type_2: [] for type_2 in types.loc[lambda type: type["defense-type1"] == "normal"]["defense-type2"]
    } for type_1 in type_names
}

print(vulnerabilities)