import numpy as np

import pandas as pd 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/student-mat.csv")



print("Keskmine vanus uuritavate seas on", df["age"].mean())

print("Väikseim vanus uuritavate seas on", df["age"].min())

print("Suurim vanus uuritavate seas on", df["age"].max())

df["Fjob"].value_counts()