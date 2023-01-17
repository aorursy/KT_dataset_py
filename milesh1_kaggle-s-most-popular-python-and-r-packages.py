import pandas as pd
from collections import Counter
# Python packages
df_p = pd.read_json('../input/py_packages.json')
packages_p = []
for i in range(len(df_p)):
    for j in df_p:
        if df_p[j][i] is not None:
            packages_p.append(df_p[j][i][1])
# R packages
df_r = pd.read_json('../input/r_packages.json')
packages_r = []
for i in range(len(df_r)):
    for j in df_r:
        if df_r[j][i] is not None:
            packages_r.append(df_r[j][i][1])
packages_count_p = Counter(packages_p)
packages_count_p.most_common()[0:50]
packages_count_r = Counter(packages_r)
packages_count_r.most_common()[0:50]