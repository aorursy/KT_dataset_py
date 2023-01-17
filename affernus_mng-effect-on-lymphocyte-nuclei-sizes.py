import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
morpho_mng_path = "../input/mng-effect-on-the-size-of-human-lymphocyte-nuclei/MNG_lymphocyte_nuclei_morphometry.csv"
morpho_mng = pd.read_csv(morpho_mng_path)

morpho_mng
morpho_mng.info()
first_nucleus_data = morpho_mng.loc[morpho_mng.loc[:, "nucleus N"] == 1].reset_index()

second_nucleus_data = morpho_mng.loc[morpho_mng.loc[:, "nucleus N"] == 2].reset_index()

comparsion = pd.DataFrame()

comparsion.loc[:, "MNG dose"] = first_nucleus_data.loc[:, "MNG dose (mcg by ml)"]

comparsion.loc[:, "first nucleus area"] = first_nucleus_data.loc[:, "Area (px)"]

comparsion.loc[:, "second nucleus area"] = second_nucleus_data.loc[:, "Area (px)"]

comparsion.loc[:, "areas difference"] = abs(comparsion.loc[:, "first nucleus area"] - comparsion.loc[:, "second nucleus area"])

comparsion.loc[:, "areas sum"] = comparsion.loc[:, "first nucleus area"] + comparsion.loc[:, "second nucleus area"]

comparsion.loc[:, "areas difference normalized"] = ((comparsion.loc[:, "areas difference"] / comparsion.loc[:, "areas sum"])*100).round(2)

comparsion
plt.figure(figsize=(12,6))

sns.distplot(a=comparsion.loc[0:99, "areas difference"], label=0, kde=False)

sns.distplot(a=comparsion.loc[100:199, "areas difference"], label=0.75, kde=False)

sns.distplot(a=comparsion.loc[200:299, "areas difference"], label=1.50, kde=False)

sns.distplot(a=comparsion.loc[300:399, "areas difference"], label=3.00, kde=False)

sns.distplot(a=comparsion.loc[400:499, "areas difference"], label=6.00, kde=False)

plt.legend()

plt.title("Histogram of areas differences, by MNG dose \n")
plt.figure(figsize=(12,6))

sns.kdeplot(data=comparsion.loc[0:99, "areas difference"], label=0)

sns.kdeplot(data=comparsion.loc[100:199, "areas difference"], label=0.75)

sns.kdeplot(data=comparsion.loc[200:299, "areas difference"], label=1.50)

sns.kdeplot(data=comparsion.loc[300:399, "areas difference"], label=3.00)

sns.kdeplot(data=comparsion.loc[400:499, "areas difference"], label=6.00)

plt.title("Distributions of areas differences, by MNG dose \n")

plt.xlabel("areas difference")
plt.figure(figsize=(12,6))

sns.kdeplot(data=comparsion.loc[0:99, "areas difference normalized"], label=0)

sns.kdeplot(data=comparsion.loc[100:199, "areas difference normalized"], label=0.75)

sns.kdeplot(data=comparsion.loc[200:299, "areas difference normalized"], label=1.50)

sns.kdeplot(data=comparsion.loc[300:399, "areas difference normalized"], label=3.00)

sns.kdeplot(data=comparsion.loc[400:499, "areas difference normalized"], label=6.00)

plt.title("Distributions of normalized areas differences, by MNG dose \n")

plt.xlabel("normalized areas difference")
plt.figure(figsize=(12,6))

sns.set_style("whitegrid")

sns.lineplot(x="MNG dose", y="areas difference", data=comparsion)

plt.title("Changes in nuclei areas difference depending on the dose of MNG \n")
plt.figure(figsize=(12,6))

sns.set_style("whitegrid")

sns.lineplot(x="MNG dose", y="areas difference normalized", data=comparsion)

plt.title("Changes in  nozmalized nuclei areas difference depending on the dose of MNG \n")
plt.figure(figsize=(12,6))

sns.set_style("whitegrid")

sns.lineplot(x="MNG dose", y="areas sum", data=comparsion)

plt.title("Changes in nuclei areas sum depending on the dose of MNG \n")
plt.figure(figsize=(12,6))

sns.swarmplot(x=comparsion['MNG dose'], y=comparsion['areas difference normalized'])

plt.title("Changes in nuclei areas difference depending on the dose of MNG: \n categorical scatter plot\n")