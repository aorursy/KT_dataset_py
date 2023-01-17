import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from subprocess import check_output
test_data = pd.read_csv("../input/test.csv")

train_data = pd.read_csv("../input/train.csv")
species = train_data["species"].unique()

species = pd.Series(range(0, len(species)), index=list(species))

print(species.iloc[0:5])
def fix_spieces(line):

    line['species'] = species.loc[line['species']]

    return line

train_data = train_data.apply(fix_spieces, axis=1)
forest = RandomForestClassifier(n_estimators = 100)

train = train_data.values[0::, 2::]

forest = forest.fit(train, train_data.values[0::, 1])

output = forest.predict_proba(test_data.values[0::, 1::])
kn = KNeighborsClassifier(n_neighbors=3)

kn = kn.fit(train, train_data.values[0::, 1])

output2 = kn.predict_proba(test_data.values[0::, 1::])
output_final = ((output/2) + (output2)*2)/2
out = pd.DataFrame(output_final, index=test_data.values[0::, 0].astype(np.int), columns = species.index.values)

out.index.name = "id"
out.to_csv("output.csv")