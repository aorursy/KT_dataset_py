# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")
speciesCounts = Counter(df['species'])
species = sorted(list(speciesCounts.keys()))

nSpecies = len(species)
testDF = pd.read_csv("../input/test.csv")

out = []

for r in testDF.iterrows():

    id = int(r[1]['id'])

    out.append([id] + list(np.ones(nSpecies)))
submission = pd.DataFrame(out, columns=['id'] + species)
submission.to_csv('submission.csv', index=False)