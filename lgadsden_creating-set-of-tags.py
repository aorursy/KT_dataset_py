# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
bio = pd.read_csv("../input/biology.csv")

cook = pd.read_csv("../input/cooking.csv")

crypto = pd.read_csv("../input/crypto.csv")

diy = pd.read_csv("../input/diy.csv")

travel = pd.read_csv("../input/travel.csv")

samp = pd.read_csv("../input/sample_submission.csv")

test = pd.read_csv("../input/test.csv")
bio.shape
biotags = bio["tags"].unique().tolist()

biotags = [i.split() for i in biotags]

biotagsFin = set(sum(biotags,[]))
cooktags = cook["tags"].unique().tolist()

cooktags = [i.split() for i in cooktags]

cooktagsFin = set(sum(cooktags,[]))
cryptotags = crypto["tags"].unique().tolist()

cryptotags = [i.split() for i in cryptotags]

cryptotagsFin = set(sum(cryptotags,[]))
diytags = diy["tags"].unique().tolist()

diytags = [i.split() for i in diytags]

diytagsFin = set(sum(diytags,[]))
travtags = travel["tags"].unique().tolist()

travtags = [i.split() for i in travtags]

travtagsFin = set(sum(travtags,[]))
tags_tot = set(list(biotagsFin) + list(cooktagsFin) + list(cryptotagsFin) + list(diytagsFin) + list(travtagsFin))
tags_tot