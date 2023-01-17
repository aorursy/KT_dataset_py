
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# First, look at everything.
from subprocess import check_output
print(check_output(["ls", "../input/data"]).decode("utf8"))

#  Pick a Dataset you might be interested in.
#  Say, all airline-safety files...
import zipfile

Dataset = "airline-safety"

# Will unzip the files so that you can see them..
with zipfile.ZipFile("../input/data/"+Dataset+".zip","r") as z:
    z.extractall(".")
from subprocess import check_output
print(check_output(["ls", "airline-safety"]).decode("utf8"))
# There's only one file above...we'll select it.
d=pd.read_csv(Dataset+"/airline-safety.csv")
d.head()