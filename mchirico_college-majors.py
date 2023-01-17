

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from subprocess import check_output

# This gives us the listing....

#print(check_output(["ls", "../input/data"]).decode("utf8"))







#  Pick a Dataset you might be interested in.

#  

import zipfile



Dataset = "college-majors"



# Will unzip the files so that you can see them..

with zipfile.ZipFile("../input/data/"+Dataset+".zip","r") as z:

    z.extractall(".")

    

print(check_output(["ls", Dataset]).decode("utf8"))
#d=pd.read_csv(Dataset+'/'+"grad_students.csv")



print(check_output(["ls", Dataset+'/grad-students.csv']).decode("utf8"))

d=pd.read_csv("college-majors/grad-students.csv")
d.head()