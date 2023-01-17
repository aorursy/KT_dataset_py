# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
with open("../input/sandbox.csv") as f:
    data = f.readlines()
    
content = [x.strip() for x in data]

just_names = [x.split(',')[0] for x in content[1:]]

print(just_names)
from keras.models import load_model
from PIL import Image
MODEL_NAME = "test.h5"

# KOmmentieren Sie diese Zeilen ein, sobald sie einen Model namem habe
# dies gilt auch für die folge zeilen
# model = load.model(MODEL_NAME)
results = []
for c in just_names:
    img = Image.open("../input/test/test/" + c)
    
    # falls Sie vorverabrietung haben so führen Sie diese hier durch
    img = np.array(img,dtype=np.float32) / 255
    
    #res = np.argmax(model.predict(np.array[img])[0])
    # tauschen Sie die obere und die untere Zeile aus, so dass ihr Model
    # nun die vorhersage macht
    res = 1
    results.append(res)
# Rausschreiben der Ergebnisse zu einer Submission-Datei

with open("submission.csv","w") as f:
    f.write("ID,Class"+'\n')
    for ID,Class in zip(just_names,results):
        f.write(ID + "," + str(Class) + "\n")
    