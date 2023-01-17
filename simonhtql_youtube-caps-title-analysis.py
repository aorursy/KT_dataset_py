import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
ds = pd.read_csv("../input/FRvideos.csv")

ds.head()

def capsAmount(title="Ab cD "):
    capsRatio=0
    titleSize=len(title)
    nbCaps=0
    if title.isupper():
        capsRatio=1
    else:
        for char in title:
            if char.isspace() :
                titleSize-=1
            else:
                if char.isupper():
                    nbCaps+=1
        if titleSize>0:
            capsRatio=round(nbCaps/titleSize,1)
        
    return capsRatio
        
capsAmount("THIS is MY TITLE !")
from collections import Counter
titles= ds["title"]
caps=list()
for title in titles:
    caps.append(capsAmount(title))
    
c = Counter(caps)
c
