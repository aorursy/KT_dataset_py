import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
zoo=pd.read_csv('../input/zoo-animal-classification/zoo.csv')

zoo.head()
hairlist=zoo['hair'].value_counts().tolist()

hairlist
hval=hairlist

hlabel=['animals without hair','animals with hair']

plt.pie(hval,labels=hlabel,autopct='%2.1f%%')
featherslist=zoo['feathers'].value_counts().tolist()

featherslist
fval=featherslist

flabel=['animals without feathers','animals with feathers']

plt.pie(fval,labels=flabel,autopct='%2.1f%%')
c=pd.read_csv('../input/zoo-animal-classification/class.csv')

c
Class_Type=c['Class_Type'].tolist()

Class_Type
numClass_Type=c['Number_Of_Animal_Species_In_Class'].tolist()

numClass_Type
cval=numClass_Type

clabel=Class_Type

plt.pie(cval,labels=clabel,autopct='%2.1f%%')
zoo_c = c.Class_Type.unique()

for tmp in zoo_c:

  animal= zoo.set_index('animal_name').loc[c.set_index('Class_Type').loc[tmp].Animal_Names.split(', ')]



  p = []

  for ii in range(0,9,2):

    p.append(len(animal.loc[(animal.predator==1) & (animal.legs == ii)]))

  n_p = []

  for ii in range(0,9,2):

    n_p.append(len(animal.loc[(animal.predator==0) & (animal.legs == ii)]))

  labels = ['0 leg', '2 legs', '4 legs' , '6 legs' , '8 legs']

  width = 0.5      



  fig, ax = plt.subplots()



  ax.bar(labels, p, width, label='predator')

  ax.bar(labels, n_p, width, bottom=p,

        label='non predator')



  ax.set_ylabel('number of '+tmp)

  ax.set_title('number of predater and non-predator grouped by number of legs')

  ax.legend()



  plt.show()
