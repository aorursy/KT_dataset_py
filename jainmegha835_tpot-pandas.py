import pandas as pd
import numpy as np
data=pd.read_csv("../input/magic1/magic04.data",header=None)
data.head(1)
data.columns=['fLength', 'fWidth','fSize','fConc','fConcl','fAsym','fM3Long','fM3Trans','fAlpha','fDist','class']
data.head(1)
data.info()
data.describe()
data['class'].value_counts()
shuffle=data.iloc[np.random.permutation(len(data))]
tele=shuffle.reset_index(drop=True)
tele.head(2)
tele['class']=tele["class"].map({'g':0,"h":1})
tele.head(2)
tele_class=tele['class'].values
pd.isnull(tele).any

from sklearn.model_selection import train_test_split
training_indices, validation_indices = training_indices, testing_indices = train_test_split(tele.index,
                                                                                            stratify = tele_class,
                                                                                            train_size=0.75, test_size=0.25)
training_indices.size, validation_indices.size
from tpot import TPOTClassifier
from tpot import TPOTRegressor

tpot = TPOTClassifier(generations=5,verbosity=2)

tpot.fit(tele.drop('class',axis=1).loc[training_indices].values,
         tele.loc[training_indices,'class'].values)