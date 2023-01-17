!pip install ctgan

import pandas as pd

data=pd.read_csv("../input/iris/Iris.csv")

discrete_columns=data.select_dtypes(include=['object']).columns.values

from ctgan import CTGANSynthesizer

ctgan = CTGANSynthesizer()

ctgan.fit(data, discrete_columns)

samples = ctgan.sample(1000000)

samples.to_csv("Augmented_Iris.csv",index=False)

data.shape,samples.shape
