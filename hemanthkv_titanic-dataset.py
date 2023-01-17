import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
Data_train=pd.read_csv("../input/train.csv")
Data_test=pd.read_csv("../input/test.csv")
Data_train.head(5)
Data_test.head(5)
print(Data_train.shape)
print(Data_train.info())