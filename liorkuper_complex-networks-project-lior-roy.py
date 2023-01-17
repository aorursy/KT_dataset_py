import pandas as pd # for data processing



#Reading data & showing samples

data=pd.DataFrame(pd.read_csv("../input/Data_Astroph.csv"))

data.sample(3)
