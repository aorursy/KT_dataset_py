import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv" )


#Popular Lady Name 
train['Name'].str.extract('(Miss\. |Dona\. |Lady\. |Ms\. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)')[1].value_counts() 