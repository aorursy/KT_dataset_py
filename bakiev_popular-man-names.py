import pandas as pd
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv")
#Popular Man Names 
train['Name'].str.extract('(Mr\. |Don\. |Master\. |Rev\. |Col\.[A-Za-z ]*\()([A-Za-z]*)')[1].value_counts() 