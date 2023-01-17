import pandas as pd

train = pd.read_csv("../input/train.csv", dtype = {'Store': 'S20', 'DateOfWeek': 'S20', 'DayOfWeek': 'S20', 
                                                   'Open': 'bool', 'Promo': 'bool'})
#test = pd.read_csv("../input/test.csv")

print(train.describe())

train.columns, train.shape
train.groupby('Store').filter(lambda x: len(x)<942).Store.nunique()
import seaborn
