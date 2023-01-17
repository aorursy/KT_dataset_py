import pandas as pd



# data I put on kaggle!

file = "Womens_World_Cup_2019.csv"



soccer = pd.read_csv(file)

soccer.sample(n=10, random_state=33)
soccer.info()
soccer.describe()