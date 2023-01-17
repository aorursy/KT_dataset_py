import pandas as pd

decision_data = pd.read_csv('/kaggle/input/uk-human-trafficking-data/2016_decision_data.csv')

decision_data[:-1].plot.line(x='Year', 

                             y=['Total Number of Referrals','Positive Conclusive Decisions'],

                             figsize=(10,10))