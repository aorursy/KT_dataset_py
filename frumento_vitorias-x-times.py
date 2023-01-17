%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
times = pd.read_csv('../input/brasileirao/brasileirao.csv')
times['Vencedor'].value_counts()
times['Vencedor'].value_counts().plot(kind='bar', figsize=(15,7), color='darkred')
