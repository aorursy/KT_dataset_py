import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../input/datasauruses/datasauruses.csv")
data["datasaurus_name"].unique()
rex_data = data.query("datasaurus_name == 'rex'")
sns.scatterplot(data=rex_data, x="x", y="y");
stego_data = data.query("datasaurus_name == 'stego'")
sns.scatterplot(data=stego_data, x="x", y="y");
