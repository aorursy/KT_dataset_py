import pandas as pd



data = pd.read_csv("../input/DigiDB_digimonlist.csv")

data.head()
import seaborn as sns



digimon = data["Attribute"]

sns.countplot(digimon).set_title("Digimon Attributes") # bar graph of the different attributes of Digimons
sns.stripplot(x=digimon, y=data["Memory"], data=data, jitter=True); # strip plot of Digimon Attributes vs. Memory