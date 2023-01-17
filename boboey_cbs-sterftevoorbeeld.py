!pip install cbsodata
import pandas as pd
import cbsodata
import matplotlib.pyplot as plt
# Checken of 70895NED is wat we zoeken
metadata = pd.DataFrame(cbsodata.get_meta('70895NED', 'DataProperties'))
metadata.head()
# Data ophalen die bij 70895NED hoort en checken of het is wat we verwachten met .head()
data = pd.DataFrame(cbsodata.get_data('70895NED'))
data.head()
totalen_2020 = data[data.Perioden.str.contains("2020 week") & data.Geslacht.str.contains("Totaal") & data.LeeftijdOp31December.str.contains("Totaal leeftijd")]
totalen_2019 = data[data.Perioden.str.contains("2019 week") & data.Geslacht.str.contains("Totaal") & data.LeeftijdOp31December.str.contains("Totaal leeftijd")]
# totalen_2018 = data[data.Perioden.str.contains("2018 week") & data.Geslacht.str.contains("Totaal") & data.LeeftijdOp31December.str.contains("Totaal leeftijd")]
# Omzetten van de data in kolom 'Perioden': van '2019 week 2' (str), naar 2 (int)
totalen_2019.loc[:,"Perioden"] = totalen_2019["Perioden"].str.replace("2019 week", "")
totalen_2019.loc[1297,"Perioden"] = "1"
totalen_2019.loc[1349,"Perioden"] = "52"
totalen_2019.loc[:,"Perioden"] = totalen_2019["Perioden"].astype(int)

# Idem 2020
totalen_2020.loc[:,"Perioden"] = totalen_2020["Perioden"].str.replace("2020 week", "")
totalen_2020.loc[1351,"Perioden"] = "1"
totalen_2020.loc[:,"Perioden"] = totalen_2020["Perioden"].astype(int)
# Plotten van sterftecijfers per week van 2019 en 2020 tot nu toe

# Lijn van 2019 in groen ("g")
plt.plot(totalen_2019.Perioden, totalen_2019.Overledenen_1, "g", label="2019")

# Lijn van 2020 in rood ("r")
plt.plot(totalen_2020.Perioden, totalen_2020.Overledenen_1, "r", label="2020")

# Labels op x- en y-as toevoegen
plt.xlabel("Week")
plt.ylabel("Totaal overledenen")

# Legenda toevoegen
plt.legend(loc="upper right")

# Grafiek tonen
plt.show()