import pandas as pd
# Daten einlesen



data = pd.read_csv("../input/dsia19-california-housing/housing.csv")
# ersten N Datensätze anzeigen



data.head(4)
# letzten N Datensätze anzeigen



data.tail(4)
# Kurzinfo mit Datentypen für alle Spalten



data.info()
# deskriptive Statistiken aller nummerischen Spalten



data.describe()
# ein Feature selektieren



data.longitude.head(4)
# mehrere Spalten selektieren



data[["longitude", "latitude"]].head(4)
# Platz für euren Code