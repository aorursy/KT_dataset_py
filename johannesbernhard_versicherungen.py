import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.stats import gamma, pareto



plt.style.use("ggplot")
# https://www.gdv.de/de/zahlen-und-fakten/versicherungsbereiche/ueberblick-24074

beitraege = pd.read_csv("../input/versicherungends/Beitraege.csv", header=[0, 1], sep=";", nrows=13, decimal=",")

leistungen = pd.read_csv("../input/versicherungends/Leistungen.csv", header=[0, 1], sep=";", nrows=13, decimal=",")
beitraege.columns = ["VERSICHERUNGSSPARTE", "2017", "2018", "VERAENDERUNG"]

leistungen.columns = ["VERSICHERUNGSSPARTE", "2017", "2018", "VERAENDERUNG"]
for df in [beitraege, leistungen]:

    for jahr in ["2017", "2018"]:

        df[jahr] = df[jahr].str.replace(".", "").astype(int)

    df.VERAENDERUNG = df.VERAENDERUNG.str.replace(",", ".").str.replace("%", "").astype(float) / 100

    df.set_index("VERSICHERUNGSSPARTE", inplace=True)
beitraege
leistungen
df = pd.concat([beitraege["2018"], leistungen["2018"]], axis=1)

df.columns = ["BEITRAEGE", "LEISTUNGEN"]
df["LEISTUNGSQUOTE"] = df.LEISTUNGEN / df.BEITRAEGE
df
leistungsquote_sonstige_sachversicherungen = (7420 - 5969 - 1274) / (11319 - 7669 - 3142)

leistungsquote_sonstige_sachversicherungen = 0.7 # uebertrieben optimistisches Szenario

print("Erwartungswert je eingesetzten Euro", leistungsquote_sonstige_sachversicherungen)
faktor_sonstige_sachversicherungen = 1 / leistungsquote_sonstige_sachversicherungen
pd.Series(gamma.rvs(7, 1, size=1_000_000) * 20).hist(bins=100, figsize=(20, 9))
leistungen_handyversicherung = pd.Series(np.append(gamma.rvs(7, 1, size=1_000_000) * 20, [0]*9_000_000))
leistungen_handyversicherung.hist(bins=50, figsize=(20, 9))
durchschnittliche_leistung = leistungen_handyversicherung.mean()

print("Durchschnittliche Leistung", durchschnittliche_leistung)
durchschnittlicher_beitrag = leistungen_handyversicherung.mean()*faktor_sonstige_sachversicherungen

print("Durchschnittlicher Beitrag", durchschnittlicher_beitrag)
print("Durchschnittlicher Verlust", durchschnittlicher_beitrag - durchschnittliche_leistung)
print("Durchschnittlicher Verlust pro Euro Beitrag", 1-leistungsquote_sonstige_sachversicherungen)
print("Anteil Versicherungsnehmer die Verlust machen", (leistungen_handyversicherung < durchschnittlicher_beitrag).mean())
for quote_ohne_leistung in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

    anzahl_mit_leistung = int((1-quote_ohne_leistung)*10_000_000)

    anzahl_ohne_leistung = int(quote_ohne_leistung*10_000_000)

    leistungen_handyversicherung = pd.Series(np.append(gamma.rvs(7, 1, size=anzahl_mit_leistung) * 20, [0]*anzahl_ohne_leistung))

    grenzwert_sonstige_sachversicherungen = leistungen_handyversicherung.mean()*faktor_sonstige_sachversicherungen

    quote_nettozahler = (leistungen_handyversicherung < grenzwert_sonstige_sachversicherungen).mean()

    print("Quote ohne Leistung", quote_ohne_leistung, ", Quote Nettozahler", quote_nettozahler)