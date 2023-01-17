import pandas as pd

import plotly.express as px
SystensComponents = pd.read_excel ("../input/poos-presal-2020-5/dados classificacao de pocos 2020 5.xlsx")
SystensComponents = SystensComponents.drop_duplicates()
SystensComponents
SytensClassification = pd.DataFrame([SystensComponents["ANP"],SystensComponents["Metano"]+SystensComponents["Nitrogênio"],

                                     SystensComponents["Etano"]+SystensComponents["Propano"]+SystensComponents["Iso-Butano"]

                                     +SystensComponents["Butano"]+SystensComponents["Iso-Pentano"]+SystensComponents["n-Pentano"]

                                     +SystensComponents["Hexanos"]+SystensComponents["Gás Carbônico"], SystensComponents["Heptanos"]

                                     +SystensComponents["Octanos"]+SystensComponents["Nonanos"]+SystensComponents["Decanos"]

                                     +SystensComponents["Undecanos"]+SystensComponents["Oxigênio"]]).transpose()

SytensClassification.columns =["Nome Poço ANP","C₁, N₂","C₂ - C₆, CO₂", "C₇+, O₂"]
SytensClassification
def classifier(SytensClassification):

    lista = []

    for index, row in SytensClassification.iterrows():

        if row["C₁, N₂"] <= 91 and 0 <= row["C₂ - C₆, CO₂"] <= 100 and row["C₇+, O₂"] <= 12.5:

            lista.append("Gás condensado")

        if row["C₁, N₂"] <= 88 and 0 <= row["C₂ - C₆, CO₂"] <= 88 and 30 <= row["C₇+, O₂"] <= 12.5:

            lista.append("Óleo volátil")

        if row["C₁, N₂"] <= 70 and 0 <= row["C₂ - C₆, CO₂"] <= 70 and 30 <= row["C₇+, O₂"] <= 70:

            lista.append("Black Oil")

        if row["C₁, N₂"] <= 30 and 0 <= row["C₂ - C₆, CO₂"] <= 30 and row["C₇+, O₂"] >= 70:

            lista.append("Low shrinkage oil system")

        if row["C₁, N₂"] >= 91  and 0 <= row["C₂ - C₆, CO₂"] <= 88 and 30 <= row["C₇+, O₂"] <= 12.5:

            lista.append("Dry gas system")

    return lista


SytensClassification["Classificação"] = classifier(SytensClassification)

SytensClassification
fig = px.scatter_ternary(SytensClassification, a="C₁, N₂", c="C₂ - C₆, CO₂", b="C₇+, O₂", hover_name="Classificação", symbol = 'Nome Poço ANP')

fig.show()