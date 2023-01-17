# Primero importamos Pandas y PlotLy

import pandas as pd

import plotly.graph_objects as go



# Luego cargamos el conjunto de datos

corpus = pd.read_csv("../input/corpus-bics-canarias-wikidata-commons-eswiki/corpus_bics_wikidata_commons_eswiki.csv")



# Finalmente obtenemos una visualización truncada del CSV

corpus
# Conjunto de datos ordenado por etiquetas de menor a mayor

labels = corpus.sort_values("wikidata_etiquetas")

# Diez elementos de Wikidata con menor cantidad de etiquetas

labels_less = labels[labels["wikidata_etiquetas"].notna()].head(10).reset_index(drop=True)

# Diez elementos de Wikidata con mayor cantidad de etiquetas

labels_more = labels[labels["wikidata_etiquetas"].notna()].tail(10).reset_index(drop=True)



# Conjunto de datos ordenado por descripciones de menor a mayor

descriptions = corpus.sort_values("wikidata_descripciones")

# Diez elementos de Wikidata con menor cantidad de descripciones

descriptions_less = descriptions[descriptions["wikidata_descripciones"].notna()

                                ].head(10).reset_index(drop=True)

# Diez elementos de Wikidata con mayor cantidad de descripciones

descriptions_more = descriptions[descriptions["wikidata_descripciones"].notna()

                                ].tail(10).reset_index(drop=True)
# Se crea la figura para las etiquetas

more_labels = go.Figure()



# Se añade el gráfico de barras para las etiquetas

more_labels.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(labels_more["articulo"], labels_more["wikidata_id"])],

        y=[labels for labels in labels_more["wikidata_etiquetas"]],

        name="Etiquetas"

    )

)



# Se añade el título

more_labels.update_layout(title="Elementos con mayor cantidad de etiquetas")



more_labels.show()
# Se crea la figura para las etiquetas

less_labels = go.Figure()



# Se añade el gráfico de barras para las etiquetas

less_labels.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(labels_less["articulo"], labels_less["wikidata_id"])],

        y=[labels for labels in labels_less["wikidata_etiquetas"]],

        name="Etiquetas"

    )

)



# Se añade el título

less_labels.update_layout(title="Elementos con menor cantidad de etiquetas")



less_labels.show()
same_qid = labels[labels["wikidata_id"] == "Q43171574"]



print(f'- Nº. de coincidencias del id: {same_qid["wikidata_id"].count()}\n'

      f'- Coincidencias: \n{same_qid["wikidata_id"]}\n'

      f'- Coincidencia 1 (id {same_qid["id"][173]}): {same_qid["wikidata_id"][173]}\n'

      f'- Coincidencia 2 (id {same_qid["id"][174]}): {same_qid["wikidata_id"][174]}')
# Porcentaje de elementos en relación a su cantidad de etiquetas

# Se crea la figura para los elementos

group_labels = go.Figure()



# Se añade el gráfico circular para los grupos

group_labels.add_trace(

    go.Pie(

        labels=["1", "2-5", "6-10", "11-15", "+ 15"],

        values=[labels[labels["wikidata_etiquetas"] == 1 ]["wikidata_etiquetas"].count(),

                labels[(labels["wikidata_etiquetas"] >= 2)

                           & (labels["wikidata_etiquetas"] <= 5)]["wikidata_etiquetas"].count(),

                labels[(labels["wikidata_etiquetas"] >= 6)

                           & (labels["wikidata_etiquetas"] <= 10)]["wikidata_etiquetas"].count(),

                labels[(labels["wikidata_etiquetas"] >= 11)

                           & (labels["wikidata_etiquetas"] <= 15)]["wikidata_etiquetas"].count(),

                labels[(labels["wikidata_etiquetas"] > 15)]["wikidata_etiquetas"].count()],

    )

)



# Se añade el título

group_labels.update_layout(title="Porcentaje de elementos en relación a su cantidad de etiquetas")



group_labels.show()
# Se crea la figura para las descripciones

more_descriptions = go.Figure()



# Se añade el gráfico de barras para las descripciones

more_descriptions.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(descriptions_more["articulo"], descriptions_more["wikidata_id"])],

        y=[descriptions for descriptions in descriptions_more["wikidata_descripciones"]],

        name="Descripciones"

    )

)



# Se añade el título

more_descriptions.update_layout(title="Elementos con mayor cantidad de descripciones")



more_descriptions.show()
# Se crea la figura para las descripciones

less_descriptions = go.Figure()



# Se añade el gráfico de barras para las descripciones

less_descriptions.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(descriptions_less["articulo"], descriptions_less["wikidata_id"])],

        y=[labels for labels in descriptions_less["wikidata_descripciones"]],

        name="Descripciones"

    )

)



# Se añade el título

less_descriptions.update_layout(title="Elementos con menor cantidad de descripciones")



less_descriptions.show()
# Porcentaje de elementos en relación a su cantidad de descripciones

# Se crea la figura para los elementos

group_descs = go.Figure()



# Se añade el gráfico circular para los grupos

group_descs.add_trace(

    go.Pie(

        labels=["0", "1-5", "6-10", "11-15", "+ 15"],

        values=[descriptions[descriptions["wikidata_descripciones"] == 0 ]["wikidata_descripciones"].count(),

                descriptions[(descriptions["wikidata_descripciones"] >= 1)

                           & (descriptions["wikidata_descripciones"] <= 5)]["wikidata_descripciones"].count(),

                descriptions[(descriptions["wikidata_descripciones"] >= 6)

                           & (descriptions["wikidata_descripciones"] <= 10)]["wikidata_descripciones"].count(),

                descriptions[(descriptions["wikidata_descripciones"] >= 11)

                           & (descriptions["wikidata_descripciones"] <= 15)]["wikidata_descripciones"].count(),

                descriptions[(descriptions["wikidata_descripciones"] >= 15)]["wikidata_descripciones"].count()],

    )

)



# Se añade el título

group_descs.update_layout(title="Porcentaje de elementos en relación a su cantidad de descripciones")



group_descs.show()
# Se crea la figura para la relación

more_labels_descriptions = go.Figure()



# Se añade el gráfico de barras para las etiquetas

more_labels_descriptions.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(labels_more["articulo"], labels_more["wikidata_id"])],

        y=[labels for labels in labels_more["wikidata_etiquetas"]],

        name="Etiquetas"

    )

)



# Se añade el gráfico de barras para las descripciones

more_labels_descriptions.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(labels_more["articulo"], labels_more["wikidata_id"])],

        y=[descriptions for descriptions in labels_more["wikidata_descripciones"]],

        name="Descripciones"

    )

)



# Se añade el título

more_labels_descriptions.update_layout(title="Elementos con mayor cantidad de etiquetas en relación con sus descripciones",

                                       hovermode="x")



more_labels_descriptions.show()
# Se crea la figura para la relación

more_labels_descriptions = go.Figure()



# Se añade el gráfico de barras para las descripciones

more_labels_descriptions.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(descriptions_more["articulo"], descriptions_more["wikidata_id"])],

        y=[description for description in descriptions_more["wikidata_descripciones"]],

        name="Descripciones"

    )

)



# Se añade el gráfico de barras para las etiquetas

more_labels_descriptions.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(descriptions_more["articulo"], descriptions_more["wikidata_id"])],

        y=[label for label in descriptions_more["wikidata_etiquetas"]],

        name="Etiquetas"

    )

)



# Se añade el título

more_labels_descriptions.update_layout(title="Elementos con mayor cantidad de descripciones en relación con sus etiquetas",

                                       hovermode="x")



more_labels_descriptions.show()
# Descripción estadística de la columna "wikidata_etiquetas"

labels["wikidata_etiquetas"].describe()
# Descripción estadística de la columna "wikidata_descripciones"

descriptions["wikidata_descripciones"].describe()
labels["wikidata_etiquetas"].sum(), descriptions["wikidata_descripciones"].sum()
# Conjunto de datos ordenado por declaraciones de menor a mayor

statements = corpus.sort_values("wikidata_declaraciones")

# Diez elementos de Wikidata con menor cantidad de declaraciones

statements_less = statements[statements["wikidata_declaraciones"].notna()].head(10).reset_index(drop=True)

# Diez elementos de Wikidata con mayor cantidad de declaraciones

statements_more = statements[statements["wikidata_declaraciones"].notna()].tail(10).reset_index(drop=True)



# Conjunto de datos ordenado por referencias de menor a mayor

references = corpus.sort_values("wikidata_declaraciones_referencias")

# Diez elementos de Wikidata con menor cantidad de declaraciones

references_less = references[references["wikidata_declaraciones_referencias"].notna()

                            ].head(10).reset_index(drop=True)

# Diez elementos de Wikidata con mayor cantidad de declaraciones

references_more = references[references["wikidata_declaraciones_referencias"].notna()

                            ].tail(10).reset_index(drop=True)



# Conjunto de datos ordenado por declaraciones de menor a mayor

references_p143 = corpus.sort_values("wikidata_declaraciones_referencias_P143")

# Diez elementos de Wikidata con menor cantidad de declaraciones

references_p143_less = references_p143[references_p143["wikidata_declaraciones_referencias_P143"].notna()

                                      ].head(10).reset_index(drop=True)

# Diez elementos de Wikidata con mayor cantidad de declaraciones

references_p143_more = references_p143[references_p143["wikidata_declaraciones_referencias_P143"].notna()

                                      ].tail(10).reset_index(drop=True)
# Se crea la figura para las declaraciones

more_statements = go.Figure()



# Se añade el gráfico de barras para las declaraciones

more_statements.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(statements_more["articulo"], statements_more["wikidata_id"])],

        y=[statements for statements in statements_more["wikidata_declaraciones"]],

        name="Declaraciones"

    )

)



# Se añade el título

more_statements.update_layout(title="Elementos con mayor cantidad de declaraciones")



more_statements.show()
# Se crea la figura para las declaraciones

less_statements = go.Figure()



# Se añade el gráfico de barras para las declaraciones

less_statements.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(statements_less["articulo"], statements_less["wikidata_id"])],

        y=[statements for statements in statements_less["wikidata_declaraciones"]],

        name="Declaraciones"

    )

)



# Se añade el título

less_statements.update_layout(title="Elementos con menor cantidad de declaraciones")



less_statements.show()
# Porcentaje de elementos en relación a su cantidad de declaraciones

# Se crea la figura para los elementos

group_statements = go.Figure()



# Se añade el gráfico circular para los grupos

group_statements.add_trace(

    go.Pie(

        labels=["0", "1-5", "6-10", "11-15", "+ 15"],

        values=[statements[statements["wikidata_declaraciones"] == 0 ]["wikidata_declaraciones"].count(),

                statements[(statements["wikidata_declaraciones"] >= 1)

                           & (statements["wikidata_declaraciones"] <= 5)]["wikidata_declaraciones"].count(),

                statements[(statements["wikidata_declaraciones"] >= 6)

                           & (statements["wikidata_declaraciones"] <= 10)]["wikidata_declaraciones"].count(),

                statements[(statements["wikidata_declaraciones"] >= 11)

                           & (statements["wikidata_declaraciones"] <= 15)]["wikidata_declaraciones"].count(),

                statements[(statements["wikidata_declaraciones"] >= 15)]["wikidata_declaraciones"].count()],

    )

)



# Se añade el título

group_statements.update_layout(title="Porcentaje de elementos en relación a su cantidad de declaraciones")



group_statements.show()
# Cantidad de elementos que tienen entre 1 y 5 declaraciones y tienen artículo en Wikipedia en español.

statements_eswiki = statements[(statements["wikidata_declaraciones"] >= 1)

                               & (statements["wikidata_declaraciones"] <= 5)]["articulo"].count()



statements_total = statements[(statements["wikidata_declaraciones"] >= 1)

                                    & (statements["wikidata_declaraciones"] <= 5)

                                  ]["wikidata_declaraciones"].count()



print("Elementos que tienen entre 1 y 5 declaraciones\n"

      f"Elementos con artículo: {statements_eswiki}\n"

      f"Elementos sin artículo: {statements_total - statements_eswiki}\n"

      f"Elementos en total: {statements_total}")
# Cantidad de elementos que tienen entre 1 y 5 declaraciones y tienen artículo en Wikipedia en español.

statements_eswiki = statements[(statements["wikidata_declaraciones"] >= 6)

                               & (statements["wikidata_declaraciones"] <= 10)]["articulo"].count()



statements_total = statements[(statements["wikidata_declaraciones"] >= 6)

                                    & (statements["wikidata_declaraciones"] <= 10)

                                  ]["wikidata_declaraciones"].count()



print("Elementos que tienen entre 6 y 10 declaraciones\n"

      f"Elementos con artículo: {statements_eswiki}\n"

      f"Elementos sin artículo: {statements_total - statements_eswiki}\n"

      f"Elementos en total: {statements_total}")
# Cantidad de elementos que tienen entre 11 y 15 declaraciones y tienen artículo en Wikipedia en español

statements_eswiki = statements[(statements["wikidata_declaraciones"] >= 11)

                               & (statements["wikidata_declaraciones"] <= 15)]["articulo"].count()



statements_total = statements[(statements["wikidata_declaraciones"] >= 11)

                                    & (statements["wikidata_declaraciones"] <= 15)

                                  ]["wikidata_declaraciones"].count()



print("Elementos que tienen entre 11 y 15 declaraciones\n"

      f"Elementos con artículo: {statements_eswiki}\n"

      f"Elementos sin artículo: {statements_total - statements_eswiki}\n"

      f"Elementos en total: {statements_total}")
# Cantidad de elementos que tienen más de 16 declaraciones y tienen artículo en Wikipedia en español

statements_eswiki = statements[(statements["wikidata_declaraciones"] >= 15)]["articulo"].count()



statements_total = statements[(statements["wikidata_declaraciones"] >= 15)

                             ]["wikidata_declaraciones"].count()



print("Elementos que tienen más de 16 declaraciones\n"

      f"Elementos con artículo: {statements_eswiki}\n"

      f"Elementos sin artículo: {statements_total - statements_eswiki}\n"

      f"Elementos en total: {statements_total}")
# Se crea la figura para las referencias

more_references = go.Figure()



# Se añade el gráfico de barras para las referencias

more_references.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(references_more["articulo"], references_more["wikidata_id"])],

        y=[references for references in references_more["wikidata_declaraciones_referencias"]],

        name="Referencias"

    )

)



# Se añade el título

more_references.update_layout(title="Elementos con mayor cantidad de referencias")



more_references.show()
# Se crea la figura para las referencias

less_references = go.Figure()



# Se añade el gráfico de barras para las referencias

less_references.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(references_less["articulo"], references_less["wikidata_id"])],

        y=[references for references in references_less["wikidata_declaraciones_referencias"]],

        name="Referencias"

    )

)



# Se añade el título

less_references.update_layout(title="Elementos con menos referencias")



less_references.show()
# Se crea la figura para las referencias P143

more_references_p143 = go.Figure()



# Se añade el gráfico de barras para las referencias P143

more_references_p143.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(references_p143_more["articulo"], references_p143_more["wikidata_id"])],

        y=[references for references in references_p143_more["wikidata_declaraciones_referencias_P143"]],

        name="Referencias P143"

    )

)



# Se añade el título

more_references_p143.update_layout(title="Elementos con mayor cantidad de referencias P143")



more_references_p143.show()
# Se crea la figura para las referencias P143

less_references_p143 = go.Figure()



# Se filtran los elementos con al menos una referencia P143

references_p143_less_not_zero = references_p143[references_p143["wikidata_declaraciones_referencias_P143"].notna() &

                                                references_p143["wikidata_declaraciones_referencias_P143"]].head(10).reset_index(drop=True)



# Se añade el gráfico de barras para las referencias P143

less_references_p143.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(references_p143_less_not_zero["articulo"], references_p143_less_not_zero["wikidata_id"])],

        y=[references for references in references_p143_less_not_zero["wikidata_declaraciones_referencias_P143"]],

        name="Referencias P143"

    )

)



# Se añade el título

less_references_p143.update_layout(title="Elementos con al menos 1 referencia P143")



less_references_p143.show()
# Se crea la figura para la relación

more_statements_references_p143 = go.Figure()



# Se añade el gráfico de barras para las declaraciones

more_statements_references_p143.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(statements_more["articulo"], statements_more["wikidata_id"])],

        y=[statements for statements in statements_more["wikidata_declaraciones"]],

        name="Declaraciones"

    )

)



# Se añade el gráfico de barras para las referencias

more_statements_references_p143.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(statements_more["articulo"], statements_more["wikidata_id"])],

        y=[references for references in statements_more["wikidata_declaraciones_referencias"]

           - statements_more["wikidata_declaraciones_referencias_P143"]],

        name="Referencias"

    )

)



# Se añade el gráfico de barras para las referencias P143

more_statements_references_p143.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(statements_more["articulo"], statements_more["wikidata_id"])],

        y=[references for references in statements_more["wikidata_declaraciones_referencias_P143"]],

        name="P143"

    )

)



# Se añade el título

more_statements_references_p143.update_layout(title="Elementos con mayor cantidad de declaraciones en relación con sus referencias",

                                              hovermode="x")



more_statements_references_p143.show()
# Se crea la figura para la relación

less_statements_references_p143 = go.Figure()



# Se añade el gráfico de barras para las declaraciones

less_statements_references_p143.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(statements_less["articulo"], statements_less["wikidata_id"])],

        y=[statements for statements in statements_less["wikidata_declaraciones"]],

        name="Declaraciones"

    )

)



# Se añade el gráfico de barras para las referencias

less_statements_references_p143.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(statements_less["articulo"], statements_less["wikidata_id"])],

        y=[references for references in statements_less["wikidata_declaraciones_referencias"]

          - statements_less["wikidata_declaraciones_referencias_P143"]],

        name="Referencias"

    )

)



# Se añade el gráfico de barras para las referencias P143

less_statements_references_p143.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(statements_less["articulo"], statements_less["wikidata_id"])],

        y=[references for references in statements_less["wikidata_declaraciones_referencias_P143"]],

        name="P143"

    )

)



# Se añade el título

less_statements_references_p143.update_layout(title="Elementos con menor cantidad de declaraciones en relación con sus referencias",

                                              hovermode="x")



less_statements_references_p143.show()
# Porcentaje de tipos de referencias

# Se crea la figura para las referencias

group_references_p143 = go.Figure()



# Se añade el gráfico circular para el porcentaje

group_references_p143.add_trace(

    go.Pie(

        labels=["P143", "Otras propiedades"],

        values=[statements["wikidata_declaraciones_referencias_P143"].sum(),

                statements["wikidata_declaraciones_referencias"].sum()

                - statements["wikidata_declaraciones_referencias_P143"].sum()],

    )

)



# Se añade el título

group_references_p143.update_layout(title="Porcentaje de referencias con P143 y con otras propiedades")



group_references_p143.show()
# Descripción estadística de todas las referencias

references["wikidata_declaraciones_referencias"].describe()
# Descripción estadística de las P143

references_p143["wikidata_declaraciones_referencias_P143"].describe()
# Descripción de las referencias con otras propiedades

(statements["wikidata_declaraciones_referencias"]

 - statements["wikidata_declaraciones_referencias_P143"]).describe()
# Conjunto de datos ordenado por cantidad de interwikis de menor a mayor

interwiki = corpus.sort_values("wikidata_interwiki")

# Diez elementos de Wikidata con al menos 1 interwiki

interwiki_less = interwiki[interwiki["wikidata_interwiki"] != 0].head(10).reset_index(drop=True)

# Diez elementos de Wikidata con mayor cantidad de interwikis

interwiki_more = interwiki[interwiki["wikidata_interwiki"].notna()].tail(10).reset_index(drop=True)
# Se crea la figura para los interwiki

more_interwikis = go.Figure()



# Se añade el gráfico de barras para los interwiki

more_interwikis.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(interwiki_more["articulo"], interwiki_more["wikidata_id"])],

        y=[interwikis for interwikis in interwiki_more["wikidata_interwiki"]],

        name="Interwikis"

    )

)



# Se añade el título

more_interwikis.update_layout(title="Elementos con mayor cantidad de interwikis")



more_interwikis.show()
# Se crea la figura para los interwiki

less_interwikis = go.Figure()



# Se añade el gráfico de barras para los interwiki

less_interwikis.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({wikidata_item})" for article_name, wikidata_item

           in zip(interwiki_less["articulo"], interwiki_less["wikidata_id"])],

        y=[statements for statements in interwiki_less["wikidata_interwiki"]],

        name="Interwikis"

    )

)



# Se añade el título

less_interwikis.update_layout(title="Elementos con al menos 1 interwiki")



less_interwikis.show()
# Cantidad de elementos con 1 interwiki

interwiki_one = interwiki[interwiki["wikidata_interwiki"] == 1 ]["wikidata_interwiki"].count()

interwiki_one_commons = interwiki[interwiki["wikidata_interwiki"] == 1 ]["commons_categoria"].count()



print(f"- 1 interwiki: {interwiki_one}\n"

      f"- 1 interwiki y con categoría de Wikimedia Commons: {interwiki_one_commons}\n"

      f"- 1 interwiki pero sin categoría de Wikimedia Commons: {interwiki_one - interwiki_one_commons}")
# Porcentaje de artículos según sus referencias

# Se crea la figura para los artículos con menos referencias (al menos 1)

group_interwikis = go.Figure()



# Se añade el gráfico circular para los grupos

group_interwikis.add_trace(

    go.Pie(

        labels=["0", "1-5", "6-10", "11-15", "+ 16"],

        values=[interwiki[interwiki["wikidata_interwiki"] == 0 ]["wikidata_interwiki"].count(),

                interwiki[(interwiki["wikidata_interwiki"] >= 1)

                           & (interwiki["wikidata_interwiki"] <= 5)]["wikidata_interwiki"].count(),

                interwiki[(interwiki["wikidata_interwiki"] >= 6)

                           & (interwiki["wikidata_interwiki"] <= 10)]["wikidata_interwiki"].count(),

                interwiki[(interwiki["wikidata_interwiki"] >= 11)

                           & (interwiki["wikidata_interwiki"] <= 15)]["wikidata_interwiki"].count(),

                interwiki[(interwiki["wikidata_interwiki"] >= 16)]["wikidata_interwiki"].count()],

    )

)



# Se añade el título

group_interwikis.update_layout(title="Porcentaje de interwikis")



group_interwikis.show()
# Cantidad de elementos con al menos un interwiki

interwiki[interwiki["wikidata_interwiki"] >= 1 ]["wikidata_interwiki"].count()