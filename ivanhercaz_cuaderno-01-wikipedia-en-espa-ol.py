# Primero importamos Pandas y PlotLy

import pandas as pd

import plotly.graph_objects as go



# Luego cargamos el conjunto de datos

corpus = pd.read_csv("../input/corpus-bics-canarias-wikidata-commons-eswiki/corpus_bics_wikidata_commons_eswiki.csv")



# Finalmente obtenemos una visualización truncada del CSV

corpus
# Número de artículos en Wikipedia en español

corpus[pd.notnull(corpus["articulo"])]["articulo"].count()
# Elementos de Wikidata de BIC sin artículo en Wikipedia en español

corpus["wikidata_id"].count() - corpus[pd.notnull(corpus["articulo"])]["articulo"].count()
# Conjunto de datos ordenados por tamaño en bytes de menor a mayor

bytes_col = corpus.sort_values("tamano_bytes")

# Primeros diez artículos (menores)

bytes_smaller = bytes_col.head(10).reset_index(drop=True)

# Últimos diez artículos (mayores)

bytes_bigger = bytes_col[

    pd.notnull(bytes_col["tamano_bytes"])].tail(10).reset_index(drop=True)



# Conjunto de datos ordenados por tamaño en palabras de menor a mayor

words_col = corpus.sort_values("tamano_palabras")

words_smaller = words_col.head(10).reset_index(drop=True)

words_bigger = words_col[

    pd.notnull(words_col["tamano_palabras"])].tail(10).reset_index(drop=True)
# Se crea la figura para los artículos de mayor tamaño en bytes

bigger_articles = go.Figure()



# Se añade el gráfico de barras con los 10 artículos de mayor tamaño

bigger_articles.add_trace(

    go.Bar(

        x=[article_name for article_name in bytes_bigger["articulo"]],

        y=[byte_size for byte_size in bytes_bigger["tamano_bytes"]],

        name="Bytes"

    )

)



# Se añade el gráfico de barras con los tamaños en palabras de los 10 mayores artículos

bigger_articles.add_trace(

    go.Bar(

        x=[article_name for article_name in bytes_bigger["articulo"]],

        y=[word_size for word_size in bytes_bigger["tamano_palabras"]],

        name="Palabras"

    )

)



# Se añade el título

bigger_articles.update_layout(title="10 artículos de mayor tamaño en bytes en relación con su tamaño en palabras",

                              hovermode="x")



# Muestra el gráfico

bigger_articles.show()
# Se crea la figura para los artículos de menor tamaño en bytes

smaller_articles = go.Figure()



# Se añade el gráfico de barras con los 10 artículos de menor tamaño

smaller_articles.add_trace(

    go.Bar(

        x=[article_name for article_name in bytes_smaller["articulo"]],

        y=[byte_size for byte_size in bytes_smaller["tamano_bytes"]],

        name="Bytes"

    )

)



# Se añade el gráfico de barras con los tamaños en palabras de los 10 mayores artículos

smaller_articles.add_trace(

    go.Bar(

        x=[article_name for article_name in bytes_smaller["articulo"]],

        y=[word_size for word_size in bytes_smaller["tamano_palabras"]],

        name="Palabras"

    )

)



# Se añade el título

smaller_articles.update_layout(title="10 artículos de menor tamaño en bytes en relación con su tamaño en palabras",

                               hovermode="x")



# Muestra el gráfico

smaller_articles.show()
# Se crea la figura para los artículos de mayor tamaño en palabras

bigger_articles = go.Figure()



# Se añade el gráfico de barras con los 10 artículos de mayor tamaño

bigger_articles.add_trace(

    go.Bar(

        x=[article_name for article_name in words_bigger["articulo"]],

        y=[byte_size for byte_size in words_bigger["tamano_palabras"]],

        name="Palabras"

    )

)



# Se añade el gráfico de barras con los tamaños en bytes de los 10 mayores artículos

bigger_articles.add_trace(

    go.Bar(

        x=[article_name for article_name in words_bigger["articulo"]],

        y=[word_size for word_size in words_bigger["tamano_bytes"]],

        name="Bytes"

    )

)



# Se añade el título

bigger_articles.update_layout(title="10 artículos de mayor tamaño en palabras en relación con su tamaño en bytes",

                  hovermode="x")



bigger_articles.show()
# Se crea la figura para los artículos de menos tamaño en palabras

smaller_articles = go.Figure()



# Se añade el gráfico de barras con los 10 artículos de menos tamaño

smaller_articles.add_trace(

    go.Bar(

        x=[article_name for article_name in words_smaller["articulo"]],

        y=[byte_size for byte_size in words_smaller["tamano_palabras"]],

        name="Palabras"

    )

)



# Se añade el gráfico de barras con los tamaños en bytes de los 10 artículos menores

smaller_articles.add_trace(

    go.Bar(

        x=[article_name for article_name in words_smaller["articulo"]],

        y=[word_size for word_size in words_smaller["tamano_bytes"]],

        name="Bytes"

    )

)



# Se añade el título

smaller_articles.update_layout(title="10 artículos de menor tamaño en palabras en relación con su tamaño en bytes",

                  hovermode="x")



smaller_articles.show()
bytes_col["tamano_bytes"].describe()
words_col["tamano_palabras"].describe()
# Conjunto de datos ordenados por fecha de creación de más antiguo a más reciente

date_data = corpus.sort_values("fecha_creacion")

# Primeros diez artículos (más antiguos)

date_older = date_data.head(10).reset_index(drop=True)

# Últimos diez artículos (más recientes)

date_recent = date_data[pd.notnull(date_data["fecha_creacion"])].tail(10).reset_index(drop=True)
# Se crea la figura para los artículos más antiguos

older_articles = go.Figure()



# Se añade el gráfico de dispersión para la fecha de creación

older_articles.add_trace(

    go.Scatter(

        x=[article_name for article_name in date_older["articulo"]],

        y=[date for date in date_older["fecha_creacion"]],

        mode="markers",

        marker=dict(size=[15 for article in date_older["articulo"]]),

        name="Fecha de creación"

    )

)



# Se añade el gráfico de dispersión para la fecha de su última revisión

older_articles.add_trace(

    go.Scatter(

        x=[article_name for article_name in date_older["articulo"]],

        y=[date for date in date_older["fecha_ultima_revision"]],

        mode="markers",

        marker=dict(size=[15 for article in date_older["articulo"]]),

        name="Última revisión"

    )

)



# Se añade el título

older_articles.update_layout(title="Artículos más antiguos en relación con su última revisión",

                  hovermode="x")



older_articles.show()
# Se crea la figura para los artículos más recientes

recent_articles = go.Figure()



# Se añade el gráfico para las fechas de creación

recent_articles.add_trace(

    go.Scatter(

        x=[article_name for article_name in date_recent["articulo"]],

        y=[date for date in date_recent["fecha_creacion"]],

        mode="markers",

        marker=dict(size=[15 for article in date_recent["articulo"]]),

        name="Fecha de creación"

    )

)



# Se añade el gráfico para las fechas de su última revisión

recent_articles.add_trace(

    go.Scatter(

        x=[article_name for article_name in date_recent["articulo"]],

        y=[date for date in date_recent["fecha_ultima_revision"]],

        mode="markers",

        marker=dict(size=[15 for article in date_recent["articulo"]]),

        name="Última revisión"

    )

)



recent_articles.update_yaxes(tickformat="%Y-%m-%d")



# Se añade el título

recent_articles.update_layout(title="Artículos más recientes en relación con su última revisión",

                  hovermode="x")



recent_articles.show()
# Conjunto de datos ordenados por cantidad de editores registrados de menor a mayor

registered_data = corpus.sort_values("editores_registrados")

registered_less = registered_data.head(10).reset_index(drop=True)

registered_more = registered_data[

    pd.notnull(registered_data["editores_registrados"])].tail(10).reset_index(drop=True)



# Conjunto de datos ordenados por cantidad de editores anónimos de menor a mayor

anonymous_data = corpus.sort_values("editores_anonimos")

anonymous_less = anonymous_data.head(10).reset_index(drop=True)

anonymous_more = anonymous_data[

    pd.notnull(anonymous_data["editores_anonimos"])].tail(10).reset_index(drop=True)
# Se crea la figura para los artículos con más editores

more_registered = go.Figure()



# Se añade el gráfico de barras para los editores registrados

more_registered.add_trace(

    go.Bar(

        x=[article_name for article_name in registered_more["articulo"]],

        y=[registered for registered in registered_more["editores_registrados"]],

        name="Registrados"

    )

)



# Se añade el gráfico de barras para los editores anónimos

more_registered.add_trace(

    go.Bar(

        x=[article_name for article_name in registered_more["articulo"]],

        y=[registered for registered in registered_more["editores_anonimos"]],

        name="Anónimos"

    )

)



# Se añade el título

more_registered.update_layout(title="Artículos con mayor cantidad de editores registrados en relación con anónimos",

                              hovermode="x")



more_registered.show()
# Se crea la figura para los artículos con más editores

less_registered = go.Figure()



# Se añade el gráfico de barras para los editores registrados

less_registered.add_trace(

    go.Bar(

        x=[article_name for article_name in registered_less["articulo"]],

        y=[registered for registered in registered_less["editores_registrados"]],

        name="Registrados"

    )

)



# Se añade el gráfico de barras para los editores anónimos

less_registered.add_trace(

    go.Bar(

        x=[article_name for article_name in registered_less["articulo"]],

        y=[registered for registered in registered_less["editores_anonimos"]],

        name="Anónimos"

    )

)



# Se añade el título

less_registered.update_layout(title="Artículos con menor cantidad de editores registrados en relación con anónimos",

                              hovermode="x")



less_registered.show()
# Se crea la figura para los artículos con más editores

more_anonymous = go.Figure()



# Se añade el gráfico de barras para los editores registrados

more_anonymous.add_trace(

    go.Bar(

        x=[article_name for article_name in anonymous_more["articulo"]],

        y=[registered for registered in anonymous_more["editores_anonimos"]],

        name="Anónimos"

    )

)



# Se añade el gráfico de barras para los editores anónimos

more_anonymous.add_trace(

    go.Bar(

        x=[article_name for article_name in anonymous_more["articulo"]],

        y=[registered for registered in anonymous_more["editores_registrados"]],

        name="Registrados"

    )

)



# Se añade el título

more_anonymous.update_layout(title="Artículos con mayor cantidad de editores anónimos en relación con registrados",

                              hovermode="x")



more_anonymous.show()
# Se crea la figura para los artículos con más editores

less_anonymous = go.Figure()



# Se añade el gráfico de barras para los editores registrados

less_anonymous.add_trace(

    go.Bar(

        x=[article_name for article_name in anonymous_less["articulo"]],

        y=[registered for registered in anonymous_less["editores_anonimos"]],

        name="Anónimos"

    )

)



# Se añade el gráfico de barras para los editores anónimos

less_anonymous.add_trace(

    go.Bar(

        x=[article_name for article_name in anonymous_less["articulo"]],

        y=[registered for registered in anonymous_less["editores_registrados"]],

        name="Registrados"

    )

)



# Se añade el título

less_anonymous.update_layout(title="Artículos con menor cantidad de editores anónimos en relación con registrados",

                              hovermode="x")



less_anonymous.show()
# Editores registrados

registered_data["editores_registrados"].mean()
# Editores anónimos

anonymous_data["editores_anonimos"].mean()
# Conjunto de datos ordenado por referencias de menor a mayor

references = corpus.sort_values("referencias")

# Diez artículos con menos referencias (al menos 1)

references_less = references[

    references["referencias"] != 0].head(10).reset_index(drop=True)

# Diez artículos con más referencias

references_more = references[

    pd.notnull(references["referencias"])].tail(10).reset_index(drop=True)
# Se crea la figura para los artículos con más referencias

more_references = go.Figure()



# Se añade el gráfico de barras para las referencias

more_references.add_trace(

    go.Bar(

        x=[article_name for article_name in references_more["articulo"]],

        y=[references for references in references_more["referencias"]],

        name="Referencias"

    )

)



# Se añade el título

more_references.update_layout(title="Artículos con más referencias")



more_references.show()
# Se crea la figura para los artículos con menos referencias (al menos 1)

less_references = go.Figure()



# Se añade el gráfico de barras para las referencias

less_references.add_trace(

    go.Bar(

        x=[article_name for article_name in references_less["articulo"]],

        y=[references for references in references_less["referencias"]],

        name="Referencias"

    )

)



# Se añade el título

less_references.update_layout(title="Artículos con menos referencias (al menos 1)")



less_references.show()
# Se crea la figura para los artículos con más referencias

more_words_references = go.Figure()



hover_text = [f"Palabras: {words}<br>Referencias: {refs}"

              for words, refs in zip(words_bigger["tamano_palabras"], words_bigger["referencias"])]



# Se añade el gráfico para las fechas de creación

more_words_references.add_trace(

    go.Scatter(

        x=[article_name for article_name in words_bigger["articulo"]],

        y=[references for references in words_bigger["referencias"]],

        text=hover_text,

        mode='markers',

        marker=dict(

            opacity=0.8,

            size=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55],

        )

    )

)

    

more_words_references.update_layout(title="Artículos con mayor cantidad de palabras en relación con sus referencias",)





more_words_references.show()
# Se crea la figura para los artículos con más referencias

less_words_references = go.Figure()



hover_text = [f"Palabras: {words}<br>Referencias: {refs}"

              for words, refs in zip(words_smaller["tamano_palabras"], words_smaller["referencias"])]



# Se añade el gráfico para las fechas de creación

less_words_references.add_trace(

    go.Scatter(

        x=[article_name for article_name in words_smaller["articulo"]],

        y=[references for references in words_smaller["referencias"]],

        text=hover_text,

        mode='markers',

        marker=dict(

            opacity=0.8,

            size=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55],

        )

    )

)

    

less_words_references.update_layout(title="Artículos con menor cantidad de palabras en relación con sus referencias",)





less_words_references.show()
# Se crea la figura para los artículos con más referencias

more_references = go.Figure()



# Se añade el gráfico de barras para las referencias

more_references.add_trace(

    go.Bar(

        x=[article_name for article_name in references_more["articulo"]],

        y=[references for references in references_more["referencias"]],

        name="Referencias"

    )

)



# Se añade el gráfico de barras para las referencias

more_references.add_trace(

    go.Bar(

        x=[article_name for article_name in references_more["articulo"]],

        y=[references for references in references_more["bibliografía"]],

        name="Bibliografía"

    )

)



# Se añade el título

more_references.update_layout(title="Artículos con más referencias en relación con su bibliografía",

                              hovermode="x")



more_references.show()
# Media de referencias

references["referencias"].mean()
# Porcentaje de artículos según sus referencias

# Se crea la figura para los artículos con menos referencias (al menos 1)

group_references = go.Figure()



# Se añade el gráfico circular para los grupos

group_references.add_trace(

    go.Pie(

        labels=["0", "1-5", "6-10", "11-20", "+ 21"],

        values=[references[references["referencias"] == 0 ]["referencias"].count(),

                references[(references["referencias"] >= 1)

                           & (references["referencias"] <= 5)]["referencias"].count(),

                references[(references["referencias"] >= 6)

                           & (references["referencias"] <= 10)]["referencias"].count(),

                references[(references["referencias"] >= 11)

                           & (references["referencias"] <= 20)]["referencias"].count(),

                references[(references["referencias"] >= 21)]["referencias"].count()],

    )

)



# Se añade el título

group_references.update_layout(title="Porcentaje de artículos en relación al número de referencias")



group_references.show()
# Conjunto de datos ordenado por bibliografía de menor a mayor

bibliography = corpus.sort_values("bibliografía")

# Diez artículos con menos bibliografía (al menos 1)

bibliography_less = bibliography[

    bibliography["bibliografía"] != 0].head(10).reset_index(drop=True)

# Diez artículos con más bibliografía

bibliography_more = bibliography[

    pd.notnull(bibliography["bibliografía"])].tail(10).reset_index(drop=True)
# Se crea la figura para los artículos con más bibliografía

more_bibliography = go.Figure()



# Se añade el gráfico de barras para la bibliografía

more_bibliography.add_trace(

    go.Bar(

        x=[article_name for article_name in bibliography_more["articulo"]],

        y=[bibliography for bibliography in bibliography_more["bibliografía"]],

        name="Bibliografía"

    )

)



# Se añade el título

more_bibliography.update_layout(title="Artículos con más bibliografía")



more_bibliography.show()
# Se crea la figura para los artículos con al menos 1 elemento en su bibliografía

less_bibliography = go.Figure()



# Se añade el gráfico de barras para la bibliografía

less_bibliography.add_trace(

    go.Bar(

        x=[article_name for article_name in bibliography_less["articulo"]],

        y=[bibliography for bibliography in bibliography_less["bibliografía"]],

        name="Bibliografía"

    )

)



# Se añade el título

less_bibliography.update_layout(title="Artículos con al menos 1 elemento en su bibliografía")



less_bibliography.show()
# Se crea la figura para los artículos con más bibliografía en relación con sus referencias

more_bibliography = go.Figure()



# Se añade el gráfico de barras para la bibliografía

more_bibliography.add_trace(

    go.Bar(

        x=[article_name for article_name in bibliography_more["articulo"]],

        y=[references for references in bibliography_more["bibliografía"]],

        name="Bibliografía"

    )

)



# Se añade el gráfico de barras para las referencias

more_bibliography.add_trace(

    go.Bar(

        x=[article_name for article_name in bibliography_more["articulo"]],

        y=[references for references in bibliography_more["referencias"]],

        name="Referencias"

    )

)



# Se añade el título

more_bibliography.update_layout(title="Artículos con más bibliografía en relación con sus referencias",

                              hovermode="x")



more_bibliography.show()
# Se crea la figura para los artículos con al menos 1 elemento en su bibliografía

# en relación con sus referencias

less_bibliography = go.Figure()



# Se añade el gráfico de barras para la bibliografía

less_bibliography.add_trace(

    go.Bar(

        x=[article_name for article_name in bibliography_less["articulo"]],

        y=[bibliography for bibliography in bibliography_less["bibliografía"]],

        name="Bibliografía"

    )

)



# Se añade el gráfico de barras para las referencias

less_bibliography.add_trace(

    go.Bar(

        x=[article_name for article_name in bibliography_less["articulo"]],

        y=[bibliography for bibliography in bibliography_less["referencias"]],

        name="Referencias"

    )

)



# Se añade el título

less_bibliography.update_layout(title="Artículos con al menos 1 elemento en su bibliografía en relación con sus referencias",

                                hovermode="x")



less_bibliography.show()