# Primero importamos Pandas y PlotLy

import pandas as pd

import plotly.graph_objects as go



# Luego cargamos el conjunto de datos

corpus = pd.read_csv("../input/corpus-bics-canarias-wikidata-commons-eswiki/corpus_bics_wikidata_commons_eswiki.csv")



# Finalmente obtenemos una visualización truncada del CSV

corpus
# Conjunto de datos ordenado por imágenes de menor a mayor

images = corpus.sort_values("imagenes_cantidad")

# Diez artículos con menos imágenes (al menos 1)

images_less = images[images["imagenes_cantidad"] != 0].head(10).reset_index(drop=True)

# Diez artículos con más imágenes

images_more = images[pd.notnull(images["imagenes_cantidad"])].tail(10).reset_index(drop=True)
# Se crea la figura para los artículos con más imágenes

more_images = go.Figure()



# Se añade el gráfico de barras para la cantidad de imágenes

more_images.add_trace(

    go.Bar(

        x=[article_name for article_name in images_more["articulo"]],

        y=[images for images in images_more["imagenes_cantidad"]],

        name="Imágenes"

    )

)



# Se añade el título

more_images.update_layout(title="Artículos con más imágenes")



more_images.show()
# Media de imágenes de los 10 artículos con más imágenes

images_more["imagenes_cantidad"].mean()
# Se crea la figura para los artículos con menos imágenes

less_images = go.Figure()



# Se añade el gráfico de barras para la cantidad de imágenes

less_images.add_trace(

    go.Bar(

        x=[article_name for article_name in images_less["articulo"]],

        y=[images for images in images_less["imagenes_cantidad"]],

        name="Imágenes"

    )

)



# Se añade el título

less_images.update_layout(title="Artículos con menos imágenes")



less_images.show()
# Artículos con al menos una imagen

images[images["imagenes_cantidad"] == 1]["imagenes_cantidad"].count()
# Artículos con ninguna imagen

images[images["imagenes_cantidad"] == 0]["imagenes_cantidad"].count()
# Se crea la figura para los artículos con más imágenes

more_images_size = go.Figure()



more_images_size.add_trace(

    go.Bar(

        x=[article_name for article_name in images_more["articulo"]],

        y=[images for images in images_more["imagenes_cantidad"]],

        name="Imágenes"

    )

)



# Se añade el gráfico de barras para el tamaño en palabras

more_images_size.add_trace(

    go.Bar(

        x=[article_name for article_name in images_more["articulo"]],

        y=[size_words for size_words in images_more["tamano_palabras"]],

        name="Palabras"

    )

)



# Se añade el título

more_images_size.update_layout(title="Artículos con más imágenes en relación con su tamaño en palabras",

                               hovermode="x")



more_images_size.show()
# Se crea la figura para los artículos con más imágenes

less_images_size = go.Figure()



# Se añade el gráfico de barras para la cantidad de imágenes

less_images_size.add_trace(

    go.Bar(

        x=[article_name for article_name in images_less["articulo"]],

        y=[images for images in images_less["imagenes_cantidad"]],

        name="Imágenes"

    )

)



# Se añade el gráfico de barras para el tamaño en palabras

less_images_size.add_trace(

    go.Bar(

        x=[article_name for article_name in images_less["articulo"]],

        y=[size_words for size_words in images_less["tamano_palabras"]],

        name="Palabras"

    )

)



# Se añade el título

less_images_size.update_layout(title="Artículos con al menos 1 imagen en relación con su tamaño en palabras",

                               hovermode="x")



less_images_size.show()
images["imagenes_cantidad"].describe()
# Conjunto de datos ordenado por la cantidad de archivos en Commons de menor a mayor

commons_files = corpus.sort_values("commons_archivos")

# Diez artículos con menos archivos (al menos 1)

commons_files_less = commons_files[commons_files["commons_archivos"] != 0].head(10).reset_index(drop=True)

# Diez artículos con más archivos

commons_files_more = commons_files[commons_files["commons_archivos"].notna()].tail(10).reset_index(drop=True)



# Conjunto de datos ordenado por la cantidad de subcategorías en Commons de menor a mayor

commons_subcats = corpus.sort_values("commons_subcats")

# Diez artículos con menos subcategorías (al menos 1)

commons_subcats_less = commons_subcats[commons_subcats["commons_subcats"] != 0].head(10).reset_index(drop=True)

# Diez artículos con más subcategorías

commons_subcats_more = commons_subcats[commons_subcats["commons_subcats"].notna()].tail(10).reset_index(drop=True)
# Se crea la figura para las categorías con más archivos

more_files_commons = go.Figure()



# Se añade el gráfico de barras para los archivos

more_files_commons.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({commons_category})" for article_name, commons_category 

           in zip(commons_files_more["articulo"], commons_files_more["commons_categoria"])],

        y=[files for files in commons_files_more["commons_archivos"]],

        name="Archivos"

    )

)



# Se añade el gráfico de barras para las subcategorías

more_files_commons.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({commons_category})" for article_name, commons_category 

           in zip(commons_files_more["articulo"], commons_files_more["commons_categoria"])],

        y=[files for files in commons_files_more["commons_subcats"]],

        name="Subcategorías"

    )

)



# Se añade el título

more_files_commons.update_layout(title="Cantidad de archivos en Commons en relación con sus subcategorías",

                                 hovermode="x")



more_files_commons.show()
# Se crea la figura para las categorías con más subcategorías

more_subcats_commons = go.Figure()



# Se añade el gráfico de barras para las subcategorías

more_subcats_commons.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({commons_category})" for article_name, commons_category 

           in zip(commons_subcats_more["articulo"], commons_subcats_more["commons_categoria"])],

        y=[subcats for subcats in commons_subcats_more["commons_subcats"]],

        name="Subcategorías"

    )

)



# Se añade el gráfico de barras para los archivos

more_subcats_commons.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({commons_category})" for article_name, commons_category 

           in zip(commons_subcats_more["articulo"], commons_subcats_more["commons_categoria"])],

        y=[files for files in commons_subcats_more["commons_archivos"]],

        name="Archivos"

    )

)



# Se añade el título

more_subcats_commons.update_layout(title="Cantidad de subcategorías en Commons en relación con sus archivos",

                                 hovermode="x")



more_subcats_commons.show()
# Se crea la figura para las categorías con más subcategorías

more_images_subcats_files_commons = go.Figure()



# Se descartan los artículos que no tengan categoría en Commons

images_more = images[images["commons_categoria"].notna() 

                     & images["imagenes_cantidad"].notna()].tail(10).reset_index(drop=True)



# Se añade el gráfico de barras para las imágenes

more_images_subcats_files_commons.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({commons_category})" for article_name, commons_category 

           in zip(images_more["articulo"], images_more["commons_categoria"])],

        y=[images for images in images_more["imagenes_cantidad"]],

        name="Imágenes"

    )

)



# Se añade el gráfico de barras para los archivos

more_images_subcats_files_commons.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({commons_category})" for article_name, commons_category 

           in zip(images_more["articulo"], images_more["commons_categoria"])],

        y=[files for files in images_more["commons_archivos"]],

        name="Archivos"

    )

)



# Se añade el gráfico de barras para las subcategorías

more_images_subcats_files_commons.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({commons_category})" for article_name, commons_category 

           in zip(images_more["articulo"], images_more["commons_categoria"])],

        y=[files for files in images_more["commons_subcats"]],

        name="Subcategorías"

    )

)



# Se añade el título

more_images_subcats_files_commons.update_layout(title="Cantidad de imágenes en artículos en relación con las subcategorías y archivos en Commons",

                                 hovermode="x")



more_images_subcats_files_commons.show()
# Se crea la figura para las categorías con más subcategorías

less_images_subcats_files_commons = go.Figure()



# Se descartan los artículos que no tengan categoría en Commons

commons_subcats_less = commons_subcats[commons_subcats["commons_subcats"] != 0].head(10).reset_index(drop=True)



images_less_not_na = images[images["commons_categoria"].notna() &

                            images["imagenes_cantidad"] != 0].head(10).reset_index(drop=True)



# Se añade el gráfico de barras para las imágenes

less_images_subcats_files_commons.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({commons_category})" for article_name, commons_category 

           in zip(images_less_not_na["articulo"], images_less_not_na["commons_categoria"])],

        y=[images for images in images_less_not_na["imagenes_cantidad"]],

        name="Imágenes"

    )

)



# Se añade el gráfico de barras para los archivos

less_images_subcats_files_commons.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({commons_category})" for article_name, commons_category 

           in zip(images_less_not_na["articulo"], images_less_not_na["commons_categoria"])],

        y=[files for files in images_less_not_na["commons_archivos"]],

        name="Archivos"

    )

)



# Se añade el gráfico de barras para las subcategorías

less_images_subcats_files_commons.add_trace(

    go.Bar(

        x=[f"{article_name}<br>({commons_category})" for article_name, commons_category 

           in zip(images_less_not_na["articulo"], images_less_not_na["commons_categoria"])],

        y=[files for files in images_less_not_na["commons_subcats"]],

        name="Subcategorías"

    )

)



# Se añade el título

less_images_subcats_files_commons.update_layout(title="Artículos con al menos 1 imagen en relación con las subcategorías y archivos en Commons",

                                 hovermode="x")



less_images_subcats_files_commons.show()
# Suma total de todos los archivos

commons_files["commons_archivos"].sum()
# Cantidad de elementos sin categoría en Wikimedia Commons

print(f"Sin categoría: {corpus['commons_categoria'].isna().sum()}\n"

      f"Total de artículos con categoría: {corpus['commons_categoria'].notna().sum()}")