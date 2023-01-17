import pandas as pd

bienes = pd.read_csv("../input/bienes-secuestrados-decomisados-durante-proceso-penal-20191231.csv")
cantidad_por_juzgado = bienes.groupby("juzgado").juzgado.count().sort_values(ascending=False)

cantidad_por_juzgado
cantidad_por_tipo = bienes.groupby("tipo_bien").tipo_bien.count().sort_values(ascending=False)

cantidad_por_tipo
cantidad_por_delito = bienes.groupby("tipo_delito").tipo_delito.count().sort_values(ascending=False)

cantidad_por_delito
bienes_segun_delito = bienes.groupby(["tipo_delito", "tipo_bien"]).tipo_bien.count().sort_values(ascending=False).head(20)

bienes_segun_delito