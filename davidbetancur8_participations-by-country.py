import os

import json

from tqdm import tqdm

import pandas as pd

from collections import Counter

import pycountry

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go
cord_path = '../input/CORD-19-research-challenge'

dirs = ["biorxiv_medrxiv", "comm_use_subset", "noncomm_use_subset", "custom_license"]
docs = []

for d in dirs:

    for file in tqdm(os.listdir(f"{cord_path}/{d}/{d}")):

        file_path = f"{cord_path}/{d}/{d}/{file}"

        j = json.load(open(file_path, "rb"))



        title = j["metadata"]["title"]

        authors = j["metadata"]["authors"]



        try:

            abstract = j["abstract"][0]

        except:

            abstract = ""



        full_text = ""

        for text in j["body_text"]:

            full_text += text["text"] + "\n\n"

        docs.append([title, authors, abstract, full_text])



df = pd.DataFrame(docs, columns=["title", "authors", "abstract", "full_text"])
paises = []

for autores in tqdm(df.authors):

    for autor in autores:

        try:

            pais = autor["affiliation"]["location"]["country"]

            pais = pais.lower()

            pais = pais.replace(".", "")

            pais = pais.replace(";", "")

            pais = ''.join([i for i in pais if not i.isdigit()])

            pais = pais.split(",")[0]

            paises_errores = ["china", "congo", "brasil", "korea", "italy", 

                              "german", "uk", "taiwan","poland", "usa"]

            for err in paises_errores:

                if err in pais:

                    pais = err

            if "uk" in pais:

                pais = "united kingdom"

            paises.append(pais)

        except:

            pass
country_counts = Counter(paises)

country_counts
names = []

codes = []

counts = []

for country in country_counts:

    try:

        c = pycountry.countries.search_fuzzy(country)[0]

        names.append(c.name)

        codes.append(c.alpha_3)

        counts.append(country_counts[country])

    except:

        pass

              
df_counts = pd.DataFrame({"names":names, "codes":codes, "counts":counts})

df_counts = pd.DataFrame(df_counts.groupby(["names", "codes"])["counts"].sum()).reset_index()

df_counts.head()
data=dict(

    type = 'choropleth',

    locations = df_counts['codes'],

    z = df_counts['counts'],

    text = df_counts['names'],

    colorscale = 'YlOrRd',

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_title = 'Number of participations',

)



layout = dict(title_text='Participations in articles about Coronavirus',

    geo=dict(

        showframe=False,

        showcoastlines=True,

        projection_type='equirectangular'

    ))



fig = go.Figure(data = [data], layout = layout)

iplot(fig)