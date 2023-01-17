import pandas as pd
df = pd.read_csv("../input/karl-fristons-papers-metadata/frisons_papers_metadata.csv")

df.head()
df_first_author = df[df['authorString'].str.startswith('Friston')]

len(df_first_author)
df_first_author_oa = df_first_author[df_first_author['isOpenAccess']]

len(df_first_author_oa)
import requests

url_tmpl = "https://www.ebi.ac.uk/europepmc/webservices/rest/{id}/fullTextXML"

for id in df_first_author_oa['id'].to_list():

    r = requests.get(url=url_tmpl.format(id=id))

    assert r.status_code == 200, "Downloading {id} failed.".format(id=id)

    with open(id + '.xml','w') as f:

        f.write(r.text)

        
!ls