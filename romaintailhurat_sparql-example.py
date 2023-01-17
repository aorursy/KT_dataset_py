import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
URL = "http://graphdb.linked-open-statistics.org/repositories/pop5"

QUERY = """
PREFIX skos:<http://www.w3.org/2004/02/skos/core#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX cog2017: <http://id.insee.fr/codes/cog2017/>
SELECT ?id ?code
WHERE {
    ?id skos:notation ?code ;
    rdf:type cog2017:DepartementOuCommuneOuArrondissementMunicipal .
} LIMIT 100
"""

# The SPARQLWrapper will store network info and query logic
hackathon_endpoint = SPARQLWrapper(URL)
hackathon_endpoint.setQuery(QUERY)

# We want the resulting data in JSON format
hackathon_endpoint.setReturnFormat(JSON)
query_res = hackathon_endpoint.query().convert()

# Data structures for storing codes and URIs
uris = []
codes = []
for results in query_res["results"]["bindings"]:
    code = results['code']['value']
    codes.append(code)
    uri = results['id']['value']
    uris.append(uri)

# Creating the data frame
pd.DataFrame(data = {"codes": codes, "uris": uris})