# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# importer la librairie de fonctions d'aide
import bq_helper

# creer un objet aide pour le dataset
open_aq = bq_helper.BigQueryHelper (active_project= "bigquery-public-data",
                                    dataset_name= "openaq")

# afficher toutes les tables du dataset (il y en a une seule !)
open_aq.list_tables ()

# Afficher les premieres lignes (observations) du dataset "global_air_quality"
open_aq.head ("global_air_quality")
# Requete pour selectionner tous les items pour la variable "city" ou le "pays" est "us"
query = """SELECT city 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE lower (country) = 'us'
        """

# La requete BigQueryHelper.query_to_pandas_safe() retourne un resultat
# uniquement s'il est moins de 1 GB (par defaut)
us_cities = open_aq.query_to_pandas_safe(query)
# Quels villes ont-elles les plus hautes mesures?
us_cities.city.value_counts ().head (num_rows=10)
# Countries using a unit other than ppm to measure pollution
query2 = """ SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
        """
unit_other_than_ppm = open_aq.query_to_pandas_safe(query2)
# unit_other_than_ppm.head ()
unit_other_than_ppm
# Which Pollutants have exactly 0
query3 = """ SELECT DISTINCT pollutant, country, source_name 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER By country
        """
# Execute the request
pollutant_zero = open_aq.query_to_pandas_safe(query3)
# Print the pollutants that have zero 
pollutant_zero.head ()