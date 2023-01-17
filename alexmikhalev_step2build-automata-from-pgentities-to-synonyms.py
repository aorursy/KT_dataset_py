#!/usr/bin/env python
# coding: utf-8

# %conda install -y psycopg2
"""
entities_dict created out of dump in previous step using:

CREATE UNLOGGED TABLE entities_dict as select ent_text,
canonical_name, concept_id, unnest(string_to_array(synonims,'|')) 
as syn_term from entities_synonims;
create index idx_entities_dict_name_id
ON entities_dict(canonical_name,concept_id);
"""

import psycopg2
from config import config

import joblib
import ahocorasick

A = ahocorasick.Automaton()

conn = None
print("Building automata")
try:
    # read the connection parameters
    params = config()
    # connect to the PostgreSQL server
    conn = psycopg2.connect(**params)
    # named cursor is important to fetch in batches of 2000, otherwise will run out of memory
    cursor = conn.cursor("secred dict")
    cursor.execute("select ent_text, syn_term, concept_id from entities_dict")
    # print("Print each row and it's columns values")
    for row in cursor:
        ent_text, syn_term,concept_id = row
        # print(f"{ent_text} syn term {syn_term} and concept_id {concept_id}")
        if len(ent_text)>4:
            A.add_word(ent_text, (concept_id,syn_term))
    # close communication with the PostgreSQL database server
    cursor.close()
    # commit the changes
    conn.commit()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()

A.make_automaton()
print("Aho corasic automata ")
print(A.get_stats())

joblib.dump(A,"./kaggle/working/automata_ent_only.pkl")
print("Automata Saved")

