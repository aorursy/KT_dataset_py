# This notebook helps generate an dataframe to literature review 

# load libraries
from pybtex.database.input import bibtex # pip install pybtex
import pandas as pd
# here is the function for contruct dataframe using .bib file
def construct_refs_dataframe(refs_file):
    df = pd.DataFrame(columns=['title', 'year', 'url', 'keys'])
    parser = bibtex.Parser()
    bibdata = parser.parse_file(refs_file)
    
    #loop through the individual references
    for ref_id in bibdata.entries:
        ref = bibdata.entries[ref_id].fields
        
        row = [
            {'title': ref["title"],
             'year': ref["year"],
             'url': ref["url"],
             'keys': str(ref["keywords"])
            }]


        df = df.append(row)

    return df
refs_file = "../input/ScienceDirect_uns_feat_selection_202004.bib"

ref_data = construct_refs_dataframe(refs_file)

# show the created dataframe
ref_data