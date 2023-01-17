# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input/cord19-expertsystem-mesh/cord19_expertsystem_mesh_060320'):

    if 'json' in dirname:

        print(f"{'/'.join(dirname.split('/')[-2:])} has {len(filenames)} files")

        #uncomment to list all files under the input directory

        #for filename in filenames:  

            #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import logging

import json

from pathlib import Path

from pprint import pprint



logger = logging.getLogger(__name__)



logging.basicConfig(format="%(asctime)s %(levelname)s [PID: %(process)d - %(filename)s %(funcName)s] - %(message)s",

                    level=logging.INFO)

logger.info('start')
# local functions

def load_taxonomies(dataset_root):

    """Given a path with the root directory, we navigate the path and read all the files having .json as extension.

    Filename by filename, we build up a dictionary in the form {key : value, key1 : value1, ...} in which the key is the name of an entry in MeSH taxonomy, while the value is a set of paper ids for all the papers that contain that entry"""

    tax_in_files = {}



    for json_file in dataset_root.glob('**/*'):

        if json_file.is_dir():

            continue

        if not json_file.name.lower().endswith('.json'):

            logger.warning(f'unused file {json_file.name}')

            continue

        json_data = json.load(json_file.open())

        for tax_entry in json_data.get('MeSH_tax', []):

            tax = tax_entry['tax']

            tax_in_files[tax] = tax_in_files.get(tax, set())

            tax_in_files[tax].add(json_data['paper_id'])

    return tax_in_files



def filter_data(tax_in_files, taxonomy_entry):

    """Takes as input the dictionary built in load_taxonomies, tax_in_files, and a specific taxonomy entry.

    Returns a new dictionary tax_in_files in which only the entries that have documents in common with the latest taxonomy entry are maintained. 

    In other words, the function can be called progressively on multiple taxonomies entries, so to filter the dictionary and have, in the end, only the sets of documents that share the entries all of interest."""

    if taxonomy_entry not in tax_in_files:

        logger.error(f"In the current set of entries there are no documents with tax = {taxonomy_entry}")

        return None

    selected_set = tax_in_files[taxonomy_entry]

    return {

        taxonomy_entry: docs & selected_set

        for taxonomy_entry, docs in tax_in_files.items()

        if docs & selected_set

    }



def dump_most_common_taxes(tax_in_files, rank=10):

    """Given the output of load_taxonomies(), the fuction returns the top entries (up to "rank") of the taxonomy, sorted decreasingly with respect to the number of documents they appear in."""

    tax_in_files_counter = [(taxonomy_entry, len(set_files))

                            for taxonomy_entry, set_files in tax_in_files.items()]

    return sorted(tax_in_files_counter, key=lambda x: x[1], reverse=True)[:rank]
%%time

# Input data files are available in the "../input/" directory.

# Let's load the latest version of the data.

dataset_root = Path('/kaggle/input/cord19-expertsystem-mesh/cord19_expertsystem_mesh_060320/')

tax_in_files = load_taxonomies(dataset_root)



# for instance, let's filter documents by only selecting those that show the following selected_tax, about Vaccines

selected_tax = '/MeSH Taxonomy/Chemicals and Drugs/Complex Mixtures/Biological Products/Vaccines'

selection_step = filter_data(tax_in_files, selected_tax)

pprint(dump_most_common_taxes(selection_step, 10))

print()

%%time

# Let's progressively select a subset of categories and find all the documents that contain all the following categories



requested_tax_entries = [

    '/MeSH Taxonomy/Organisms/Viruses/RNA Viruses/Nidovirales/Coronaviridae/Coronavirus',

    '/MeSH Taxonomy/Phenomena and Processes/Immune System Phenomena/Immunity', 

    '/MeSH Taxonomy/Phenomena and Processes/Microbiological Phenomena/Virulence',

    '/MeSH Taxonomy/Health Care/Environment and Public Health/Public Health/Disease Outbreaks/Epidemics/Pandemics',

    '/MeSH Taxonomy/Phenomena and Processes/Physiological Phenomena/Virus Shedding',

    '/MeSH Taxonomy/Organisms/Viruses/DNA Viruses/Herpesviridae/Alphaherpesvirinae/Simplexvirus'] 



for selected_tax in requested_tax_entries:

    selection_step = filter_data(selection_step, selected_tax)

    pprint(dump_most_common_taxes(selection_step, 10))

    print()
selected_docs = set()

for _, docs_with_tax in selection_step.items():

    selected_docs |= docs_with_tax



pprint(selected_docs)