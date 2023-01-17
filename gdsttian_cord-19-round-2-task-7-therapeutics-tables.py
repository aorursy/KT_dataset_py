import pandas as pd



table_hypercoagulability = pd.read_csv(f"/kaggle/input/cord19round2task7/summary_table_hypercoagulability.csv", na_filter= False)

print(f"Table 1: What is the best method to combat the hypercoagulable state seen in COVID-19_.csv")

print(f"{'Number of entries in the table:':20}{table_hypercoagulability.shape[0]:5}")

table_hypercoagulability.head()
table_therapeuticefficacy = pd.read_csv(f"/kaggle/input/cord19round2task7/summary_table_therapeutic_efficacy.csv", na_filter= False)

print(f"Table 2: What is the efficacy of novel therapeutics being tested currently_.csv")

print(f"{'Number of entries in the table:':20}{table_therapeuticefficacy.shape[0]:5}")

table_therapeuticefficacy.head()
# !python preprocess_get_ids.py

# !python preprocess_cord_data.py
# !pip install git+https://bitbucket.org/nmonath/befree.git

# !python entities_get_pubtator_annotation.py

# !python entities_post_tiabs_to_pubtator.py

# !python entities_retrieve_tiabs_from_pubtator.py

# !python entities_process_pubtator_annotation.py

# !python entities_additional_annotation.py
# !python data_aggregation.py

# !python data_nodes_relations.py

# !python data_indexing_time.py

# !python data_indexing_word.py
# data pathes

data_path = '/kaggle/input/cord-19-data-with-tagged-named-entities/data' # folder for system data

json_path = '/kaggle/input/cord-19-data-with-tagged-named-entities/data/json_files/json_files' # path of final json files

mapping_pnid = 'mapping_corduid2nid.json' # dictionary mapping cord_uid to numeric id for each paper



index_year = 'index_time_year.json' # dictionary of list of papers for each publish year

index_title = 'index_word_title.json' # dictionary of list of papers for each word in title

index_abstract = 'index_word_abstract.json' # dictionary of list of papers for each word in abstract

word_counts = 'paper_word_counts.json' # word counts by paper

index_table = 'index_word_table.json'

paper_tables = 'paper_tables.json'



entity_lists = 'entity_lists.json' # entity checking lists including disease list, blacklist etc.

entity_nodes = 'entity_nodes.json' # entities dictionary

entity_relations = 'entity_relations.json' # entity relation dictionary



mapping_sents = 'mapping_sents2nid.json' # mapping sent id to numeric id

index_sents = 'index_word_sents.json' # mapping word to a list of numeric sent id

sentences = 'sentences.json' # dictionary of all sentences with unique id

# packages

from utils import *

from mining_search_tool import *

import os

csv_path = 'csv'

os.makedirs(csv_path, exist_ok=True)
papers = SearchPapers(data_path, json_path, mapping_pnid, index_year,

                      index_title, index_abstract, word_counts, index_table, paper_tables,

                      entity_lists, entity_nodes, entity_relations, index_sents, mapping_sents, sentences)
covid19_names = """covid-19, covid19, covid, sars-cov-2, sars-cov2, sarscov2,

                   novel coronavirus, 2019-ncov, 2019ncov, wuhan coronavirus

                """
papers_covid19 = papers.search_papers(covid19_names, section = None, publish_year = '2020')

print(f"{'Total papers relevant to COVID-19:':20}{len(papers_covid19):6}")
papers.display_papers(papers_covid19[:1])
query_coagulation = """coagulation, anticoagulant, decoagulant, anticoagulation,

                       hypercoagulable, hypercoagulability, coagulopathy, vasoconstrictive,

                       thromboprophylaxis, thrombosis, thrombotic, thromboembolism, thromboprophylaxis

                    """
query_therapy = """therapy, therapeutic, therapeutics,inhibitor,

                   inhibitors, medicine, medication, drug, drugs,

                   treat, treatment, pharmaceutical, pharmaceutic

                """
query_effect = """effect, effects, efficacy, effective, effectiveness, benifit, benifits

               """
sys_review = ['systematic review', 'meta-analysis',

              'search: PubMed, PMC, Medline, Embase, Google Scholar, UpToDate, Web of Science',

              'searched: PubMed, PMC, Medline, Embase, Google Scholar, UpToDate, Web of Science',

              'in: PubMed, PMC, Medline, Embase, Google Scholar, UpToDate, Web of Science']

retro_study = ['record review','retrospective', 'observational cohort', 'scoping review']

simulation = ['modelling','model','molecular docking','modeling','immunoinformatics', 'simulation', 'in silico', 'in vitro']
def get_papers_by_study_type(study_type_list):

    papers_for_the_type = set()

    for phrase in study_type_list:

        papers_by_phrase = papers.search_papers(phrase, section = 'abs', publish_year = '2020')

        papers_for_the_type = papers_for_the_type.union(set(papers_by_phrase))

    return papers_for_the_type



papers_review = get_papers_by_study_type(sys_review).intersection(set(papers_covid19))

print(f"{'Systematic Review: ':25}{len(papers_review):6}")



papers_retro = get_papers_by_study_type(retro_study).intersection(set(papers_covid19))

papers_retro = papers_retro - (papers_retro & papers_review)

print(f"{'Retrospective Study: ':25}{len(papers_retro):6}")



papers_simulation = get_papers_by_study_type(simulation).intersection(set(papers_covid19))

papers_simulation = papers_simulation - (papers_simulation & (papers_retro | papers_review))

print(f"{'Simulation: ':25}{len(papers_simulation):6}")



papers_others = set(papers_covid19) - (papers_review | papers_retro | papers_simulation)

print(f"{'Other Studies: ':25}{len(papers_others):6}")
drug_blocklist = ['/fio2', '04-MAY-2020', '2-o', '25-hydroxycholecalcifoerol', '25ohd', '3350',

                  '3mtm', "5'-tp", '6-o', '80-82 oc', 'acid', 'adverse drug', 'alcohol',

                  'alcohols', 'alkaline', 'amino acid-related', 'amp', 'amplify', 'androgen',

                  'antagonist', 'asp355asn', 'atp', 'bi', 'biopharmaceutical', 'bipap',

                  'bis(monoacylglycero)phosphate', 'bmp', 'cai', 'calcium channel', 'carbon dioxide',

                  'cationic', 'chemical', 'chemical compounds', 'chemical space', 'chemicals', 'co',

                  'co2', 'compound', 'compounds', 'copper', 'cov-2 poc', 'covalent', 'covalent fragments',

                  'covid-19', 'creatinine', 'cs', 'ctpa', 'cu', 'cys141', 'd-d', 'd-dimer', 'daegu', 'dfu',

                  'dic', 'dmec', 'drug molecules', 'drug products', 'effector', 'electron', 'electrophile',

                  'electrophilic fragment', 'eosin', 'ethylene oxide', 'eto', 'exoflo', 'extracellularly',

                  'fdp', 'fio2', 'food', 'food vacuole', 'foods', 'fumigants', 'gdp', 'glu166',

                  'glucose', 'glycan', 'h2o2', 'h7gmu', 'heme', 'hemoglobin', 'hemoglobin molecule',

                  'hepatocellular type', 'hfc%', 'hfno', 'hg', 'his', 'hormone', 'hormones',

                  'hpgp', 'hs', 'hydrogen', 'hydrogen peroxide', 'immunomodulators', 'immunomodulatory',

                  'in10', 'ingredients', 'inhibitor', 'inhibitors', 'iron', 'iso13485', 'ketone',

                  'l506a-f', 'lipid bis(monoacylglycero) phosphate', 'lipid', 'low-dose', 'lu',

                  'lysosomotropic drugs', 'magnesium', 'magnesium sulfate', 'mesh', 'metabolites', 'metal',

                  'metal ions', 'metals', 'meteorological', 'mineral', 'molecular', 'molecular electrostatic',

                  'molecular probes', 'molecule', 'molecules', 'mrna', 'mt039887', 'mt184913', 'mt259229',

                  'n', 'n-95', 'n95', 'nacl', 'ncpp', 'nct04330638', 'nct04355364', 'nct04359654',

                  'nitric oxide', 'nitrogen', 'niv', 'nmdd', 'no2', 'non-toxic', 'nps', 'nucleic acid',

                  'nucleoside', 'nucleotide', 'nutrients', 'o2', 'organs', 'outbroke', 'oxygen',

                  'oxygen heterocyclic compounds', 'oxygen partial', 'oxygen species', 'ozone', 'pao(2)/fio(2',

                  'pao2', 'peptidomimetics', 'pergamum', 'pesticides', 'pharmaceuticals', 'pharmacies',

                  'pharmacist', 'pharmacologically active substances', 'phd', 'phospholipid', 'phosphorus',

                  'phytochemicals', 'pic', 'pico', 'pigmented', 'plant', 'pm', 'pollutants', 'ppe',

                  'prodrug', 'progesterone', 'protein', 'proteins', 'quarantine', 'r +', 'radical', 'reagent',

                  'renine', 'residue', 'residues', 'ribose', 's2', 'sanitizer', 'sao2', 'se', 'ser', 'silica',

                  'silver', 'small-molecule inhibitors', 'sodium', 'spo(2)', 'spo2', 'srq-20', 'steroid',

                  'steroids', 'substance', 'substances', 'supplements', 'therapeutic', 'therapeutic agents',

                  'therapeutic anticoagulation', 'therapeutic drugs', 'thr27arg', 'topical', 'toxic',

                  'trizol', 'ultraviolet', 'urea', 'urea nitrogen', 'vdi', 'vph', 'water', 'xenobiotics']
print(len(drug_blocklist))
ss_patient = re.compile(r'(\s)([0-9,]+)(\s|\s[^0-9,\.\s]+\s|\s[^0-9,\.\s]+\s[^0-9,\.\s]+\s)(patients|persons|cases|subjects|records)')

ss_review = re.compile(r'(\s)([0-9,]+)(\s|\s[^0-9,\.\s]+\s|\s[^0-9,\.\s]+\s[^0-9,\.\s]+\s)(studies|papers|articles|publications|reports|records)')
severity_words = ['mild', 'moderate', 'severe', 'critical', 'icu', 'non-icu',

                  'fatality','mortality','mortalities','death','deaths','dead','casualty']
primary_phrase = ['primary outcome', 'primary endpoint']

improve_phrase = ['improve', 'better', 'amend', 'ameliorate', 'meliorate']
papers_coagulation = papers.search_papers(query_coagulation, section = None, publish_year = '2020')

papers_coagulation = list(set(papers_coagulation) & set(papers_covid19))

print(f"{'Papers containing any hypercoagulability keywords: ':60}{len(papers_coagulation):6}")



selectpapers_coagulation = []

for paper in papers_coagulation:

    if 'Chemical' in papers.entity_nodes[str(paper)]:

        chemicals = [chem for chem in papers.entity_nodes[str(paper)]['Chemical'].keys() if chem.lower() not in drug_blocklist]

        if len(chemicals) > 0:

            selectpapers_coagulation.append(paper)



print(f"{'Selected papers addressing hypercoagulability:':60}{len(selectpapers_coagulation):6}")
papers.display_papers(selectpapers_coagulation[:1])
entity_stats = papers.get_entity_stats(selectpapers_coagulation)

therapeutics = {}

for k, v in entity_stats['Chemical'].items():

    if k.lower() not in drug_blocklist:

        if k.lower() not in therapeutics:

            therapeutics[k.lower()] = v

        else:

            therapeutics[k.lower()] += v

print('-'*45)

print(f"| {'Chemicals':32} | {'Counts':6} |")

print('-'*45)

for i in sorted(therapeutics.items(), key = lambda x:x[1], reverse = True)[:15]:

      print(f"| {i[0]:32} | {i[1]:6} |")

print('-'*45)
import csv

from datetime import date

from spacy.lang.en import English

nlp = English()

sentencizer = nlp.create_pipe("sentencizer")

nlp.add_pipe(sentencizer)

file_name = 'summary_table_hypercoagulability'

with open(f"{csv_path}/{file_name}.csv", 'w', encoding = 'utf-8') as fcsv:

    csv_writer = csv.writer(fcsv)

    csv_writer.writerow(['Date', 'Study', 'Study Link', 'Journal', 'Study Type',

                         'Therapeutics', 'Sample Size', 'Severity', 'General Outcome',

                         'Primary Endpoints', 'Clinical Improvement', 'Added On',

                         'DOI', 'CORD_UID'])



    for pid in selectpapers_coagulation:

        file = json.load(open(f'{json_path}/{papers.nid2corduid[int(pid)]}.json', 'r', encoding = 'utf-8'))

        abstract = file['abstract']['text']

        if abstract == '': continue

        doc = nlp(abstract)

        sents_abs = list(doc.sents)

        if len(sents_abs) == 1:

            if 'copyright' in sents_abs[0].text:

                continue

        elif len(sents_abs) == 2:

            if 'copyright' in sents_abs[0].text:

                continue

            elif 'copyright' in sents_abs[1].text:

                sents_abs = sents_abs[0]

        else:

            if 'copyright' in sents_abs[-2].text:

                sents_abs = sents_abs[:-2]

            elif 'copyright' in sents_abs[-1].text:

                sents_abs = sents_abs[:-1]

        if len(sents_abs) == 0: continue

        pub_date = file['publish_time']

        study = file['title']['text']

        study_link = file['url']

        journal = file['journal']

        #study type

        if int(pid) in papers_review:

            study_type = 'Systematic Review'

        elif int(pid) in papers_retro:

            study_type = 'Retrospective Study'

        elif int(pid) in papers_simulation:

            study_type = 'Simulation'

        else:

            study_type = 'Other'

        # therapeutics

        chemicals = list(set(chem.lower() for chem in papers.entity_nodes[str(pid)]['Chemical'].keys() if chem.lower() not in drug_blocklist))

        therapeutics = ', '.join(chemicals)

        # sample size

        sample_size = ''

        if study_type == 'Systematic Review':

            matches = re.findall(ss_review, abstract)

            for match in matches:

                if match[1].isdigit() and int(match[1]) != 2019:

                    sample_size = sample_size + ''.join(match[1:]) + '; '

        elif study_type == 'Retrospective Study' or study_type == 'Other' :

            matches = re.findall(ss_patient, abstract)

            for match in matches:

                if match[1].isdigit() and int(match[1]) != 2019:

                    sample_size = sample_size + ''.join(match[1:]) + '; '

        # severity

        severity = []

        for phrase in severity_words:

            if phrase in abstract.lower():

                severity.append(phrase)

        severity = ', '.join(severity)

        # general outcome

        conclusion = ''

        conclusion_match = re.search(r'(?<=\s)(Conclusion[^,]?:\s?)(.*)', abstract, flags = re.I)

        if conclusion_match != None:

            conclusion = conclusion_match[2].strip()

        if conclusion != '':

            gen_outcome = conclusion

        else:

            if len(sents_abs) <= 2:

                gen_outcome = ' '.join(sent.text for sent in sents_abs)

            else:

                sents = []

                num = len(sents_abs)

                for sent_i, sent in enumerate(sents_abs):

                    if any(chem.lower() in sent.text.lower() for chem in chemicals) and sent_i < num-2:

                        sents.append(sent.text)

                if len(sents) > 0:

                    gen_outcome = sents[-1] + ' ' + sents_abs[-2].text + ' ' + sents_abs[-1].text

                else:

                    gen_outcome = sents_abs[-2].text + ' ' + sents_abs[-1].text

        # primary endpoint

        primary_endponit = ''

        for sent in doc.sents:

            if any(phrase.lower() in sent.text.lower() for phrase in primary_phrase):

                primary_endponit = primary_endponit + sent.text + ' '

        # clinical improvement

        clinical_improvement = ''

        if any(phrase.lower() in sent.text.lower() for phrase in improve_phrase):

            clinical_improvement = 'Y'

        # added on

        added_on = date.today().strftime('%m/%d/%Y')

        doi = file['doi']

        cord_uid = file['cord_uid']

            

        csv_writer.writerow([pub_date, study, study_link, journal, study_type, therapeutics, sample_size,

                             severity, gen_outcome, primary_endponit, clinical_improvement, added_on, doi, cord_uid])
import pandas as pd

table_hypercoagulability = pd.read_csv(f"{csv_path}/{file_name}.csv", na_filter= False)

print(f"{'Total papers:':20}{table_hypercoagulability.shape[0]:5}")

table_hypercoagulability.head()
papers_therapy = papers.search_papers(query_therapy, section = None, publish_year = '2020')

papers_therapy = list(set(papers_therapy) & set(papers_covid19))

print(f"{'Papers containing any therapeutic keywords:':60}{len(papers_therapy):6}")



papers_effect = papers.search_papers(query_effect, section = None, publish_year = '2020')

papers_effect = list(set(papers_effect) & set(papers_covid19))

print(f"{'Papers containing any efficacy keywords:':60}{len(papers_effect):6}")



papers_therapyeffects = list(set(papers_therapy) & set(papers_effect))

print(f"{'Papers containing both therapeutic and efficacy keywords:':60}{len(papers_therapyeffects):6}")



selectpapers_therapyeffects = []

for paper in papers_therapyeffects:

    if 'Chemical' in papers.entity_nodes[str(paper)]:

        chemicals = [chem for chem in papers.entity_nodes[str(paper)]['Chemical'].keys() if chem.lower() not in drug_blocklist]

        if len(chemicals) > 0:

            selectpapers_therapyeffects.append(paper)



print(f"{'Selected papers addressing therapeutic efficacy:':60}{len(selectpapers_therapyeffects):6}")
papers.display_papers(selectpapers_therapyeffects[:1])
entity_stats = papers.get_entity_stats(selectpapers_therapyeffects)

therapeutics = {}

for k, v in entity_stats['Chemical'].items():

    if k.lower() not in drug_blocklist:

        if k.lower() not in therapeutics:

            therapeutics[k.lower()] = v

        else:

            therapeutics[k.lower()] += v

print('-'*45)

print(f"| {'Chemicals':32} | {'Counts':6} |")

print('-'*45)

for i in sorted(therapeutics.items(), key = lambda x:x[1], reverse = True)[:15]:

      print(f"| {i[0]:32} | {i[1]:6} |")

print('-'*45)
import csv

from datetime import date

from spacy.lang.en import English

nlp = English()

sentencizer = nlp.create_pipe("sentencizer")

nlp.add_pipe(sentencizer)

file_name = 'summary_table_therapeutic_efficacy'

with open(f"{csv_path}/{file_name}.csv", 'w', encoding = 'utf-8') as fcsv:

    csv_writer = csv.writer(fcsv)

    csv_writer.writerow(['Date', 'Study', 'Study Link', 'Journal', 'Study Type',

                         'Therapeutics', 'Sample Size', 'Severity', 'General Outcome',

                         'Primary Endpoints', 'Clinical Improvement', 'Added On',

                         'DOI', 'CORD_UID'])



    for pid in selectpapers_therapyeffects:

        file = json.load(open(f'{json_path}/{papers.nid2corduid[int(pid)]}.json', 'r', encoding = 'utf-8'))

        abstract = file['abstract']['text']

        if abstract == '': continue

        doc = nlp(abstract)

        sents_abs = list(doc.sents)

        if len(sents_abs) == 1:

            if 'copyright' in sents_abs[0].text:

                continue

        elif len(sents_abs) == 2:

            if 'copyright' in sents_abs[0].text:

                continue

            elif 'copyright' in sents_abs[1].text:

                sents_abs = sents_abs[0]

        else:

            if 'copyright' in sents_abs[-2].text:

                sents_abs = sents_abs[:-2]

            elif 'copyright' in sents_abs[-1].text:

                sents_abs = sents_abs[:-1]

        if len(sents_abs) == 0: continue

        pub_date = file['publish_time']

        study = file['title']['text']

        study_link = file['url']

        journal = file['journal']

        #study type

        if int(pid) in papers_review:

            study_type = 'Systematic Review'

        elif int(pid) in papers_retro:

            study_type = 'Retrospective Study'

        elif int(pid) in papers_simulation:

            study_type = 'Simulation'

        else:

            study_type = 'Other'

        # therapeutics

        chemicals = list(set(chem.lower() for chem in papers.entity_nodes[str(pid)]['Chemical'].keys() if chem.lower() not in drug_blocklist))

        therapeutics = ', '.join(chemicals)

        # find relevant information from abstract

        abstract = file['abstract']['text']

        doc = nlp(abstract)

        # sample size

        sample_size = ''

        if study_type == 'Systematic Review':

            matches = re.findall(ss_review, abstract)

            for match in matches:

                if match[1].isdigit() and int(match[1]) != 2019:

                    sample_size = sample_size + ''.join(match[1:]) + '; '

        elif study_type == 'Retrospective Study' or study_type == 'Other' :

            matches = re.findall(ss_patient, abstract)

            for match in matches:

                if match[1].isdigit() and int(match[1]) != 2019:

                    sample_size = sample_size + ''.join(match[1:]) + '; '

        # severity

        severity = []

        for phrase in severity_words:

            if phrase in abstract.lower():

                severity.append(phrase)

        severity = ', '.join(severity)

        # general outcome

        conclusion = ''

        conclusion_match = re.search(r'(?<=\s)(Conclusion[^,]?:\s?)(.*)', abstract, flags = re.I)

        if conclusion_match != None:

            conclusion = conclusion_match[2].strip()

        if conclusion != '':

            gen_outcome = conclusion

        else:

            if len(sents_abs) <= 2:

                gen_outcome = ' '.join(sent.text for sent in sents_abs)

            else:

                sents = []

                num = len(sents_abs)

                for sent_i, sent in enumerate(sents_abs):

                    if any(chem.lower() in sent.text.lower() for chem in chemicals) and sent_i < num-2:

                        sents.append(sent.text)

                if len(sents) > 0:

                    gen_outcome = sents[-1] + ' ' + sents_abs[-2].text + ' ' + sents_abs[-1].text

                else:

                    gen_outcome = sents_abs[-2].text + ' ' + sents_abs[-1].text

        # primary endpoint

        primary_endponit = ''

        for sent in doc.sents:

            if any(phrase.lower() in sent.text.lower() for phrase in primary_phrase):

                primary_endponit = primary_endponit + sent.text + ' '

        # clinical improvement

        clinical_improvement = ''

        if any(phrase.lower() in sent.text.lower() for phrase in improve_phrase):

            clinical_improvement = 'Y'

        # added on

        added_on = date.today().strftime('%m/%d/%Y')

        doi = file['doi']

        cord_uid = file['cord_uid']

            

        csv_writer.writerow([pub_date, study, study_link, journal, study_type, therapeutics, sample_size,

                             severity, gen_outcome, primary_endponit, clinical_improvement, added_on, doi, cord_uid])
import pandas as pd

table_therapeuticefficacy = pd.read_csv(f"{csv_path}/{file_name}.csv", na_filter= False)

print(f"{'Total papers:':20}{table_therapeuticefficacy.shape[0]:5}")

table_therapeuticefficacy.head()