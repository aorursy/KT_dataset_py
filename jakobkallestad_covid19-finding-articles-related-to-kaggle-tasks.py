import os

import json

import numpy as np

import pandas as pd

from copy import deepcopy

from tqdm.notebook import tqdm



!pip install langdetect

from langdetect import detect
def format_name(author):

    middle_name = " ".join(author['middle'])

    

    if author['middle']:

        return " ".join([author['first'], middle_name, author['last']])

    else:

        return " ".join([author['first'], author['last']])





def format_affiliation(affiliation):

    text = []

    location = affiliation.get('location')

    if location:

        text.extend(list(affiliation['location'].values()))

    

    institution = affiliation.get('institution')

    if institution:

        text = [institution] + text

    return ", ".join(text)



def format_authors(authors, with_affiliation=False):

    name_ls = []

    

    for author in authors:

        name = format_name(author)

        if with_affiliation:

            affiliation = format_affiliation(author['affiliation'])

            if affiliation:

                name_ls.append(f"{name} ({affiliation})")

            else:

                name_ls.append(name)

        else:

            name_ls.append(name)

    

    return ", ".join(name_ls)



def format_body(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}

    

    for section, text in texts:

        texts_di[section] += text



    body = ""



    for section, text in texts_di.items():

        body += section

        body += "\n\n"

        body += text

        body += "\n\n"

    

    return body



def format_bib(bibs):

    if type(bibs) == dict:

        bibs = list(bibs.values())

    bibs = deepcopy(bibs)

    formatted = []

    

    for bib in bibs:

        bib['authors'] = format_authors(

            bib['authors'], 

            with_affiliation=False

        )

        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]

        formatted.append(", ".join(formatted_ls))



    return "; ".join(formatted)



def load_files(dirname):

    filenames = os.listdir(dirname)

    raw_files = []



    for filename in tqdm(filenames):

        filename = dirname + filename

        file = json.load(open(filename, 'rb'))

        raw_files.append(file)

    

    return raw_files



def generate_clean_df(all_files):

    cleaned_files = []

    

    for file in tqdm(all_files):

        features = [

            file['paper_id'],

            file['metadata']['title'],

            format_authors(file['metadata']['authors']),

            format_authors(file['metadata']['authors'], 

                           with_affiliation=True),

            format_body(file['abstract']),

            format_body(file['body_text']),

            format_bib(file['bib_entries']),

            file['metadata']['authors'],

            file['bib_entries']

        ]



        cleaned_files.append(features)



    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 

                 'bibliography','raw_authors','raw_bibliography']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()

    

    return clean_df
main_dir = '/kaggle/input/CORD-19-research-challenge/'

sub_dirs = ['biorxiv_medrxiv/biorxiv_medrxiv/', 'comm_use_subset/comm_use_subset/', 'custom_license/custom_license/', 

            'noncomm_use_subset/noncomm_use_subset/']

final_dir = 'pdf_json/'



df_list = []

for sub_dir in sub_dirs:

    files = load_files(main_dir + sub_dir + final_dir)

    df = generate_clean_df(files)

    df_list.append(df)

complete_df = pd.concat(df_list)

complete_df.shape
long_texts = complete_df['text'].apply(lambda x: True if len(x.split()) > 1000 else False)

complete_df = complete_df[long_texts]

complete_df.reset_index(inplace=True, drop=True)
len(complete_df)
long_abstracts = complete_df['abstract'].apply(lambda x: True if len(x) > 500 else False)

complete_df = complete_df[long_abstracts]

complete_df.reset_index(inplace=True, drop=True)
len(complete_df)
# Common abstract template that exists across a multitude of articles. Needs to be removed.

bad_abstract = complete_df['abstract'].apply(lambda x: False if x.startswith("Abstract\n\npublicly funded repositories, such as the WHO COVID database with rights for unrestricted research re-use and analyses in any form or by any means with acknowledgement of the original source. These permissions are granted for free by Elsevier for as long as the COVID-19 resource centre remains active") else True)

complete_df = complete_df[bad_abstract]

complete_df.reset_index(inplace=True, drop=True)
len(complete_df)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from gensim.parsing import preprocess_string

import multiprocessing



cores = multiprocessing.cpu_count()

print("Using {} cores".format(cores))

tagged_docs = [TaggedDocument(preprocess_string(doc), [i]) for i, doc in enumerate(complete_df['abstract']+complete_df['text'])]

print("tagged")

model = Doc2Vec(vector_size=200, epochs=20, window=3, min_count=3, workers=cores, seed=42)

print("model init")

model.build_vocab(tagged_docs)

print("vocab")

model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

print("training finished")
model.save('doc2vec.model')
task_1 = """What is known about transmission, incubation, and environmental stability of COVID-19? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?

Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.

Prevalence of asymptomatic shedding and transmission (e.g., particularly children).

Seasonality of transmission.

Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).

Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).

Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).

Natural history of the virus and shedding of it from an infected person

Implementation of diagnostics and products to improve clinical processes

Disease models, including animal models for infection, disease and transmission

Tools and studies to monitor phenotypic change and potential adaptation of the virus

Immune response and immunity

Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings

Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings

Role of the environment in transmission"""



task_2 = """What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?

Data on potential risks factors

Smoking, pre-existing pulmonary disease

Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities

Neonates and pregnant women

Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.

Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors

Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups

Susceptibility of populations

Public health mitigation measures that could be effective for control"""



task_3 = """What do we know about virus genetics, origin, and evolution? What do we know about the virus origin and management measures at the human-animal interface?

Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.

Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.

Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.

Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.

Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.

Experimental infections to test host range for this pathogen.

Animal host(s) and any evidence of continued spill-over to humans

Socioeconomic and behavioral risk factors for this spill-over

Sustainable risk reduction strategies"""



task_4 = """What do we know about vaccines and therapeutics? What has been published concerning research and development and evaluation efforts of vaccines and therapeutics?

Effectiveness of drugs being developed and tried to treat COVID-19 patients.

Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.

Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.

Exploration of use of best animal models and their predictive value for a human vaccine.

Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.

Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.

Efforts targeted at a universal coronavirus vaccine.

Efforts to develop animal models and standardize challenge studies

Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers

Approaches to evaluate risk for enhanced disease after vaccination

Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]"""





task_5 = """What do we know about the effectiveness of non-pharmaceutical interventions? What is known about equity and barriers to compliance for non-pharmaceutical interventions?

Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.

Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.

Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.

Methods to control the spread in communities, barriers to compliance and how these vary among different populations..

Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.

Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.

Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).

Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay."""



task_6 = """What do we know about diagnostics and surveillance? What has been published concerning systematic, holistic approach to diagnostics (from the public health surveillance perspective to being able to predict clinical outcomes)?

How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).

Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.

Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.

National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).

Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.

Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).

Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.

Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.

Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.

Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.

Policies and protocols for screening and testing.

Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.

Technology roadmap for diagnostics.

Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.

New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.

Coupling genomics and diagnostic testing on a large scale.

Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.

Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.

One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.

        """



task_7 = """What has been published about medical care? What has been published concerning surge capacity and nursing homes? What has been published concerning efforts to inform allocation of scarce resources? What do we know about personal protective equipment? What has been published concerning alternative methods to advise on disease management? What has been published concerning processes of care? What do we know about the clinical characterization and management of the virus?

Resources to support skilled nursing facilities and long term care facilities.

Mobilization of surge medical staff to address shortages in overwhelmed communities

Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies

Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients

Outcomes data for COVID-19 after mechanical ventilation adjusted for age.

Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest.

Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.

Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks.

Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries.

Guidance on the simple things people can do at home to take care of sick people and manage disease.

Oral medications that might potentially work.

Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually.

Best practices and critical challenges and innovative solutions and technologies in hospital flow and organization, workforce protection, workforce allocation, community-based support resources, payment, and supply chain management to enhance capacity, efficiency, and outcomes.

Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials

Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials

Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)

"""



task_8 = """What has been published concerning ethical considerations for research? What has been published concerning social sciences at the outbreak response?

Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019

Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight

Efforts to support sustained education, access, and capacity building in the area of ethics

Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.

Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)

Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.

Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.

"""



task_9 = """What has been published about information sharing and inter-sectoral collaboration? What has been published about data standards and nomenclature? What has been published about governmental public health? What do we know about risk communication? What has been published about communicating with high-risk populations? What has been published to clarify community measures? What has been published about equity considerations and problems of inequity?

Methods for coordinating data-gathering with standardized nomenclature.

Sharing response information among planners, providers, and others.

Understanding and mitigating barriers to information-sharing.

How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).

Integration of federal/state/local public health surveillance systems.

Value of investments in baseline public health response infrastructure preparedness

Modes of communicating with target high-risk populations (elderly, health care workers).

Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too).

Communication that indicates potential risk of disease to all population groups.

Misunderstanding around containment and mitigation.

Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.

Measures to reach marginalized and disadvantaged populations.

Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities.

Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.

Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care

"""
covid_terms = [

    'covid-19', 

    'covid 19',

    'covid-2019',

    '2019 novel coronavirus', 

    'corona virus disease 2019',

    'coronavirus disease 19',

    'coronavirus 2019',

    '2019-ncov',

    'ncov-2019', 

    'wuhan virus',

    'wuhan coronavirus',

    'wuhan pneumonia',

    'ncip',

    'sars-cov-2',

    'sars-cov2']





def contains_corona(d):

    in_title = any(covid in complete_df.iloc[d]['title'].lower() for covid in covid_terms)

    return in_title





def is_english(d):

    return detect(complete_df.iloc[d]['text'].lower())=='en'





def find_relevant_articles(model, query, n_docs=3):

    infer_vector = model.infer_vector(preprocess_string(query))

    relevant_articles = model.docvecs.most_similar([infer_vector], topn = n_docs)

    return relevant_articles



def output_task(i, t):

    # Header

    query_question = "Task {} - ".format(i) + t[:t.find('?')+1]

    print("="*len(query_question))

    print('\033[95m', query_question, '\033[0m', sep='')

    print("="*len(query_question))

    print()



    # Body

    docs = find_relevant_articles(model, t, n_docs=100)

    docs = [(d, c) for d, c in docs if contains_corona(d) and is_english(d)][:3]

    for d, c in docs:

        print('\033[94m', "Article_ID={}\n".format(d), '\033[0m', '\033[93mTitle: ', complete_df.iloc[d]['title'], "\033[0m\n", complete_df.iloc[d]['abstract'], '\n', sep='')

    print()
output_task(1, task_1)
output_task(2, task_2)
output_task(3, task_3)
output_task(4, task_4)
output_task(5, task_5)
output_task(6, task_6)
output_task(7, task_7)
output_task(8, task_8)
output_task(9, task_9)