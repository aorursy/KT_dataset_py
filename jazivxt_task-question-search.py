import pandas as pd

import numpy as np

from glob import glob

import json



#Metadata

sources = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')



#https://www.kaggle.com/dgunning/browsing-the-papers-with-a-bm25-search-index

def doi_url(d): return f'http://{d}' if d.startswith('doi.org') else f'http://doi.org/{d}'

sources.doi = sources.doi.fillna('').apply(doi_url)

sources['sha'] = sources['sha'] + '.json'



#JSON Fulltext Files Path

samples = [pd.DataFrame(glob('/kaggle/input/CORD-19-research-challenge/2020-03-13/'+f+'/**/**'), columns=['path']) 

           for f in ['biorxiv_medrxiv', 'noncomm_use_subset', 'comm_use_subset', 'pmc_custom_license']]

samples = pd.concat(samples)

samples['sha'] = samples.path.map(lambda x: x.split('/')[-1])

sources = pd.merge(sources, samples, how='left', on='sha')



#Add JSON Text

def get_json(p):

    try: return json.loads(open(p).read())

    except: return ''



sources['json_text'] = sources.path.map(lambda x: get_json(x))

sources = sources[sources.has_full_text == True].reset_index(drop=True)

sources.head(1)
tasks={0:{'task': 'What is known about transmission, incubation, and environmental stability?', 

         'details': 'What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?',

         'questions': ['    Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.',

 '    Prevalence of asymptomatic shedding and transmission (e.g., particularly children).',

 '    Seasonality of transmission.',

 '    Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).',

 '    Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).',

 '    Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).',

 '    Natural history of the virus and shedding of it from an infected person',

 '    Implementation of diagnostics and products to improve clinical processes',

 '    Disease models, including animal models for infection, disease and transmission',

 '    Tools and studies to monitor phenotypic change and potential adaptation of the virus',

 '    Immune response and immunity',

 '    Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings',

 '    Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings',

 '    Role of the environment in transmission']},

1:{'task': 'What do we know about COVID-19 risk factors?', 

         'details': 'What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?',

         'questions': ['    Data on potential risks factors',

 '        Smoking, pre-existing pulmonary disease',

 '        Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities',

 '        Neonates and pregnant women',

 '        Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.',

 '    Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors',

 '    Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups',

 '    Susceptibility of populations',

 '    Public health mitigation measures that could be effective for control']},

2:{'task': 'Help us understand how geography affects virality.', 

         'details': 'At the time of writing, COVID-19 has spread to at least 114 countries. With viral flu, there are often geographic variations in how the disease will spread and if there are different variations of the virus in different areas. We???d like to explore what the literature and data say about this through this Task. ',

         'questions': ['    Are there geographic variations in the rate of COVID-19 spread?',

 '    Are there geographic variations in the mortality rate of COVID-19?',

 '    Is there any evidence to suggest geographic based virus mutations?']},

3:{'task': 'What do we know about virus genetics, origin, and evolution?', 

         'details': 'What do we know about virus genetics, origin, and evolution? What do we know about the virus origin and management measures at the human-animal interface?',

         'questions': ['    Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.',

 '    Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.',

 '    Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.',

 '        Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.',

 '        Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.',

 '        Experimental infections to test host range for this pathogen.',

 '    Animal host(s) and any evidence of continued spill-over to humans',

 '    Socioeconomic and behavioral risk factors for this spill-over',

 '    Sustainable risk reduction strategies']},

4:{'task': 'What do we know about vaccines and therapeutics?', 

         'details': 'What do we know about vaccines and therapeutics? What has been published concerning research and development and evaluation efforts of vaccines and therapeutics?',

         'questions': ['    Effectiveness of drugs being developed and tried to treat COVID-19 patients.',

 '        Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.',

 '    Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.',

 '    Exploration of use of best animal models and their predictive value for a human vaccine.',

 '    Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.',

 '    Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.',

 '    Efforts targeted at a universal coronavirus vaccine.',

 '    Efforts to develop animal models and standardize challenge studies',

 '    Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers',

 '    Approaches to evaluate risk for enhanced disease after vaccination',

 '    Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]']},

5:{'task': 'What has been published about ethical and social science considerations?', 

         'details': 'What has been published concerning ethical considerations for research? What has been published concerning social sciences at the outbreak response?',

         'questions': ['    Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019',

 '    Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight',

 '    Efforts to support sustained education, access, and capacity building in the area of ethics',

 '    Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.',

 '    Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)',

 '    Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.',

 '    Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.']},

6:{'task': 'What do we know about non-pharmaceutical interventions?', 

         'details': 'What do we know about the effectiveness of non-pharmaceutical interventions? What is known about equity and barriers to compliance for non-pharmaceutical interventions?',

         'questions': ['    Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.',

 '    Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.',

 '    Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.',

 '    Methods to control the spread in communities, barriers to compliance and how these vary among different populations..',

 '    Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.',

 '    Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.',

 '    Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).',

 '    Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay.']},

7:{'task': 'What do we know about diagnostics and surveillance?',

         'details': 'What do we know about diagnostics and surveillance? What has been published concerning systematic, holistic approach to diagnostics (from the public health surveillance perspective to being able to predict clinical outcomes)?',

         'questions': ['    How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).',

 '    Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.',

 '    Recruitment, support, and coordination of local expertise and capacity (public, private???commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.',

 '    National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).',

 '    Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.',

 '    Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).',

 '    Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.',

 '    Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.',

 '    Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.',

 '    Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.',

 '    Policies and protocols for screening and testing.',

 '    Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.',

 '    Technology roadmap for diagnostics.',

 '    Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.',

 '    New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.',

 '    Coupling genomics and diagnostic testing on a large scale.',

 '    Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.',

 '    Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.',

 '    One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.']},

8:{'task': 'What has been published about medical care?', 

         'details': 'What has been published about medical care? What has been published concerning surge capacity and nursing homes? What has been published concerning efforts to inform allocation of scarce resources? What do we know about personal protective equipment? What has been published concerning alternative methods to advise on disease management? What has been published concerning processes of care? What do we know about the clinical characterization and management of the virus?',

         'questions': ['    Resources to support skilled nursing facilities and long term care facilities.',

 '    Mobilization of surge medical staff to address shortages in overwhelmed communities',

 '    Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure ??? particularly for viral etiologies',

 '    Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients',

 '    Outcomes data for COVID-19 after mechanical ventilation adjusted for age.',

 '    Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest.',

 '    Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.',

 '    Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks.',

 '    Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries.',

 '    Guidance on the simple things people can do at home to take care of sick people and manage disease.',

 '    Oral medications that might potentially work.',

 '    Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually.',

 '    Best practices and critical challenges and innovative solutions and technologies in hospital flow and organization, workforce protection, workforce allocation, community-based support resources, payment, and supply chain management to enhance capacity, efficiency, and outcomes.',

 '    Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials',

 '    Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials',

 '    Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)']},

9:{'task': 'What has been published about information sharing and inter-sectoral collaboration?', 

         'details': 'What has been published about information sharing and inter-sectoral collaboration? What has been published about data standards and nomenclature? What has been published about governmental public health? What do we know about risk communication? What has been published about communicating with high-risk populations? What has been published to clarify community measures? What has been published about equity considerations and problems of inequity?',

         'questions': ['    Methods for coordinating data-gathering with standardized nomenclature.',

 '    Sharing response information among planners, providers, and others.',

 '    Understanding and mitigating barriers to information-sharing.',

 '    How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).',

 '    Integration of federal/state/local public health surveillance systems.',

 '    Value of investments in baseline public health response infrastructure preparedness',

 '    Modes of communicating with target high-risk populations (elderly, health care workers).',

 '    Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations??? families too).',

 '    Communication that indicates potential risk of disease to all population groups.',

 '    Misunderstanding around containment and mitigation.',

 '    Action plan to mitigate gaps and problems of inequity in the Nation???s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.',

 '    Measures to reach marginalized and disadvantaged populations.',

 '    Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities.',

 '    Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.',

 '    Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care']}

}



for task_id in tasks:

    print(tasks[task_id]['task'])
from nltk.corpus import stopwords

from tqdm.auto import tqdm

tqdm.pandas()

import re



sw = list(set(stopwords.words('english'))) + ['known', 'know', 'help', 'published', 'coronavirus']



def search(s, q):

    s = re.sub("[^a-z]"," ", s.lower())

    s = [w for w in list(set(s.split(' '))) if len(w)>2]

    r = 0

    for w in q:

        if w in s:

            r += 1

    return r

    

for i in range(10): 

    q = ' '.join([re.sub("[^a-z]"," ", q.lower()) for q in tasks[i]['questions']])

    q = [w for w in list(set(q.split(' '))) if w not in sw]

    q = [w for w in q if len(w)>2]

    sources[i] = sources.json_text.progress_map(lambda x: search(str(x), q))
from IPython.display import HTML, display



for i in range(10):

    print(tasks[i]['task'])

    results = sources.sort_values(by=[i], ascending=False)[:10]

    h = '<br/>'.join(['<a href="' + l + '" target="_blank">'+ n + '</a>' for l, n in results[['doi','title']].values])

    display(HTML(h))