import pandas as pd



def load_corpus():

    cols = ['paper_id', 'title', 'authors', 'text']

    

    biorxiv_clean = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv', na_filter=False, usecols=cols)

    clean_comm_use = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv', na_filter=False, usecols=cols)

    clean_noncomm_use =  pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv', na_filter=False, usecols=cols)

    clean_pmc =  pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv', na_filter=False, usecols=cols)



    corpus = pd.concat([biorxiv_clean, 

                        clean_comm_use, 

                        clean_noncomm_use, 

                        clean_pmc])

    

    # 'paper_id' is used as an index column

    corpus.set_index('paper_id', inplace=True)

    

    return corpus
corpus = load_corpus()

corpus.shape
corpus.head(3)
def get_paragraphs(corpus, sep="\n", min_length=100, verbose=10000):

    paragraphs, paper_ids = [], []

    

    for i, (paper_id, row) in enumerate(corpus.iterrows()):

        for s in row['text'].split(sep):

            

            if len(s) >= min_length:

                paragraphs.append(s)

                paper_ids.append(paper_id)

        

        # print progress if needed

        if verbose > 0 and (i + 1) % verbose == 0:

            print(f"Progress: {i + 1}")

            

    return paragraphs, paper_ids
paragraphs, paper_ids = list(get_paragraphs(corpus, verbose=-1))

len(paragraphs)
paragraphs[0]
from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import preprocess_string



def get_tokens(docs, preprocess=preprocess_string, verbose=10000):

    

    for i, doc in enumerate(docs):

        yield preprocess(doc)

        

        # print progress if needed

        if verbose > 0 and (i + 1) % verbose == 0:

            print(f"Progress: {i + 1}")
paragraph_tokens = list(get_tokens(paragraphs))
from gensim.summarization.bm25 import BM25



bm25 = BM25(paragraph_tokens)
import numpy as np



def get_top_n(bm25, query, n=10):

    

    # score docs

    scores = np.array(bm25.get_scores(query))

    

    # get indices of top N scores

    idx = np.argpartition(scores, -n)[-n:]

    

    # sort top N scores and return their indices

    return idx[np.argsort(-scores[idx])]
test_query = ["covid","coronavirus"]

top_idx = get_top_n(bm25, test_query)[0]

paragraphs[top_idx]
paper_ids[top_idx]
from IPython.core.display import display, HTML



def mark(s, color='black'):

    return "<text style=color:{}>{}</text>".format(color, s)



def highlight(keywords, tokens, color='red'):

    kw_set = set(keywords)

    tokens_hl = []

    

    for t in tokens:

        if t in kw_set:

            tokens_hl.append(mark(t, color=color))

        else:

            tokens_hl.append(t)

    

    return " ".join(tokens_hl)
HTML(highlight(test_query, paragraph_tokens[top_idx]))
import nltk



def get_sentences(docs, verbose=10000):



    for i, doc in enumerate(docs):

        for s in nltk.sent_tokenize(doc):

            yield s

            

        # print progress if needed

        if verbose > 0 and (i + 1) % verbose == 0:

            print(f"Progress: {i + 1}")
sentences = list(get_sentences(paragraphs))
len(sentences)
from gensim.models.fasttext import FastText



ft_model = FastText(

    sg=1, # use skip-gram: usually gives better results

    size=100, # embedding dimension (default)

    window=10, # window size: 10 tokens before and 10 tokens after to get wider context

    min_count=10, # only consider tokens with at least 10 occurrences in the corpus

    negative=15, # negative subsampling: bigger than default to sample negative examples more

    min_n=2, # min character n-gram

    max_n=5 # max character n-gram

)
ft_model.build_vocab(get_tokens(sentences, verbose=100000))
epochs = 3



for epoch in range(epochs):

    print(f"Epoch {epoch}")

    

    ft_model.train(

        get_tokens(sentences, verbose=100000),

        epochs=1,

        total_examples=ft_model.corpus_count, 

        total_words=ft_model.corpus_total_words)
ft_model.wv.most_similar("coronavirus", topn=10)
ft_model.save('cord_19_fasttext.model')
def expand_query(query, wv, topn=10):

    expanded_query = [t for t in query]

    

    for t in query:

        expanded_query.extend(s for s, f in wv.most_similar(t, topn=topn))

        

    return expanded_query
test_query
test_query_exp = expand_query(test_query, ft_model.wv)

top_idx = get_top_n(bm25, test_query_exp)[0]
HTML(highlight(test_query_exp, paragraph_tokens[top_idx]))
t1 = {

    "txt": "What is known about transmission, incubation, and environmental stability?",

    "qs": [

             "Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.",

             "Prevalence of asymptomatic shedding and transmission (e.g., particularly children).",

             "Seasonality of transmission.",

             "Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).",

             "Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).",

             "Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).",

             "Natural history of the virus and shedding of it from an infected person",

             "Implementation of diagnostics and products to improve clinical processes",

             "Disease models, including animal models for infection, disease and transmission",

             "Tools and studies to monitor phenotypic change and potential adaptation of the virus",

             "Immune response and immunity",

             "Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",

             "Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings",

             "Role of the environment in transmission"

    ]

}
t2 = {

    "txt": "What do we know about COVID-19 risk factors?",

    "qs": [

        "Smoking, pre-existing pulmonary disease",

        "Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities",

        "Neonates and pregnant women",

        "Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.",

        "Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors",

        "Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups",

        "Susceptibility of populations",

        "Public health mitigation measures that could be effective for control"

    ]

}
t3 = {

    "txt": "What do we know about virus genetics, origin, and evolution?",

    "qs": [

        "Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.",

        "Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.",

        "Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.",

        "Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.",

        "Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.",

        "Experimental infections to test host range for this pathogen.",

        "Animal host(s) and any evidence of continued spill-over to humans",

        "Socioeconomic and behavioral risk factors for this spill-over",

        "Sustainable risk reduction strategies"

    ]

}
t4 = {

    "txt": "What do we know about vaccines and therapeutics?",

    "qs": [

        "Effectiveness of drugs being developed and tried to treat COVID-19 patients.",

        "Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.",

        "Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.",

        "Exploration of use of best animal models and their predictive value for a human vaccine.",

        "Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.",

        "Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.",

        "Efforts targeted at a universal coronavirus vaccine.",

        "Efforts to develop animal models and standardize challenge studies",

        "Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers",

        "Approaches to evaluate risk for enhanced disease after vaccination",

        "Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]"

    ]

}
t5 = {

    "txt": "What has been published about medical care?",

    "qs": [

        "Resources to support skilled nursing facilities and long term care facilities.",

        "Mobilization of surge medical staff to address shortages in overwhelmed communities",

        "Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure ??? particularly for viral etiologies",

        "Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients",

        "Outcomes data for COVID-19 after mechanical ventilation adjusted for age.",

        "Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest.",

        "Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.",

        "Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks.",

        "Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries.",

        "Guidance on the simple things people can do at home to take care of sick people and manage disease.",

        "Oral medications that might potentially work.",

        "Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually.",

        "Best practices and critical challenges and innovative solutions and technologies in hospital flow and organization, workforce protection, workforce allocation, community-based support resources, payment, and supply chain management to enhance capacity, efficiency, and outcomes.",

        "Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials",

        "Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials",

        "Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)"

    ]

}
t6 = {

    "txt": "What do we know about non-pharmaceutical interventions?",

    "qs": [

        "Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.",

        "Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.",

        "Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.",

        "Methods to control the spread in communities, barriers to compliance and how these vary among different populations..",

        "Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.",

        "Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.",

        "Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).",

        "Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay."

    ]

}
t7 = {

    "txt": "What do we know about diagnostics and surveillance?",

    "qs": [

        "How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).",

        "Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.",

        "Recruitment, support, and coordination of local expertise and capacity (public, private???commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.",

        "National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).",

        "Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.",

        "Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).",

        "Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.",

        "Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.",

        "Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.",

        "Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.",

        "Policies and protocols for screening and testing.",

        "Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.",

        "Technology roadmap for diagnostics.",

        "Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.",

        "New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.",

        "Coupling genomics and diagnostic testing on a large scale.",

        "Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.",

        "Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.",

        "One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors."

    ]

}
t8 = {

    "txt": "What has been published about ethical and social science considerations?",

    "qs": [

        "Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019",

        "Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight",

        "Efforts to support sustained education, access, and capacity building in the area of ethics",

        "Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.",

        "Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)",

        "Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.",

        "Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media."

    ]

}
t9 = {

    "txt": "What has been published about information sharing and inter-sectoral collaboration?",

    "qs": [

        "Methods for coordinating data-gathering with standardized nomenclature.",

        "Sharing response information among planners, providers, and others.",

        "Understanding and mitigating barriers to information-sharing.",

        "How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).",

        "Integration of federal/state/local public health surveillance systems.",

        "Value of investments in baseline public health response infrastructure preparedness",

        "Modes of communicating with target high-risk populations (elderly, health care workers).",

        "Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations??? families too).",

        "Communication that indicates potential risk of disease to all population groups.",

        "Misunderstanding around containment and mitigation.",

        "Action plan to mitigate gaps and problems of inequity in the Nation???s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.",

        "Measures to reach marginalized and disadvantaged populations.",

        "Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities.",

        "Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.",

        "Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care"

    ]

}
tasks = {"t1": t1, 

         "t2": t2, 

         "t3": t3, 

         "t4": t4,

         "t5": t5,

         "t6": t6,

         "t7": t7,

         "t8": t8,

         "t9": t9}
import json



with open('/kaggle/working/cord_19_tasks.json', 'w') as f:

    json.dump(tasks, f)
def answer_questions(n_hits=10, print_hits=0, preprocess=preprocess_string):

    

    # make results dictionary

    results = {}

    

    # answer questions

    for tid, task in tasks.items():

        results[tid] = {"txt": task["txt"]}

        title_query = preprocess(task["txt"])

        

        for qid, question in enumerate(task["qs"]):

            results[tid][f"q{qid + 1}"] = {"txt": question}

            question_query = preprocess(question)

            

            # make a query

            query = ["coronavirus"]

            query.extend(title_query)

            query.extend(question_query)

            expanded_query = expand_query(query, ft_model.wv)

            

            # get top hits

            top_idx = get_top_n(bm25, expanded_query, n=n_hits)

            

            # fill in results

            for hid, idx in enumerate(top_idx):

                results[tid][f"q{qid + 1}"][f"h{hid + 1}"] = {

                    "txt": paragraphs[idx], "pid": paper_ids[idx]}

            

            # print if needed

            for hid in range(min(print_hits, n_hits)):

                display(HTML("<br/>".join([

                    mark(f'{tid.upper()}: {task["txt"]}', color="blue"),

                    mark(f'Q{qid + 1}: {question}', color="green"),

                    mark(f'H{hid + 1}:', color="red"),

                    highlight(expanded_query, paragraph_tokens[top_idx[hid]], color="red")

                ])))

            

    return results
results = answer_questions(n_hits=10, print_hits=1)
with open('/kaggle/working/cord_19_answers.json', 'w') as f:

    json.dump(results, f)
t1q1h1_txt = results["t1"]["q1"]["h1"]["txt"]

print(t1q1h1_txt)
t1q1h1_pid = results["t1"]["q1"]["h1"]["pid"]
corpus.loc[[t1q1h1_pid]]