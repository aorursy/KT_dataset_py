%matplotlib inline

import pandas as pd

import numpy as np

import json

import math

import glob



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from nltk.stem.snowball import SnowballStemmer

from scipy.spatial import distance

from matplotlib import pyplot as plt

from wordcloud import WordCloud

import ipywidgets as widgets



from IPython.display import Image

from IPython.display import display, HTML
root_dir = '/kaggle/input/CORD-19-research-challenge'

#root_dir = 'CORD-19-research-challenge/2020-03-13'
df = pd.read_csv(f'{root_dir}/metadata.csv')

doc_paths = glob.glob(f'{root_dir}/*/*/*.json')

df.sha.fillna("", inplace=True)



#get text for articles that are available

def get_text(sha):

    if sha == "":

        return ""

    document_path = [x for x in doc_paths if sha in x]

    if not document_path:

        return ""

    with open(document_path[0]) as f:

        file = json.load(f)

        full_text = []

        #iterate over abstract and body part

        for part in ['abstract', 'body_text']:

            # iterate over each paragraph

            for text_part in file[part]:

                text = text_part['text']

                # remove citations from each paragraph

                for citation in text_part['cite_spans']:

                    text = text.replace(citation['text'], "")

                full_text.append(text)

            

        return str.join(' ', full_text)

%time df['text'] = df.apply(lambda x: get_text(x.sha), axis=1)
analyzer = CountVectorizer().build_analyzer()

stemmer = SnowballStemmer("english")



def preprocess(doc):

    doc=doc.lower()

    return str.join(" ", [stemmer.stem(w) for w in analyzer(doc)])



def preprocess_row(row):

    text = str.join(' ', [str(row.title), str(row.abstract), str(row.text)])

    return preprocess(text)



%time df['preprocessed'] = df.apply(lambda x: preprocess_row(x), axis=1)
cv = CountVectorizer(max_df=0.95, stop_words='english')

%time word_count = cv.fit_transform(df.preprocessed)

tfidf_tr = TfidfTransformer(smooth_idf=True, use_idf=True)

%time tfidf_tr.fit(word_count)
def get_word_vector(document):

    w_vector = tfidf_tr.transform(cv.transform([document]))

    return w_vector 



%time df['word_vector'] = df.preprocessed.apply(get_word_vector)
def show_word_cloud(word_vector):

    cloud = WordCloud(background_color='white',

        width=500,

        height=500,

        max_words=20,

        colormap='tab10',

        prefer_horizontal=1.0)

    word_frequency = dict(get_words_with_value(word_vector))

    cloud.generate_from_frequencies(word_frequency)

    plt.gca().imshow(cloud)

    plt.gca().axis('off')



feature_names = cv.get_feature_names()

def get_words_with_value(w_vector):

    return sorted([(feature_names[ind], val) for ind, val in zip(w_vector.indices, w_vector.data)], key=lambda x: x[1], reverse=True)
def calculate_distance_between_words_vectors(search_words_indices, search_vec, document_vector):

    document_vec = document_vector[0, search_words_indices].toarray()

    return distance.euclidean(search_vec, document_vec)



def get_related_documents(text, number_of_documents):    

    search_vector = get_word_vector(preprocess(text))

    search_words_indices = search_vector.indices

    search_vec = search_vector[0, search_words_indices].toarray()

    distance_idx = df.apply(lambda x: calculate_distance_between_words_vectors(search_words_indices, search_vec, x.word_vector), axis=1)

    relevant_indexes = distance_idx.sort_values().head(number_of_documents).index 

    result_columns = ["title", "doi", "pmcid", "pubmed_id", "license", "authors", "word_vector"]

    result = df[result_columns].iloc[relevant_indexes].fillna("")

    return result
tasks = {

    "What is known about transmission, incubation, and environmental stability?" :

    [

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

    ],

    "What do we know about COVID-19 risk factors?":

    [

        "Data on potential risks factors",

        "Smoking, pre-existing pulmonary disease",

        "Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities",

        "Neonates and pregnant women",

        "Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.",

        "Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors",

        "Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups",

        "Susceptibility of populations",

        "Public health mitigation measures that could be effective for control"

    ],

    "What do we know about virus genetics, origin, and evolution?":

    [

        "Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.",

        "Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.",

        "Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.",

        "Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.",

        "Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.",

        "Experimental infections to test host range for this pathogen.",

        "Animal host(s) and any evidence of continued spill-over to humans",

        "Socioeconomic and behavioral risk factors for this spill-over",

        "Sustainable risk reduction strategies"

    ],

    "What do we know about vaccines and therapeutics?":

    [

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

        "Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics],",

    ],

    "What do we know about diagnostics and surveillance?":

    [

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

    ],

    "What do we know about non-pharmaceutical interventions?":

    [

        "Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.",

        "Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.",

        "Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.",

        "Methods to control the spread in communities, barriers to compliance and how these vary among different populations.",

        "Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.",

        "Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.",

        "Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).",

        "Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay."

    ],

    "What has been published about medical care?":

    [

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

    ],

    "What has been published about information sharing and inter-sectoral collaboration?":

    [

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

    ],

    "What has been published about ethical and social science considerations?":

    [

        "Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019",

        "Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight",

        "Efforts to support sustained education, access, and capacity building in the area of ethics",

        "Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.",

        "Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)",

        "Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.",

        "Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.",

    ]

}
def display_friendly_results(df_result):

    display_columns = ["title", "doi", "pmcid", "pubmed_id", "authors"]

    display(df_result[display_columns].reset_index(drop=True))

    rows = math.ceil(len(df_result)/3)

    plt.rcParams["figure.figsize"] = (20,15)

    for i in range(len(df_result)):

        row = df_result.iloc[1]

        plt.subplot(rows, 3, i+1)

        show_word_cloud(row.word_vector)

        plt.title(f'Paper {i}', fontsize=20)

    plt.show()



def display_topics_results(task):

    for topic in tasks[task]:

        display(HTML(f"<h3>{topic}<h3>"))

        topic_result = get_related_documents(topic, 6)

        display_friendly_results(topic_result)
display_topics_results("What is known about transmission, incubation, and environmental stability?")
display_topics_results("What do we know about COVID-19 risk factors?")
display_topics_results("What do we know about virus genetics, origin, and evolution?")
display_topics_results("What do we know about vaccines and therapeutics?")
display_topics_results("What do we know about diagnostics and surveillance?")
display_topics_results("What do we know about non-pharmaceutical interventions?")
display_topics_results("What has been published about medical care?")
display_topics_results("What has been published about information sharing and inter-sectoral collaboration?")
display_topics_results("What has been published about ethical and social science considerations?")
def search():

    def document_change(event):

        if not (event['type'] == 'change' and event['name'] == 'value'):

            return

        render_tab(result_tab, results.iloc[doc_select.index])



    def render_tab(tab, r):

        paper_info = widgets.VBox([widgets.Label(f"{x} : {r[x]}") for x in dict(r) if x != "word_vector"])

        keywords = widgets.Output()

        with keywords:

            plt.show(show_word_cloud(r.word_vector ))

        similiar_doc = widgets.Label("TODO")

        tab.children = [paper_info, keywords, similiar_doc]

        tab.set_title(0, "Paper info")

        tab.set_title(1, "Keywords")

        tab.set_title(2, "Similiar documents")



    def run_search(b):

        global results

        if not search_text.value:

            print("text cant be empty")

        else:

            search_widgets.layout.display = 'none'

            search_in_progress.layout.display = None

            results = get_related_documents(search_text.value, number_of_ducuments.value)

            doc_select.options = [x for x in results.title]

            search_in_progress.layout.display = 'none'

            result_widgets.layout.display = None



    search_text = widgets.Text(

        value='potential risks factors Neonates and pregnant women',

        placeholder='Search text',

        description='Search for',

        disabled=False

    )



    number_of_ducuments = widgets.IntText(

        value=5,

        description='Any:',

        disabled=False

    )



    run_button = widgets.Button(

        description='Search',

        disabled=False,

        button_style='',

        tooltip='Search',

        icon='check'

    )

    run_button.on_click(run_search)

    search_widgets = widgets.VBox([search_text, number_of_ducuments, run_button])



    search_in_progress = widgets.Label("searching...")

    search_in_progress.layout.display = 'none'

    

    doc_select = widgets.Select(

        options=[],

        description='Results',

        disabled=False

    )

    doc_select.layout.width = "90%"

    doc_select.observe(document_change)

    result_tab = widgets.Tab()

    result_widgets = widgets.VBox([doc_select, result_tab])

    result_widgets.layout.display = 'none'

    return widgets.VBox([search_widgets, search_in_progress, result_widgets])

search()
Image('/kaggle/input/search-results/1.PNG')
Image('/kaggle/input/search-results/2.PNG')