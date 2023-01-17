from sklearn.feature_extraction.text import CountVectorizer # convert collection of txt docs to matrix of token counts

from sklearn.metrics.pairwise import euclidean_distances # compute distance matrix between each pair of vectors

from datetime import datetime # timestamps



import numpy as np # linear algebra

import pandas as pd # data processing, csv file i/o (e.g. pd.read_csv)

import os # misc os interfaces

import json # json encoder + decoder

# import jsonstreams # writes json as stream

import sys # for constants, functions and methods of py interpreter

import re # regex

import pprint # pretty print

# import jsbeautifier # beautify, unpack or deobfuscate js

# import threading # i/o operations

import concurrent.futures # asynchronously execute callables with threads or processes

import multiprocessing # cpu-heavy operations

import psutil # process and system monitoring

# import logging # event logging for apps and libs
### tasks/questions ###

questions = [

    'WHAT IS KNOWN ABOUT TRANSMISSION, INCUBATION, AND ENVIRONMENTAL STABILITY?',

    'WHAT DO WE KNOW ABOUT COVID-19 RISK FACTORS?',

    'WHAT DO WE KNOW ABOUT VIRUS GENETICS, ORIGIN, AND EVOLUTION?',

    'HELP US UNDERSTAND HOW GEOGRAPHY AFFECTS VIRALITY?',

    'WHAT DO WE KNOW ABOUT VACCINES AND THERAPEUTICS?',

    'WHAT DO WE KNOW ABOUT NON-PHARMACEUTICAL INTERVENTIONS?',

    'WHAT HAS BEEN PUBLISHED ABOUT MEDICAL CARE?',

    'WHAT HAS BEEN PUBLISHED ABOUT ETHICAL AND SOCIAL SCIENCE CONSIDERATIONS?',

    'WHAT HAS BEEN PUBLISHED ABOUT INFORMATION SHARING AND INTER-SECTORAL COLLABORATION?',

    'WHAT DO WE KNOW ABOUT DIAGNOSTICS AND SURVEILLANCE?'

    ]



### detailed questions ###

# answers_0_detail = ['X is known about transmission, incubation, and environmental stability.'] # test

questions_0_detail = [

    'WHAT DO WE KNOW ABOUT NATURAL HISTORY, TRANSMISSION, AND DIAGNOSTICS FOR THE VIRUS?',

    'WHAT HAVE WE LEARNED ABOUT INFECTION PREVENTION AND CONTROL?'

    ]

questions_1_detail = [

    'WHAT HAVE WE LEARNED FROM EPIDEMIOLOGICAL STUDIES?'

    ]

questions_2_detail = [

    'WHAT DO WE KNOW ABOUT THE VIRUS ORIGIN AND MANAGEMENT MEASURES AT THE HUMAN-ANIMAL INTERFACE?'

]

questions_3_detail = [

    'ARE THERE GEOGRAPHIC VARIATIONS IN HOW THE DISEASE WILL SPREAD?',

    'ARE THERE DIFFERENT VARIATIONS OF THE VIRUS IN DIFFERENT AREAS?'

]

questions_4_detail = [

    'WHAT HAS BEEN PUBLISHED CONCERNING RESEARCH AND DEVELOPMENT AND EVALUATION EFFORTS OF VACCINES AND THERAPEUTICS?'

]

questions_5_detail = [

    'WHAT DO WE KNOW ABOUT THE EFFECTIVENESS OF NON-PHARMACEUTICAL INTERVENTIONS?',

    'WHAT IS KNOWN ABOUT EQUITY AND BARRIERS TO COMPLIANCE FOR NON-PHARMACEUTICAL INTERVENTIONS?'

]

questions_6_detail = [

    'WHAT HAS BEEN PUBLISHED CONCERNING SURGE CAPACITY AND NURSING HOMES?',

    'WHAT HAS BEEN PUBLISHED CONCERNING EFFORTS TO INFORM ALLOCATION OF SCARCE RESOURCES?',

    'WHAT DO WE KNOW ABOUT PERSONAL PROTECTIVE EQUIPMENT?',

    'WHAT HAS BEEN PUBLISHED CONCERNING ALTERNATIVE METHODS TO ADVISE ON DISEASE MANAGEMENT?',

    'WHAT HAS BEEN PUBLISHED CONCERNING PROCESSES OF CARE?',

    'WHAT DO WE KNOW ABOUT THE CLINICAL CHARACTERIZATION AND MANAGEMENT OF THE VIRUS?'

]

questions_7_detail = [

    'WHAT HAS BEEN PUBLISHED CONCERNING ETHICAL CONSIDERATIONS FOR RESEARCH?',

    'WHAT HAS BEEN PUBLISHED CONCERNING SOCIAL SCIENCES AT THE OUTBREAK RESPONSE?'

]

questions_8_detail = [

    'WHAT HAS BEEN PUBLISHED ABOUT DATA STANDARDS AND NOMENCLATURE?',

    'WHAT HAS BEEN PUBLISHED ABOUT GOVERNMENTAL PUBLIC HEALTH? WHAT DO WE KNOW ABOUT RISK COMMUNICATION?',

    'WHAT HAS BEEN PUBLISHED ABOUT COMMUNICATING WITH HIGH-RISK POPULATIONS?',

    'WHAT HAS BEEN PUBLISHED TO CLARIFY COMMUNITY MEASURES?',

    'WHAT HAS BEEN PUBLISHED ABOUT EQUITY CONSIDERATIONS AND PROBLEMS OF INEQUITY?'

]

questions_9_detail = [

    'WHAT HAS BEEN PUBLISHED CONCERNING SYSTEMATIC, HOLISTIC APPROACH TO DIAGNOSTICS (FROM THE PUBLIC HEALTH SURVEILLANCE PERSPECTIVE TO BEING ABLE TO PREDICT CLINICAL OUTCOMES)?'

]

###

questions_detail = [questions_0_detail, questions_1_detail, questions_2_detail, questions_3_detail, questions_4_detail, questions_5_detail, questions_6_detail, questions_7_detail, questions_8_detail, questions_9_detail]



### specific questions ###

questions_0_specific = [

    'RANGE OF INCUBATION PERIODS FOR THE DISEASE IN HUMANS (AND HOW THIS VARIES ACROSS AGE AND HEALTH STATUS) AND HOW LONG INDIVIDUALS ARE CONTAGIOUS, EVEN AFTER RECOVERY?',

    'PREVALENCE OF ASYMPTOMATIC SHEDDING AND TRANSMISSION (E.G., PARTICULARLY CHILDREN)?',

    'SEASONALITY OF TRANSMISSION?',

    'PHYSICAL SCIENCE OF THE CORONAVIRUS (E.G., CHARGE DISTRIBUTION, ADHESION TO HYDROPHILIC/PHOBIC SURFACES, ENVIRONMENTAL SURVIVAL TO INFORM DECONTAMINATION EFFORTS FOR AFFECTED AREAS AND PROVIDE INFORMATION ABOUT VIRAL SHEDDING)?',

    'PERSISTENCE AND STABILITY ON A MULTITUDE OF SUBSTRATES AND SOURCES (E.G., NASAL DISCHARGE, SPUTUM, URINE, FECAL MATTER, BLOOD)?',

    'PERSISTENCE OF VIRUS ON SURFACES OF DIFFERENT MATERIALS (E,G., COPPER, STAINLESS STEEL, PLASTIC)?',

    'NATURAL HISTORY OF THE VIRUS AND SHEDDING OF IT FROM AN INFECTED PERSON?',

    'IMPLEMENTATION OF DIAGNOSTICS AND PRODUCTS TO IMPROVE CLINICAL PROCESSES?'

    'DISEASE MODELS, INCLUDING ANIMAL MODELS FOR INFECTION, DISEASE AND TRANSMISSION?',

    'TOOLS AND STUDIES TO MONITOR PHENOTYPIC CHANGE AND POTENTIAL ADAPTATION OF THE VIRUS?',

    'IMMUNE RESPONSE AND IMMUNITY?',

    'EFFECTIVENESS OF MOVEMENT CONTROL STRATEGIES TO PREVENT SECONDARY TRANSMISSION IN HEALTH CARE AND COMMUNITY SETTINGS?',

    'EFFECTIVENESS OF PERSONAL PROTECTIVE EQUIPMENT (PPE) AND ITS USEFULNESS TO REDUCE RISK OF TRANSMISSION IN HEALTH CARE AND COMMUNITY SETTINGS?',

    'ROLE OF THE ENVIRONMENT IN TRANSMISSION?'

    ]

questions_1_specific = [

    'DATA ON POTENTIAL RISKS FACTORS?',

    'SMOKING, PRE-EXISTING PULMONARY DISEASE?',

    'CO-INFECTIONS (DETERMINE WHETHER CO-EXISTING RESPIRATORY/VIRAL INFECTIONS MAKE THE VIRUS MORE TRANSMISSIBLE OR VIRULENT) AND OTHER CO-MORBIDITIES?',

    'NEONATES AND PREGNANT WOMEN?',

    'SOCIO-ECONOMIC AND BEHAVIORAL FACTORS TO UNDERSTAND THE ECONOMIC IMPACT OF THE VIRUS AND WHETHER THERE WERE DIFFERENCES?',

    'TRANSMISSION DYNAMICS OF THE VIRUS, INCLUDING THE BASIC REPRODUCTIVE NUMBER, INCUBATION PERIOD, SERIAL INTERVAL, MODES OF TRANSMISSION AND ENVIRONMENTAL FACTORS?',

    'SEVERITY OF DISEASE, INCLUDING RISK OF FATALITY AMONG SYMPTOMATIC HOSPITALIZED PATIENTS, AND HIGH-RISK PATIENT GROUPS?',

    'SUSCEPTIBILITY OF POPULATIONS?',

    'PUBLIC HEALTH MITIGATION MEASURES THAT COULD BE EFFECTIVE FOR CONTROL?'

]

questions_2_specific = [

    'REAL-TIME TRACKING OF WHOLE GENOMES AND A MECHANISM FOR COORDINATING THE RAPID DISSEMINATION OF THAT INFORMATION TO INFORM THE DEVELOPMENT OF DIAGNOSTICS AND THERAPEUTICS AND TO TRACK VARIATIONS OF THE VIRUS OVER TIME?',

    'ACCESS TO GEOGRAPHIC AND TEMPORAL DIVERSE SAMPLE SETS TO UNDERSTAND GEOGRAPHIC DISTRIBUTION AND GENOMIC DIFFERENCES, AND DETERMINE WHETHER THERE IS MORE THAN ONE STRAIN IN CIRCULATION. MULTI-LATERAL AGREEMENTS SUCH AS THE NAGOYA PROTOCOL COULD BE LEVERAGED?',

    'EVIDENCE THAT LIVESTOCK COULD BE INFECTED (E.G., FIELD SURVEILLANCE, GENETIC SEQUENCING, RECEPTOR BINDING) AND SERVE AS A RESERVOIR AFTER THE EPIDEMIC APPEARS TO BE OVER?',

    'EVIDENCE OF WHETHER FARMERS ARE INFECTED, AND WHETHER FARMERS COULD HAVE PLAYED A ROLE IN THE ORIGIN?',

    'SURVEILLANCE OF MIXED WILDLIFE- LIVESTOCK FARMS FOR SARS-COV-2 AND OTHER CORONAVIRUSES IN SOUTHEAST ASIA?',

    'EXPERIMENTAL INFECTIONS TO TEST HOST RANGE FOR THIS PATHOGEN?',

    'ANIMAL HOST(S) AND ANY EVIDENCE OF CONTINUED SPILL-OVER TO HUMANS?',

    'SOCIOECONOMIC AND BEHAVIORAL RISK FACTORS FOR THIS SPILL-OVER?',

    'SUSTAINABLE RISK REDUCTION STRATEGIES?'

]

questions_3_specific = [

    'ARE THERE GEOGRAPHIC VARIATIONS IN THE RATE OF COVID-19 SPREAD?',

    'ARE THERE GEOGRAPHIC VARIATIONS IN THE MORTALITY RATE OF COVID-19?',

    'IS THERE ANY EVIDENCE TO SUGGEST GEOGRAPHIC BASED VIRUS MUTATIONS?'

]

questions_4_specific = [

    'EFFECTIVENESS OF DRUGS BEING DEVELOPED AND TRIED TO TREAT COVID-19 PATIENTS?',

    'CLINICAL AND BENCH TRIALS TO INVESTIGATE LESS COMMON VIRAL INHIBITORS AGAINST COVID-19 SUCH AS NAPROXEN, CLARITHROMYCIN, AND MINOCYCLINETHAT THAT MAY EXERT EFFECTS ON VIRAL REPLICATION?',

    'METHODS EVALUATING POTENTIAL COMPLICATION OF ANTIBODY-DEPENDENT ENHANCEMENT (ADE) IN VACCINE RECIPIENTS?',

    'EXPLORATION OF USE OF BEST ANIMAL MODELS AND THEIR PREDICTIVE VALUE FOR A HUMAN VACCINE?',

    'CAPABILITIES TO DISCOVER A THERAPEUTIC (NOT VACCINE) FOR THE DISEASE, AND CLINICAL EFFECTIVENESS STUDIES TO DISCOVER THERAPEUTICS, TO INCLUDE ANTIVIRAL AGENTS?',

    'ALTERNATIVE MODELS TO AID DECISION MAKERS IN DETERMINING HOW TO PRIORITIZE AND DISTRIBUTE SCARCE, NEWLY PROVEN THERAPEUTICS AS PRODUCTION RAMPS UP. THIS COULD INCLUDE IDENTIFYING APPROACHES FOR EXPANDING PRODUCTION CAPACITY TO ENSURE EQUITABLE AND TIMELY DISTRIBUTION TO POPULATIONS IN NEED?',

    'EFFORTS TARGETED AT A UNIVERSAL CORONAVIRUS VACCINE?',

    'EFFORTS TO DEVELOP ANIMAL MODELS AND STANDARDIZE CHALLENGE STUDIES?',

    'EFFORTS TO DEVELOP PROPHYLAXIS CLINICAL STUDIES AND PRIORITIZE IN HEALTHCARE WORKERS?',

    'APPROACHES TO EVALUATE RISK FOR ENHANCED DISEASE AFTER VACCINATION?',

    'ASSAYS TO EVALUATE VACCINE IMMUNE RESPONSE AND PROCESS DEVELOPMENT FOR VACCINES, ALONGSIDE SUITABLE ANIMAL MODELS [IN CONJUNCTION WITH THERAPEUTICS]?'

]

questions_5_specific = [

    'GUIDANCE ON WAYS TO SCALE UP NPIS IN A MORE COORDINATED WAY (E.G., ESTABLISH FUNDING, INFRASTRUCTURE AND AUTHORITIES TO SUPPORT REAL TIME, AUTHORITATIVE (QUALIFIED PARTICIPANTS) COLLABORATION WITH ALL STATES TO GAIN CONSENSUS ON CONSISTENT GUIDANCE AND TO MOBILIZE RESOURCES TO GEOGRAPHIC AREAS WHERE CRITICAL SHORTFALLS ARE IDENTIFIED) TO GIVE US TIME TO ENHANCE OUR HEALTH CARE DELIVERY SYSTEM CAPACITY TO RESPOND TO AN INCREASE IN CASES?',

    'RAPID DESIGN AND EXECUTION OF EXPERIMENTS TO EXAMINE AND COMPARE NPIS CURRENTLY BEING IMPLEMENTED. DHS CENTERS FOR EXCELLENCE COULD POTENTIALLY BE LEVERAGED TO CONDUCT THESE EXPERIMENTS?',

    'RAPID ASSESSMENT OF THE LIKELY EFFICACY OF SCHOOL CLOSURES, TRAVEL BANS, BANS ON MASS GATHERINGS OF VARIOUS SIZES, AND OTHER SOCIAL DISTANCING APPROACHES?',

    'METHODS TO CONTROL THE SPREAD IN COMMUNITIES, BARRIERS TO COMPLIANCE AND HOW THESE VARY AMONG DIFFERENT POPULATIONS?',

    'MODELS OF POTENTIAL INTERVENTIONS TO PREDICT COSTS AND BENEFITS THAT TAKE ACCOUNT OF SUCH FACTORS AS RACE, INCOME, DISABILITY, AGE, GEOGRAPHIC LOCATION, IMMIGRATION STATUS, HOUSING STATUS, EMPLOYMENT STATUS, AND HEALTH INSURANCE STATUS?',

    'POLICY CHANGES NECESSARY TO ENABLE THE COMPLIANCE OF INDIVIDUALS WITH LIMITED RESOURCES AND THE UNDERSERVED WITH NPIS?',

    'RESEARCH ON WHY PEOPLE FAIL TO COMPLY WITH PUBLIC HEALTH ADVICE, EVEN IF THEY WANT TO DO SO (E.G., SOCIAL OR FINANCIAL COSTS MAY BE TOO HIGH)?',

    'RESEARCH ON THE ECONOMIC IMPACT OF THIS OR ANY PANDEMIC. THIS WOULD INCLUDE IDENTIFYING POLICY AND PROGRAMMATIC ALTERNATIVES THAT LESSEN/MITIGATE RISKS TO CRITICAL GOVERNMENT SERVICES, FOOD DISTRIBUTION AND SUPPLIES, ACCESS TO CRITICAL HOUSEHOLD SUPPLIES, AND ACCESS TO HEALTH DIAGNOSES, TREATMENT, AND NEEDED CARE, REGARDLESS OF ABILITY TO PAY?'

]

questions_6_specific = [

    'RESOURCES TO SUPPORT SKILLED NURSING FACILITIES AND LONG TERM CARE FACILITIES?',

    'MOBILIZATION OF SURGE MEDICAL STAFF TO ADDRESS SHORTAGES IN OVERWHELMED COMMUNITIES?',

    'AGE-ADJUSTED MORTALITY DATA FOR ACUTE RESPIRATORY DISTRESS SYNDROME (ARDS) WITH/WITHOUT OTHER ORGAN FAILURE – PARTICULARLY FOR VIRAL ETIOLOGIES?',

    'EXTRACORPOREAL MEMBRANE OXYGENATION (ECMO) OUTCOMES DATA OF COVID-19 PATIENTS?',

    'OUTCOMES DATA FOR COVID-19 AFTER MECHANICAL VENTILATION ADJUSTED FOR AGE?',

    'KNOWLEDGE OF THE FREQUENCY, MANIFESTATIONS, AND COURSE OF EXTRAPULMONARY MANIFESTATIONS OF COVID-19, INCLUDING, BUT NOT LIMITED TO, POSSIBLE CARDIOMYOPATHY AND CARDIAC ARREST?',

    'APPLICATION OF REGULATORY STANDARDS (E.G., EUA, CLIA) AND ABILITY TO ADAPT CARE TO CRISIS STANDARDS OF CARE LEVEL?',

    'APPROACHES FOR ENCOURAGING AND FACILITATING THE PRODUCTION OF ELASTOMERIC RESPIRATORS, WHICH CAN SAVE THOUSANDS OF N95 MASKS?',

    'BEST TELEMEDICINE PRACTICES, BARRIERS AND FACIITATORS, AND SPECIFIC ACTIONS TO REMOVE/EXPAND THEM WITHIN AND ACROSS STATE BOUNDARIES?',

    'GUIDANCE ON THE SIMPLE THINGS PEOPLE CAN DO AT HOME TO TAKE CARE OF SICK PEOPLE AND MANAGE DISEASE?',

    'ORAL MEDICATIONS THAT MIGHT POTENTIALLY WORK?',

    'USE OF AI IN REAL-TIME HEALTH CARE DELIVERY TO EVALUATE INTERVENTIONS, RISK FACTORS, AND OUTCOMES IN A WAY THAT COULD NOT BE DONE MANUALLY?',

    'BEST PRACTICES AND CRITICAL CHALLENGES AND INNOVATIVE SOLUTIONS AND TECHNOLOGIES IN HOSPITAL FLOW AND ORGANIZATION, WORKFORCE PROTECTION, WORKFORCE ALLOCATION, COMMUNITY-BASED SUPPORT RESOURCES, PAYMENT, AND SUPPLY CHAIN MANAGEMENT TO ENHANCE CAPACITY, EFFICIENCY, AND OUTCOMES?',

    'EFFORTS TO DEFINE THE NATURAL HISTORY OF DISEASE TO INFORM CLINICAL CARE, PUBLIC HEALTH INTERVENTIONS, INFECTION PREVENTION CONTROL, TRANSMISSION, AND CLINICAL TRIALS?',

    'EFFORTS TO DEVELOP A CORE CLINICAL OUTCOME SET TO MAXIMIZE USABILITY OF DATA ACROSS A RANGE OF TRIALS?',

    'EFFORTS TO DETERMINE ADJUNCTIVE AND SUPPORTIVE INTERVENTIONS THAT CAN IMPROVE THE CLINICAL OUTCOMES OF INFECTED PATIENTS (E.G. STEROIDS, HIGH FLOW OXYGEN)?'

]

questions_7_specific = [

    'EFFORTS TO ARTICULATE AND TRANSLATE EXISTING ETHICAL PRINCIPLES AND STANDARDS TO SALIENT ISSUES IN COVID-2019?',

    'EFFORTS TO EMBED ETHICS ACROSS ALL THEMATIC AREAS, ENGAGE WITH NOVEL ETHICAL ISSUES THAT ARISE AND COORDINATE TO MINIMIZE DUPLICATION OF OVERSIGHT?',

    'EFFORTS TO SUPPORT SUSTAINED EDUCATION, ACCESS, AND CAPACITY BUILDING IN THE AREA OF ETHICS?',

    'EFFORTS TO ESTABLISH A TEAM AT WHO THAT WILL BE INTEGRATED WITHIN MULTIDISCIPLINARY RESEARCH AND OPERATIONAL PLATFORMS AND THAT WILL CONNECT WITH EXISTING AND EXPANDED GLOBAL NETWORKS OF SOCIAL SCIENCES?',

    'EFFORTS TO DEVELOP QUALITATIVE ASSESSMENT FRAMEWORKS TO SYSTEMATICALLY COLLECT INFORMATION RELATED TO LOCAL BARRIERS AND ENABLERS FOR THE UPTAKE AND ADHERENCE TO PUBLIC HEALTH MEASURES FOR PREVENTION AND CONTROL. THIS INCLUDES THE RAPID IDENTIFICATION OF THE SECONDARY IMPACTS OF THESE MEASURES. (E.G. USE OF SURGICAL MASKS, MODIFICATION OF HEALTH SEEKING BEHAVIORS FOR SRH, SCHOOL CLOSURES)?',

    'EFFORTS TO IDENTIFY HOW THE BURDEN OF RESPONDING TO THE OUTBREAK AND IMPLEMENTING PUBLIC HEALTH MEASURES AFFECTS THE PHYSICAL AND PSYCHOLOGICAL HEALTH OF THOSE PROVIDING CARE FOR COVID-19 PATIENTS AND IDENTIFY THE IMMEDIATE NEEDS THAT MUST BE ADDRESSED?',

    'EFFORTS TO IDENTIFY THE UNDERLYING DRIVERS OF FEAR, ANXIETY AND STIGMA THAT FUEL MISINFORMATION AND RUMOR, PARTICULARLY THROUGH SOCIAL MEDIA?'

]

questions_8_specific = [

    'METHODS FOR COORDINATING DATA-GATHERING WITH STANDARDIZED NOMENCLATURE?',

    'SHARING RESPONSE INFORMATION AMONG PLANNERS, PROVIDERS, AND OTHERS?',

    'UNDERSTANDING AND MITIGATING BARRIERS TO INFORMATION-SHARING?',

    'HOW TO RECRUIT, SUPPORT, AND COORDINATE LOCAL (NON-FEDERAL) EXPERTISE AND CAPACITY RELEVANT TO PUBLIC HEALTH EMERGENCY RESPONSE (PUBLIC, PRIVATE, COMMERCIAL AND NON-PROFIT, INCLUDING ACADEMIC)?',

    'INTEGRATION OF FEDERAL/STATE/LOCAL PUBLIC HEALTH SURVEILLANCE SYSTEMS?',

    'VALUE OF INVESTMENTS IN BASELINE PUBLIC HEALTH RESPONSE INFRASTRUCTURE PREPAREDNESS?',

    'MODES OF COMMUNICATING WITH TARGET HIGH-RISK POPULATIONS (ELDERLY, HEALTH CARE WORKERS)?',

    'RISK COMMUNICATION AND GUIDELINES THAT ARE EASY TO UNDERSTAND AND FOLLOW (INCLUDE TARGETING AT RISK POPULATIONS’ FAMILIES TOO)?',

    'COMMUNICATION THAT INDICATES POTENTIAL RISK OF DISEASE TO ALL POPULATION GROUPS?',

    'MISUNDERSTANDING AROUND CONTAINMENT AND MITIGATION?',

    'ACTION PLAN TO MITIGATE GAPS AND PROBLEMS OF INEQUITY IN THE NATION’S PUBLIC HEALTH CAPABILITY, CAPACITY, AND FUNDING TO ENSURE ALL CITIZENS IN NEED ARE SUPPORTED AND CAN ACCESS INFORMATION, SURVEILLANCE, AND TREATMENT?',

    'MEASURES TO REACH MARGINALIZED AND DISADVANTAGED POPULATIONS?',

    'DATA SYSTEMS AND RESEARCH PRIORITIES AND AGENDAS INCORPORATE ATTENTION TO THE NEEDS AND CIRCUMSTANCES OF DISADVANTAGED POPULATIONS AND UNDERREPRESENTED MINORITIES?',

    'MITIGATING THREATS TO INCARCERATED PEOPLE FROM COVID-19, ASSURING ACCESS TO INFORMATION, PREVENTION, DIAGNOSIS, AND TREATMENT?',

    'UNDERSTANDING COVERAGE POLICIES (BARRIERS AND OPPORTUNITIES) RELATED TO TESTING, TREATMENT, AND CARE?'

]

questions_9_specific = [

    'HOW WIDESPREAD CURRENT EXPOSURE IS TO BE ABLE TO MAKE IMMEDIATE POLICY RECOMMENDATIONS ON MITIGATION MEASURES. DENOMINATORS FOR TESTING AND A MECHANISM FOR RAPIDLY SHARING THAT INFORMATION, INCLUDING DEMOGRAPHICS, TO THE EXTENT POSSIBLE. SAMPLING METHODS TO DETERMINE ASYMPTOMATIC DISEASE (E.G., USE OF SEROSURVEYS (SUCH AS CONVALESCENT SAMPLES) AND EARLY DETECTION OF DISEASE (E.G., USE OF SCREENING OF NEUTRALIZING ANTIBODIES SUCH AS ELISAS)?',

    'EFFORTS TO INCREASE CAPACITY ON EXISTING DIAGNOSTIC PLATFORMS AND TAP INTO EXISTING SURVEILLANCE PLATFORMS?',

    'RECRUITMENT, SUPPORT, AND COORDINATION OF LOCAL EXPERTISE AND CAPACITY (PUBLIC, PRIVATE—COMMERCIAL, AND NON-PROFIT, INCLUDING ACADEMIC), INCLUDING LEGAL, ETHICAL, COMMUNICATIONS, AND OPERATIONAL ISSUES?',

    'NATIONAL GUIDANCE AND GUIDELINES ABOUT BEST PRACTICES TO STATES (E.G., HOW STATES MIGHT LEVERAGE UNIVERSITIES AND PRIVATE LABORATORIES FOR TESTING PURPOSES, COMMUNICATIONS TO PUBLIC HEALTH OFFICIALS AND THE PUBLIC)?',

    'DEVELOPMENT OF A POINT-OF-CARE TEST (LIKE A RAPID INFLUENZA TEST) AND RAPID BED-SIDE TESTS, RECOGNIZING THE TRADEOFFS BETWEEN SPEED, ACCESSIBILITY, AND ACCURACY?',

    'RAPID DESIGN AND EXECUTION OF TARGETED SURVEILLANCE EXPERIMENTS CALLING FOR ALL POTENTIAL TESTERS USING PCR IN A DEFINED AREA TO START TESTING AND REPORT TO A SPECIFIC ENTITY. THESE EXPERIMENTS COULD AID IN COLLECTING LONGITUDINAL SAMPLES, WHICH ARE CRITICAL TO UNDERSTANDING THE IMPACT OF AD HOC LOCAL INTERVENTIONS (WHICH ALSO NEED TO BE RECORDED)?',

    'SEPARATION OF ASSAY DEVELOPMENT ISSUES FROM INSTRUMENTS, AND THE ROLE OF THE PRIVATE SECTOR TO HELP QUICKLY MIGRATE ASSAYS ONTO THOSE DEVICES?',

    'EFFORTS TO TRACK THE EVOLUTION OF THE VIRUS (I.E., GENETIC DRIFT OR MUTATIONS) AND AVOID LOCKING INTO SPECIFIC REAGENTS AND SURVEILLANCE/DETECTION SCHEMES?',

    'LATENCY ISSUES AND WHEN THERE IS SUFFICIENT VIRAL LOAD TO DETECT THE PATHOGEN, AND UNDERSTANDING OF WHAT IS NEEDED IN TERMS OF BIOLOGICAL AND ENVIRONMENTAL SAMPLING?',

    'USE OF DIAGNOSTICS SUCH AS HOST RESPONSE MARKERS (E.G., CYTOKINES) TO DETECT EARLY DISEASE OR PREDICT SEVERE DISEASE PROGRESSION, WHICH WOULD BE IMPORTANT TO UNDERSTANDING BEST CLINICAL PRACTICE AND EFFICACY OF THERAPEUTIC INTERVENTIONS?',

    'POLICIES AND PROTOCOLS FOR SCREENING AND TESTING?',

    'POLICIES TO MITIGATE THE EFFECTS ON SUPPLIES ASSOCIATED WITH MASS TESTING, INCLUDING SWABS AND REAGENTS?',

    'TECHNOLOGY ROADMAP FOR DIAGNOSTICS?',

    'BARRIERS TO DEVELOPING AND SCALING UP NEW DIAGNOSTIC TESTS (E.G., MARKET FORCES), HOW FUTURE COALITION AND ACCELERATOR MODELS (E.G., COALITION FOR EPIDEMIC PREPAREDNESS INNOVATIONS) COULD PROVIDE CRITICAL FUNDING FOR DIAGNOSTICS, AND OPPORTUNITIES FOR A STREAMLINED REGULATORY ENVIRONMENT?',

    'NEW PLATFORMS AND TECHNOLOGY (E.G., CRISPR) TO IMPROVE RESPONSE TIMES AND EMPLOY MORE HOLISTIC APPROACHES TO COVID-19 AND FUTURE DISEASES?',

    'COUPLING GENOMICS AND DIAGNOSTIC TESTING ON A LARGE SCALE?',

    'ENHANCE CAPABILITIES FOR RAPID SEQUENCING AND BIOINFORMATICS TO TARGET REGIONS OF THE GENOME THAT WILL ALLOW SPECIFICITY FOR A PARTICULAR VARIANT?',

    'ENHANCE CAPACITY (PEOPLE, TECHNOLOGY, DATA) FOR SEQUENCING WITH ADVANCED ANALYTICS FOR UNKNOWN PATHOGENS, AND EXPLORE CAPABILITIES FOR DISTINGUISHING NATURALLY-OCCURRING PATHOGENS FROM INTENTIONAL?',

    'ONE HEALTH SURVEILLANCE OF HUMANS AND POTENTIAL SOURCES OF FUTURE SPILLOVER OR ONGOING EXPOSURE FOR THIS ORGANISM AND FUTURE PATHOGENS, INCLUDING BOTH EVOLUTIONARY HOSTS (E.G., BATS) AND TRANSMISSION HOSTS (E.G., HEAVILY TRAFFICKED AND FARMED WILDLIFE AND DOMESTIC FOOD AND COMPANION SPECIES), INCLUSIVE OF ENVIRONMENTAL, DEMOGRAPHIC, AND OCCUPATIONAL RISK FACTORS?'

]

###

questions_specific = [questions_0_specific, questions_1_specific, questions_2_specific, questions_3_specific, questions_4_specific, questions_5_specific, questions_6_specific, questions_7_specific, questions_8_specific, questions_9_specific]
def fct_create_dict_of_papers(input_dir, input_path_of_papers, input_type_of_papers='json'): # see glob

    dict_of_papers = {}

    dict_of_papers['paper'] = []

    dict_file = '/kaggle/working/papers.' + input_dir + '.json'

    os.makedirs(os.path.dirname(dict_file), exist_ok=True)

    with open(dict_file, 'w') as open_file:

        for dirname, _, filenames in os.walk(input_path_of_papers):

            for filename in filenames:

                dir_file = os.path.join(dirname, filename)

                if (filename).endswith(input_type_of_papers) and re.search('^(?=.*[0-9])(?=.*[a-z])([a-z0-9]+)$', filename.split('.')[0]) and ('/' + input_dir) in dirname: # comm/noncomm # regex: 860ffcab2771f1935ac5a59e986d416a603b3de3

                    dict_of_papers['paper'].extend([dir_file])

                break # test for kaggle

        json.dump(dict_of_papers, open_file, indent=2, separators=(',', ': '))

    print('*** fct_create_dict_of_papers ' + str(datetime.now()) + ' ***')

    # return dict_of_papers # test



### multiprocessing by dir - /biorxiv_medrxiv, /comm_use_subset, /noncomm_use_subset, /pmc_custom_license: v. multithreading ###

# print('parent process id: ', os.getppid())

# print('child process id: ', os.getpid())



process_0_create_dict_of_papers = multiprocessing.Process(target=fct_create_dict_of_papers, args=('biorxiv_medrxiv', '/kaggle/input/CORD-19-research-challenge', 'json')) # args=(,) so it is a tuple if there is only 1 args

# process_1_create_dict_of_papers = multiprocessing.Process(target=fct_create_dict_of_papers, args=('comm_use_subset', '/kaggle/input/CORD-19-research-challenge', 'json'))

# process_2_create_dict_of_papers = multiprocessing.Process(target=fct_create_dict_of_papers, args=('noncomm_use_subset', '/kaggle/input/CORD-19-research-challenge', 'json'))

# process_3_create_dict_of_papers = multiprocessing.Process(target=fct_create_dict_of_papers, args=('pmc_custom_license', '/kaggle/input/CORD-19-research-challenge', 'json'))



### execute processes in parallel ###

process_0_create_dict_of_papers.start()

# process_1_create_dict_of_papers.start()

# process_2_create_dict_of_papers.start()

# process_3_create_dict_of_papers.start()



### join processes back to the parent process (this) ###

process_0_create_dict_of_papers.join()

# process_1_create_dict_of_papers.join()

# process_2_create_dict_of_papers.join()

# process_3_create_dict_of_papers.join()
def fct_get_matches_from_papers(input_path_to_paper, input_question, input_min_df=0.1, input_max_df=0.9): # see DataFrame

    pp = pprint.PrettyPrinter()

    paper = []

    '''

    each paper/item has the following structure:

    {

        "paper_id": "...", *

        "metadata": {...}, *

        "abstract": [...], *

        "body_text": [...], *

        "bib_entries": {...},

        "ref_entries": {...},

        "back_matter": [...]

    }

    '''

    with open(input_path_to_paper) as paper_item: # todo: KeyError

        # v. json.loads() doesn't take the file path, but the file contents as a string

        paper_item_json_load = json.load(paper_item)

        ### question: append [1, 2, 3, [4, 5]] v. extend [1, 2, 3, 4, 5] ###

        paper.extend([input_question])

        ### id ###

        paper_id = paper_item_json_load['paper_id']

        ### title, authors ###

        metadata = paper_item_json_load['metadata']

        ### abstract, citation/reference spans ###

        abstract = paper_item_json_load['abstract']

        for index, item in enumerate(abstract):

            paper.extend([abstract[index]['text']])

            ### text, citation/reference spans ###

            body_text = paper_item_json_load['body_text']

            for index, item in enumerate(body_text):

                paper.extend([body_text[index]['text']])

                ### bibliography ###

                # bib_entries = paper_item_json_load['bib_entries']

                ### findings ###

                # ref_entries = paper_item_json_load['ref_entries']

                ### funding, conflict of interest ###

                # back_matter = paper_item_json_load['back_matter']

    # pp.pprint(paper) # test: ['WHAT HAVE WE LEARNED ABOUT INFECTION PREVENTION AND CONTROL?',...]

    

    paper_match = []

    try:

        ### min_df, max_df and ngram_range ###

        vectorizer = CountVectorizer(vocabulary=None, min_df=input_min_df, max_df=input_max_df, analyzer='word', ngram_range=(2, 3), stop_words=None)

        ### learn vocabulary dict and return term-document matrix ###    

        features = vectorizer.fit_transform(paper).todense() # todense() returns matrix v. toarray() returns ndarray

        # print(vectorizer.vocabulary_) # test: {'dock8 deficient': 2,...}

        ### for feature in features ###

        for index, item in enumerate(features):

            # print(euclidean_distances(features[0], features[i]), paper[i]) # test: [[0.]] WHAT HAVE WE LEARNED ABOUT INFECTION PREVENTION AND CONTROL?

            paper_match.append([euclidean_distances(features[0], features[index])[0][0], paper[index]])

        # print(paper_match) # test: [[0.0, 'WHAT HAVE WE LEARNED ABOUT INFECTION PREVENTION AND CONTROL?'],...]

    except ValueError: # eg papers may req diff min_df or max_df

        ### min_df, max_df and ngram_range ###

        vectorizer = CountVectorizer(vocabulary=None, min_df=0.0, max_df=1.0, analyzer='word', ngram_range=(2, 3), stop_words=None)

        ### learn vocabulary dict and return term-document matrix ###    

        features = vectorizer.fit_transform(paper).todense()

        ### for feature in features ###

        for index, item in enumerate(features):

            paper_match.append([euclidean_distances(features[0], features[index])[0][0], paper[index]])

    print('*** fct_get_matches_from_papers ' + str(datetime.now()) + ' ***')

    return sorted(paper_match, key=lambda k: k[0]) # returned vars don't get garbaged + remain accessible after the fct()
def fct_get_answer_from_matches(input_path_to_paper, input_min_df, input_max_df, input_task=0, input_top=4):

    pp = pprint.PrettyPrinter()

    answer = []

    for index in range(input_top):

        try:

            if (fct_get_matches_from_papers(input_path_to_paper, questions[input_task], input_min_df, input_max_df)[index][0] != 0.0): # remove original question (+ no perfect answer with score = 0.0?)

                answer.append(fct_get_matches_from_papers(input_path_to_paper, questions[input_task], input_min_df, input_max_df)[index][1]) # (question)[row(distance, answer)][answer]

        except IndexError:

            pass

    for question in questions_detail[input_task]:

        ### get paper/text: if euclidean distance < #, may return empty, hence top 3 (after removing original question) ###

        # print(question) # test: WHAT IS KNOWN ABOUT TRANSMISSION, INCUBATION, AND ENVIRONMENTAL STABILITY?

        for index in range(input_top):

            try:

                if (fct_get_matches_from_papers(input_path_to_paper, question, input_min_df, input_max_df)[index][0] != 0.0):

                    answer.append(fct_get_matches_from_papers(input_path_to_paper, question, input_min_df, input_max_df)[index][1])

            except IndexError:

                pass

    for question in questions_specific[input_task]:

        ### get paper/text ###

        for index in range(input_top):

            try:

                if (fct_get_matches_from_papers(input_path_to_paper, question, input_min_df, input_max_df)[index][0] != 0.0):

                    answer.append(fct_get_matches_from_papers(input_path_to_paper, question, input_min_df, input_max_df)[index][1])

            except IndexError:

                pass

    answer_unique = list(dict.fromkeys(answer)) # also remove duplicate distances

    # pp.pprint(answer_unique) # test

    print('*** fct_get_answer_from_matches ' + str(datetime.now()) + ' ***')

    return answer_unique
def fct_write_answers(input_path_of_papers, input_min_df, input_max_df, input_task, input_top):

    with open(input_path_of_papers) as open_file:

        json_file = json.load(open_file)

        for index, item in enumerate(json_file['paper']):

            print('*** ' + item + ' ' + str(datetime.now()) + ' ***')

            ### id ###

            paper_file = item.split('/')[-1]

            paper_id = paper_file.split('.')[0]

            # todo: add title

            ### json ###

            answer_json = {

                "paper_id": paper_id,

                "task": input_task,

                "abstract": questions[input_task],

                "body_text": fct_get_answer_from_matches(item, input_min_df, input_max_df, input_task, input_top) # item = input_path_to_paper

            }

            output_answer_file = '/kaggle/working' + item

            os.makedirs(os.path.dirname(output_answer_file), exist_ok=True)

            with open(output_answer_file, 'w') as open_answer:

                json.dump(answer_json, open_answer, indent=2, separators=(',', ': '))

    print('*** fct_write_answers ' + str(datetime.now()) + ' ***')



### multiprocessing ###

process_0_write_answers = multiprocessing.Process(target=fct_write_answers, args=('/kaggle/working/papers.biorxiv_medrxiv.json', 0.1, 0.9, 0, 4))

# process_1_write_answers = multiprocessing.Process(target=fct_write_answers, args=('/kaggle/working/papers.comm_use_subset.json', 0.1, 0.9, 0, 4))

# process_2_write_answers = multiprocessing.Process(target=fct_write_answers, args=('/kaggle/working/papers.noncomm_use_subset.json', 0.1, 0.9, 0, 4))

# process_3_write_answers = multiprocessing.Process(target=fct_write_answers, args=('/kaggle/working/papers.pmc_custom_license.json', 0.1, 0.9, 0, 4))



### execute processes in parallel ###

process_0_write_answers.start()

# process_1_write_answers.start()

# process_2_write_answers.start()

# process_3_write_answers.start()



### join processes back to the parent process (this) ###

process_0_write_answers.join()

# process_1_write_answers.join()

# process_2_write_answers.join()

# process_3_write_answers.join()
def fct_merge_answers(input_task, input_dir, input_path_of_answers, input_type_of_answers='json'):

    answer_file = '/kaggle/working/answers.task.' + str(input_task) + '.' + input_dir + '.json'

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    with open(answer_file, 'w') as open_file:

        open_file.write('[')

        for dirname, _, filenames in os.walk(input_path_of_answers):

            for filename in filenames:

                dir_file = os.path.join(dirname, filename)

                if (filename).endswith(input_type_of_answers) and re.search('^(?=.*[0-9])(?=.*[a-z])([a-z0-9]+)$', filename.split('.')[0]) and ('/' + input_dir) in dirname:

                    with open(dir_file) as answer_item:

                        print('*** ' + dir_file + ' ' + str(datetime.now()) + ' ***')

                        '''

                        {

                          "paper_id": "...",

                          "task": 0,

                          "abstract": "...?",

                          "body_text": ["..."]

                        }

                        '''

                        answer_item_json_load = json.load(answer_item)

                        try:

                            if (answer_item_json_load['task'] == input_task):

                                json.dump(answer_item_json_load, open_file, indent=2, separators=(',', ': '))

                        except KeyError:

                            json.dump({}, open_file, indent=2, separators=(',', ': '))

                        if (filename != filenames[-1]):

                            open_file.write(',')

        open_file.write(']')

    print('*** fct_merge_answers ' + str(datetime.now()) + ' ***')



### multiprocessing ###

process_0_merge_answers = multiprocessing.Process(target=fct_merge_answers, args=(0, 'biorxiv_medrxiv', '/kaggle/working/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/', 'json'))

# process_1_merge_answers = multiprocessing.Process(target=fct_merge_answers, args=(0, 'comm_use_subset', '/kaggle/working/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/', 'json'))

# process_2_merge_answers = multiprocessing.Process(target=fct_merge_answers, args=(0, 'noncomm_use_subset', '/kaggle/working/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/', 'json'))

# process_3_merge_answers = multiprocessing.Process(target=fct_merge_answers, args=(0, 'pmc_custom_license', '/kaggle/working/CORD-19-research-challenge/2020-03-13/pmc_custom_license/pmc_custom_license/', 'json'))



### execute processes in parallel ###

process_0_merge_answers.start()

# process_1_merge_answers.start()

# process_2_merge_answers.start()

# process_3_merge_answers.start()



### join processes back to the parent process (this) ###

process_0_merge_answers.join()

# process_1_merge_answers.join()

# process_2_merge_answers.join()

# process_3_merge_answers.join()