%cd /kaggle/input/
!pip install sns


from nltk.stem import WordNetLemmatizer
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


ls
import pandas as pd
bio = pd.read_csv("/kaggle/input/updatedcovi19dataset/biorxiv_clean.csv")
noncomm = pd.read_csv("/kaggle/input/updatedcovi19dataset/clean_noncomm_use.csv")
comm = pd.read_csv("/kaggle/input/updatedcovi19dataset/clean_comm_use.csv")
pmc = pd.read_csv("/kaggle/input/updatedcovi19dataset/clean_pmc.csv")
import seaborn as sns
fig, axes = plt.subplots(2 ,2, figsize=(30,15))
cmap = sns.light_palette((260, 75, 60), input="husl")
title = "Biorxiv"
ax = sns.heatmap(bio.isnull(), cmap=cmap, cbar=False, ax=axes[0,0])
ax.vlines([1,2,3,4,5,6,7,8,9], *ax.get_ylim(), color="black")
axes[0,0].set_title(title, fontsize=15)

title = "Non-commercial Use"
ax = sns.heatmap(noncomm.isnull(), cmap=cmap, cbar=False, ax=axes[0,1])
ax.vlines([1,2,3,4,5,6,7,8,9], *ax.get_ylim(), color="black")
axes[0,1].set_title(title, fontsize=15)

title = "Commercial Use"
ax = sns.heatmap(comm.isnull(), cmap=cmap, cbar=False, ax=axes[1,0])
ax.vlines([1,2,3,4,5,6,7,8,9], *ax.get_ylim(), color="black")
axes[1,0].set_title(title, fontsize=15)

title = "PMC"
ax = sns.heatmap(pmc.isnull(), cmap=cmap, cbar=False, ax=axes[1,1])
ax.vlines([1,2,3,4,5,6,7,8,9], *ax.get_ylim(), color="black")
axes[1,1].set_title(title, fontsize=15)

fig.suptitle("Missing Value Heatmaps for all 4 datasets", fontsize=20)
plt.show()

bio = bio.fillna("Missing")
noncomm = noncomm.fillna("Missing")
comm = comm.fillna("Missing")
pmc = pmc.fillna("Missing")
# Concatenate all the dataframes together
papers = pd.concat([bio, comm, noncomm, pmc], ignore_index=True)
papers['abstract'].describe(include='all')
papers['text'].describe(include='all')
import re
# Data Cleaning
def clean_up(t):
    """
    Cleans up the passed value
    """
    t = str(t)
    # Remove New Lines
    t = t.replace("\n"," ") # removes newlines
#     print(t)
    # Remove citaton numbers (Eg.: [4])
    t = re.sub(r"\[[0-9]+(, [0-9]+)*\]", "", t)

    # Remove et al.
    t = re.sub("et al.", "", t)

    # Remove Fig and Table
    t = re.sub(r"\( ?Fig [0-9]+ ?\)", "", t)
    t = re.sub(r"\( ?Table [0-9]+ ?\)", "", t)
    
    # Replace continuous spaces with a single space
    t = re.sub(r' +', ' ', t)
    
    # Convert all to lowercase
    t = t.lower()
    return t

papers['abstract'] = papers["abstract"].apply(clean_up)
papers['text'] = papers["text"].apply(clean_up)
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

papers['text'] = papers['text'].apply(lambda x: lower_case(str(x)))
papers['abstract'] = papers['abstract'].apply(lambda x: lower_case(str(x)))
papers.head(4)
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

stopwords = set(STOPWORDS)
#https://www.kaggle.com/gpreda/cord-19-solution-toolbox

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=30, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
show_wordcloud(papers['abstract'], title = 'All - papers Abstract - frequent words (200 sample)')
show_wordcloud(papers['text'], title = 'All - papers Body Text - frequent words (200 sample)')
all_texts = papers['text'].values

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

stopwords = stopwords.words('english')
more_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI',
    '-PRON-', 'de','la','le','en','el', 'que', 'un', 'e','los','se']
stopwords.extend(more_stop_words)
def tokenize(s, lemmatize=True, decode=False):
    
    # NLTK data 
    try:
        if decode:
            s = s.decode("utf-8")
        tokens = word_tokenize(s.lower())
    except LookupError:
        nltk.download('punkt')
        tokenize(s)
    

    
    ignored = stopwords + [punct for punct in string.punctuation]
    clean_tokens = [token for token in tokens if token not in ignored]
    
    #lemmatize the output to reduce the number of unique words and address overfitting.
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in clean_tokens]
    return clean_tokens


from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
 
vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=2000,
                                lowercase=True, tokenizer=tokenize)
X = vectorizer.fit_transform(tqdm(all_texts))
%cd working/
import joblib
joblib.dump(vectorizer, 'vectorizer.csv')
joblib.dump(X, 'xvector.csv')
pwd
import joblib
vectorizer =  joblib.load('/kaggle/input/updatedvectorscovid19/vectorizer.csv')
X =  joblib.load("/kaggle/input/updatedvectorscovid19/xvector.csv")
from sklearn.model_selection import train_test_split

# test set size of 20% of the data and the random seed 42 <3
X_train, X_test = train_test_split(X.toarray(), test_size=0.2, random_state=42)

print("X_train size:", len(X_train))
print("X_test size:", len(X_test), "\n")

!pip install faiss-gpu
import faiss    
res = faiss.StandardGpuResources()  # use a single GPU

#use the elbow rule method to determine the number of clusters

MAX_K = 25
K = range(1,MAX_K+1)
inertias = np.zeros(MAX_K)
diff = np.zeros(MAX_K)
diff2 = np.zeros(MAX_K)
diff3 = np.zeros(MAX_K)
for k in K:
    
    niter = 10
    verbose = True
    d = X.shape[1]
    kmeans = faiss.Kmeans(d, k, niter=niter, verbose=verbose)
    kmeans.train(X.toarray().astype("float32"))
    inertias[k - 1] = kmeans.obj[-1]
    print(kmeans.obj[-1])
    # first difference    
    if k > 1:
        diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    # second difference
    if k > 2:
        diff2[k - 1] = diff[k - 1] - diff[k - 2]
    # third difference
    if k > 3:
        diff3[k - 1] = diff2[k - 1] - diff2[k - 2]
elbow = np.argmin(diff3[3:]) + 3
print("Elbow "+str(elbow))
plt.plot(K, inertias, "b*-")
plt.plot(K[elbow], inertias[elbow], marker='o', markersize=12,
             markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
plt.ylabel("Inertia")
plt.xlabel("K")
plt.show()
niter =14
clusters_byElbowRule = 14 # according to the above graph total elbows are 7
kmeans = faiss.Kmeans(d, clusters_byElbowRule, niter=niter, verbose=verbose)
kmeans.train(X.toarray().astype("float32"))
D, y = kmeans.index.search(X.toarray().astype("float32"), 1)
y = y[:,0]

#scatter plot of clusters of 1000 instances

colors = ["b", "g", "r", "m", "c","y"]
for i in range(X.shape[0]):
    plt.scatter(X[i,0], X[i,1], c=colors[y[i]], s=10)    
plt.show()
from  sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=clusters_byElbowRule, random_state=0)
lda.fit(X)
import joblib
joblib.dump(lda, '/kaggle/working/lda')
# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda.score(X))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda.perplexity(X))

# See model parameters
print(lda.get_params())
for i in range(12,16):
    from  sklearn.decomposition import LatentDirichletAllocation
    lda2 = LatentDirichletAllocation(n_components=i, random_state=0)
    lda2.fit(X)
    print("Log Likelihood: ", lda2.score(X))

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", lda2.perplexity(X))

    # See model parameters
    print(lda2.get_params())
# Define Search Param 7 is by elbow method remaining are randomly added
search_params = {'n_components': [4, 5, 7, 13,15]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(X)
GridSearchCV(cv=None, error_score='raise',
       estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7, learning_method=None,
             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
             mean_change_tol=0.001, n_components=7, n_jobs=1,
             n_topics=None, perp_tol=0.1, random_state=None,
             topic_word_prior=None, total_samples=1000000.0, verbose=0),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'n_topics': [ 4,5 ,7, 13,15]},
       pre_dispatch='2*2', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
# Show top n keywords for each topic
import pandas as pd
def show_topics(n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(n_words=10)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords
# Create Document - Topic Matrix
lda_output = lda.transform(X)

# column names
ntopics = 10
topicnames = ["Topic" + str(i) for i in range(ntopics)]

# index names
docnames = ["Doc" + str(i) for i in range(len(data))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# Apply Style
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
df_document_topics
all_texts[1][:501]
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, X, vectorizer, mds='tsne')
panel
D, y = kmeans.index.search(X_test.astype("float32"), 1)
from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=5)
X_embedded = tsne.fit_transform(X_test)
print(y.shape)
y = y[:,0]
from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", len(set(y)))

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)
plt.title("t-SNE Covid-19 Articles - Clustered(K-Means) - Tf-idf with Plain Text")
# plt.savefig("plots/t-sne_covid19_label_TFID.png")
plt.show()

!apt-get install libomp-dev -y
import faiss   
# , MetricType  = 'METRIC_L2'
d = 4096
index = faiss.IndexFlat(d,faiss.METRIC_JensenShannon)   # build the index
print(faiss.METRIC_Linf)
print(index.is_trained)

index.add(X.toarray().astype("float32"))                  # add vectors to the index
print(index.ntotal)

doc_topic_distance = pd.DataFrame(lda.transform(X))
doc_topic_distance.to_csv('doc_topic_distance.csv', index=False)
joblib.dump(doc_topic_distance,'doc_topic_distance')
print(doc_topic_distance)
is_covid19_article = papers.text.str.contains('COVID-19|SARS-CoV-2|2019-nCov|SARS Coronavirus 2|2019 Novel Coronavirus')
# temp = doc_topic_distance[is_covid19_article]
print(papers.text[0])
from scipy.spatial.distance import jensenshannon
def get_k_nearest_document_hybrid_method(doc_dist, k=5, lower=1950, upper=2020, only_covid19=False, get_dist=False):
    '''
    doc_dist: topic distribution (sums to 1) of one article
    
    Returns the index of the k nearest articles (as by Jensen–Shannon divergence in topic space). 
    '''
    
#     relevant_time = papers.publish_year.between(lower, upper) relevant_time & relevant_time is_covid19_article
    
    if only_covid19:
        temp = doc_topic_distance
        
    else:
        temp = doc_topic_distance
         
    distances = temp.apply(lambda x: jensenshannon(x, doc_dist), axis=1)
    k_nearest = distances[distances != 0].nsmallest(n=k).index
    
    if get_dist:
        k_distances = distances[distances != 0].nsmallest(n=k)
        return k_nearest, k_distances
    else:
        return k_nearest
papers.head()
task1 = ["Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.",
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
"Role of the environment in transmission"]
task6 = ["Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019", 
"Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight",
"Efforts to support sustained education, access, and capacity building in the area of ethics",
"Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.",
"Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)",
"Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.",
"Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media."]


task2 = ['Data on potential risks factors',
'Smoking, pre-existing pulmonary disease',
'Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities',
'Neonates and pregnant women',
'Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.',
'Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors', 
'Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups',
'Susceptibility of populations',
'Public health mitigation measures that could be effective for control']

task3 = ['Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.',
'Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.',
'Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.',
'Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.',
'Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.',
'Experimental infections to test host range for this pathogen.',
'Animal host(s) and any evidence of continued spill-over to humans',
'Socioeconomic and behavioral risk factors for this spill-over',
'Sustainable risk reduction strategies']

task4 = ["Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.",
"Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.",
"Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.",
"Methods to control the spread in communities, barriers to compliance and how these vary among different populations..",
"Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.",
"Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.",
"Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).",
"Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay."]


task5 = ["Effectiveness of drugs being developed and tried to treat COVID-19 patients. Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.",
"Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.",
"Exploration of use of best animal models and their predictive value for a human vaccine.",
"Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.",
"Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.",
"Efforts targeted at a universal coronavirus vaccine.",
"Efforts to develop animal models and standardize challenge studies",
"Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers",
"Approaches to evaluate risk for enhanced disease after vaccination",
"Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]"]





task7 = ["How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).",
"Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.",
"Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.",
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
"One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors."]

task8 = ["Resources to support skilled nursing facilities and long term care facilities.",
"Mobilization of surge medical staff to address shortages in overwhelmed communities",
"Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies",
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
"Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)"]


task9 = ["Methods for coordinating data-gathering with standardized nomenclature.",
"Sharing response information among planners, providers, and others.",
"Understanding and mitigating barriers to information-sharing.",
"How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).",
"Integration of federal/state/local public health surveillance systems.",
"Value of investments in baseline public health response infrastructure preparedness",
"Modes of communicating with target high-risk populations (elderly, health care workers).",
"Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too).",
"Communication that indicates potential risk of disease to all population groups.",
"Misunderstanding around containment and mitigation.",
"Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.",
"Measures to reach marginalized and disadvantaged populations.",
"Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities.",
"Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.",
"Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care"]
tasks={'What is known about transmission, incubation, and environmental stability?': task1,
       'What do we know about COVID-19 risk factors?': task2, 
       'What do we know about virus genetics, origin, and evolution?': task3, 
       'What do we know about non-pharmaceutical interventions?': task4,
       'What do we know about vaccines and therapeutics?': task5, 
       'What has been published about ethical and social science considerations?': task6, 
       'What do we know about diagnostics and surveillance?': task7,
       'What has been published about medical care?': task8, 
       'What has been published about information sharing and inter-sectoral collaboration?': task9}
from IPython.core.display import HTML
def relevant_articles(tasks, k=3, lower=1950, upper=2020, only_covid19=False):
    tasks = [tasks] if type(tasks) is str else tasks 
    
    tasks_vectorized = vectorizer.transform(tasks)
    tasks_topic_dist = pd.DataFrame(lda.transform(tasks_vectorized))

    for index, bullet in enumerate(tasks):
        print(bullet)
        recommended = get_k_nearest_document_hybrid_method(tasks_topic_dist.iloc[index], k, lower, upper, only_covid19)
#         print(recommended)
        recommended = papers.iloc[recommended]
        print(recommended)
#         display_papers(recommended)
#         h = '<br/>'.join(['<a href="' + str(l) + '" target="_blank">'+ str(n) + '</a>' for l, n in recommended[['title','abstract']].values])
       
#         display(HTML(h))
papers.iloc[1249].values
def display_papers(dataframe):
    """
    Displays all the papers in a 
    data subset obtained like 
    bio_pulmonary or bio_smoking
    
    Parameters
    ----------
    dataframe : The dataframe
    
    Returns
    -------
    Prints all paper titles and paper ids
    in a given dataframe
    """
    papers = ";".join(str(comment) for comment in dataframe["title"])
    paper_ids = ";".join(str(comment) for comment in dataframe["paper_id"])
    papers = papers.split(";")
    paper_ids = paper_ids.split(";")
    for p,p_id in zip(papers, paper_ids):
        print("-> ",p," ( Paper ID :", p_id,")")
    print("----------")
           
for x in papers["title"]:
    print(x)
relevant_articles(task1, 5, only_covid19=True)
relevant_articles(task2, 5, only_covid19=True)
relevant_articles(task3, 5, only_covid19=True)
relevant_articles(task4, 5, only_covid19=True)
relevant_articles(task5, 5, only_covid19=True)
relevant_articles(task6, 5, only_covid19=True)
relevant_articles(task7, 5, only_covid19=True)
relevant_articles(task8, 5, only_covid19=True)
relevant_articles(task9, 5, only_covid19=True)