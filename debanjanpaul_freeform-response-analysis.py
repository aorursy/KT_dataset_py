import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt; plt.rcdefaults()

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation







schema = pd.read_csv("../input/schema.csv")

data_freeform = pd.read_csv("../input/freeformResponses.csv", dtype = object)

data_freeform = data_freeform.dropna(axis = 1, how='all')
schema_freeform = schema[schema['Column'].str.contains("FreeForm")]
schema_freeform_asked_all = schema_freeform[schema_freeform['Asked'] == 'All']

schema_freeform_asked_codingworker = schema_freeform[schema_freeform['Asked'] == 'CodingWorker']

schema_freeform_asked_codingworkernc = schema_freeform[schema_freeform['Asked'] == 'CodingWorker-NC']

schema_freeform_asked_learners = schema_freeform[schema_freeform['Asked'] == 'Learners']

schema_freeform_asked_nonswitcher = schema_freeform[schema_freeform['Asked'] == 'Non-switcher']

schema_freeform_asked_onlinelearners = schema_freeform[schema_freeform['Asked'] == 'OnlineLearners']

schema_freeform_asked_worker = schema_freeform[schema_freeform['Asked'].isin(['Worker1','Worker'])]
def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        #print("Topic #%d:" % topic_idx)

        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))





def extract_topics(document, n_topics):

    vect = CountVectorizer(stop_words='english',lowercase= True)

    dtm = vect.fit_transform(document.dropna())

    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,

                                    learning_method='online',

                                    learning_offset=50.,

                                    random_state=0)

    lda.fit(dtm)

    print_top_words(lda, vect.get_feature_names(), 3)





def calculate_response(col,lst):

    pct = []

    x = pd.DataFrame(data_freeform[col].dropna().str.lower())

    for word in lst:

        total_response = len(x[col])

        p = (len(x[x[col].str.contains(word)])/total_response)*100

        pct.append(p)

    return pd.DataFrame({'Percentage':pct, col:lst}, columns = [col,'Percentage']), total_response



def draw_chart(df):

    df = df[0].sort_values(by = 'Percentage', ascending = False)

    matplotlib.rcParams['figure.figsize'] = (10.0, 4.0)

    plt.bar(np.arange(len(df.iloc[:,0])), df.iloc[:,1], align='center', alpha=0.5)

    plt.xticks(np.arange(len(df.iloc[:,0])), df.iloc[:,0],rotation='vertical')

    plt.ylabel('Percentage')

    plt.title(list(df)[0])

    plt.show()
schema_freeform_asked_all[['Column','Question']]
extract_topics(data_freeform['GenderFreeForm'], 5)

calculate_response('GenderFreeForm',['helicopter','attack','apache'])
extract_topics(data_freeform['MLToolNextYearFreeForm'], 10)

resp = calculate_response('MLToolNextYearFreeForm',['tensorflow','cuda','javascript','keras','scala','haskell','pytorch','h2o','alteryx','azure','cntk','mxnet','latex','pymc3'])

draw_chart(resp)

#type(resp[0])

#resp[0].sort_values(by='Percentage',ascending = False)
extract_topics(data_freeform['MLMethodNextYearFreeForm'], 20)

resp = calculate_response('MLMethodNextYearFreeForm',['rnn','cnn','natural','gan','reinforcement','gaussian','vision','adversarial','markov','boost',])

draw_chart(resp)
extract_topics(data_freeform['LanguageRecommendationFreeForm'], 20)

resp = calculate_response('LanguageRecommendationFreeForm',['wolfram','julia','sql','vb','python','octave','rust','excel','scala','golang','weka'])

draw_chart(resp)
extract_topics(data_freeform['PublicDatasetsFreeForm'], 20)

#calculate_response('PublicDatasetsFreeForm',['crawl','scrap','collect','reddit','uci','aws','wiki','web'])
extract_topics(data_freeform['PersonalProjectsChallengeFreeForm'], 20)

resp = calculate_response('PersonalProjectsChallengeFreeForm',['clean','dirty','nois','license','missing','domain','availability'])

draw_chart(resp)
extract_topics(data_freeform['LearningPlatformCommunityFreeForm'], 20)

resp = calculate_response('LearningPlatformCommunityFreeForm',['twitter','kdnuggets','qlikview','biostars','opendatascience','facebook','mooc','coursera','vidya','vidhya','analyticsvidhya','reddit','quora','udacity','hackerearth','datatau','datacamp','indico','slack','rbloggers','stackoverflow','quantopian','linkedin','meetups'])

draw_chart(resp)
df = pd.DataFrame()

df['LearningPlatformFreeForm'] = data_freeform['LearningPlatformFreeForm1'].dropna().append([data_freeform['LearningPlatformFreeForm2'].dropna(),data_freeform['LearningPlatformFreeForm3'].dropna()])

extract_topics(df['LearningPlatformFreeForm'], 20)

resp1 = calculate_response('LearningPlatformFreeForm1',['edx','book','hackerrank','meetup','analyticsvidhya','vidhya','linkedin','mooc','reddit','google','quora','journals','github','paper','datacamp','udacity','udemy','kagglenoobs','galvanize','youtube','lynda','dataquest'])

resp2 = calculate_response('LearningPlatformFreeForm2',['edx','book','hackerrank','meetup','analyticsvidhya','vidhya','linkedin','mooc','reddit','google','quora','journals','github','paper','datacamp','udacity','udemy','kagglenoobs','galvanize','youtube','lynda','dataquest'])

resp3 = calculate_response('LearningPlatformFreeForm3',['edx','book','hackerrank','meetup','analyticsvidhya','vidhya','linkedin','mooc','reddit','google','quora','journals','github','paper','datacamp','udacity','udemy','kagglenoobs','galvanize','youtube','lynda','dataquest'])

resp = ((resp1[0]['Percentage']*resp1[1]/100 + resp2[0]['Percentage']*resp2[1]/100 + resp3[0]['Percentage']*resp3[1]/100)/(resp1[1] + resp2[1] +resp3[1]))*100

resp = pd.DataFrame(resp1[0]['LearningPlatformFreeForm1']).assign(Percentage = resp)

draw_chart((resp,resp1[1]+resp2[1]+resp3[1]))
extract_topics(data_freeform['BlogsPodcastsNewslettersFreeForm'], 20)

resp = calculate_response('BlogsPodcastsNewslettersFreeForm',['datacamp','udacity','superdatascience','datascienceweekly','data science weekly','analytics vidhya','kaggle','mlwave','hackernoon','indiseai','coursera','udemy'])

draw_chart(resp)
extract_topics(data_freeform['ImpactfulAlgorithmFreeForm'], 20)

resp = calculate_response('ImpactfulAlgorithmFreeForm',['nda','facial','bayes','decision tree','cluster','apriori','neural','xgboost','gbm','logistic','linear','regression','cnn','knn','randomforest','word2vec','segmentation','svm'])

draw_chart(resp)
extract_topics(data_freeform['InterestingProblemFreeForm'], 20)
extract_topics(data_freeform['PastJobTitlesFreeForm'], 20)

resp = calculate_response('PastJobTitlesFreeForm',['cto','ceo','accountant','technician','consultant','student','executive','tester','administrator','instructor','admin','reasearcher','linguist','mathematician','econnomist','architect','manager','teacher','actuary','tutor','officer','chief','trader','professor','scientist','designer','bioinformatician','founder','journalist'])

draw_chart(resp)
extract_topics(data_freeform['FirstTrainingFreeForm'], 20)

resp = calculate_response('FirstTrainingFreeForm',['contest','inbuilt','diploma','self','project','training','udacity','master degree','course','internships','kaggle','book','youtube','knime','bootcamp','certification'])

draw_chart(resp)
extract_topics(data_freeform['MLSkillsFreeForm'], 20)

resp = calculate_response('MLSkillsFreeForm',['genetic algorithms','hypothesis testing','text mining','predictive modelling','dimension reduction','machine learning','deep learning','time series','regression','statistics','chemometrics','financial engineering','information retrieval','robotics','fuzzy','image processing','signal processing','face recognition','speaker recognition'])

draw_chart(resp)
extract_topics(data_freeform['MLTechniquesFreeform'], 20)

#calculate_response('MLTechniquesFreeform',[])
schema_freeform_asked_learners[['Column','Question']]
df = pd.DataFrame()

df['JobSkillImportanceOtherSelect'] = data_freeform['JobSkillImportanceOtherSelect1FreeForm'].dropna().append([data_freeform['JobSkillImportanceOtherSelect2FreeForm'].dropna(),data_freeform['JobSkillImportanceOtherSelect3FreeForm'].dropna()])

extract_topics(df['JobSkillImportanceOtherSelect'], 20)
extract_topics(data_freeform['HardwarePersonalProjectsFreeForm'], 20)

resp=calculate_response('HardwarePersonalProjectsFreeForm',['laptop','cloud','server','desktop','pc'])

draw_chart(resp)
resp=calculate_response('HardwarePersonalProjectsFreeForm',['hp','dell','lenovo','macbook'])

draw_chart(resp)
resp=calculate_response('HardwarePersonalProjectsFreeForm',['windows','linux'])

draw_chart(resp)
extract_topics(data_freeform['ProveKnowledgeFreeForm'], 20)
extract_topics(data_freeform['JobSearchResourceFreeForm'], 20)

resp=calculate_response('JobSearchResourceFreeForm',['linkedin','network','meetup','analyticsvidya','fb','stackoverfow'])

draw_chart(resp)