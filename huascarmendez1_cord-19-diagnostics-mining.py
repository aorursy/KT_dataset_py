import os, glob, pandas as pd
# Paths



input_dir = os.path.abspath('/kaggle/input/')

articles_dir = input_dir + '/cord19csv/'
%%time



li_df = []



for jt in ['pdf','pmc']:

    path = f'{articles_dir}/{jt}*.csv'

    files = glob.glob(path)

    

    for file in files:

        if jt == "pdf":            

            df_pdf = pd.read_csv(file, index_col=None, header=0)

            li_df.append(df_pdf)

        else:

            df_pmc = pd.read_csv(file, index_col=None, header=0)        

            li_df.append(df_pmc)



# Combine all papers dataframes in one

df = pd.concat(li_df, axis=0, ignore_index=True, sort=False)
df.shape
df.head()
# Drop duplicated documents by paper_id

df.drop_duplicates(subset="paper_id", keep='first', inplace=True)
# Drop duplicated documents by text

df.drop_duplicates(subset="doc_text", keep='first', inplace=True)

df.shape
# Create the lists of key terms



terms_group_id = "diagnostics"



terms1 = [

    'diagnostics and surveillance','diagnostics','surveillance',

    'systematic, holistic approach to diagnostics',

    'systematic diagnostics', 'holistic diagnostics',

    'approach to diagnostics','systematic','holistic',

    'public health surveillance','health surveillance',

    'surveillance perspective','predict clinical outcomes',

    'immediate policy recommendations','mitigation measures',

    'policy recommendations','denominators for testing',

    'mechanism for rapidly sharing that information',

    'information sharing','sampling methods',

    'determine asymptomatic disease','asymptomatic disease',

    'use of serosurveys','convalescent samples','early detection',

    'screening of neutralizing antibodies','elisas','elisa',

    'screening','neutralizing antibodies'

]



terms2 = [

    'diagnostic platforms',

    'surveillance platforms'

]



terms3 = [

    'recruitment','support',

    'coordination of local expertise','private—commercial',

    'non-profit','academic','legal','ethical','communications',

    'operational issues'

]



terms4 = [

    'national guidance','private sector',

    'guidelines','best practices to states',

    'leverage universities','private laboratories',

    'testing purposes','health officials'

]



terms5 = [

    'point-of-care test','rapid influenza test',

    'rapid bed-side tests','tradeoffs between speed',

    'accessibility','accuracy'

]



terms6 = [

    'rapid design',

    'execution of targeted surveillance experiments',

    'targeted surveillance experiments',' potential testers',

    'surveillance experiments','calling for all potential testers',

    'pcr','specific entity','longitudinal samples',

    'aid in collecting longitudinal samples','instruments',

    'ad hoc local interventions','local interventions',

    'ad hoc interventions'

]



terms7 = [

    'assay development issues',

    'separation of assay development issues from instruments',

    'role of the private sector','help quickly migrate assays'

]



terms8 = [

    'track the evolution of the virus','genetic drift','mutations',

    'surveillance/detection schemes','surveillance schemes',

    'detection schemes'

]



terms9 = [

    'latency issues','viral load',

    'biological sampling','environmental sampling'

]



terms10 = [

    'host response markers','cytokines','early disease',

    'predict severe disease progression','disease progression'

]



terms11 = [

    'policies and protocols for screening and testing',

    'policies for screening and testing','screening','testing',

    'protocols for screening and testing','effects on supplies'

]



terms12 = [

    'policies to mitigate the effects on supplies','mass testing',

    'swabs','reagents'

]



terms13 = [

    'technology roadmap'

]



terms14 = [

    'barriers to developing and scaling up new diagnostic tests',

    'developing new diagnostic tests','diagnostic tests',

    'scaling up new diagnostic tests','market forces',

    'coalition and accelerator models','accelerator models',

    'coalition for epidemic preparedness innovations',

    'critical funding','funding for diagnostics',

    'opportunities for a streamlined regulatory environment',

    'regulatory environment'

]



terms15 = [

    'new platforms','new technology',

    'crispr'

]



terms16 = [

    'coupling genomics and diagnostic testing on a large scale',

    'genomics testing','large scale',

    'diagnostic testing'

]



terms17 = [

    'rapid sequencing',

    'bioinformatics','particular variant'

]



terms18 = [

    'advanced analytics',

    'unknown pathogens','explore capabilities',

    'naturally-occurring pathogens','intentional'

]



terms19 = [

    'future spillover','ongoing exposure','future pathogens',

    'evolutionary hosts','bats','heavily trafficked',

    'farmed wildlife','domestic food','companion species'

    'occupational risk factors']



terms = terms1 + terms2 + terms3 + terms4 + terms5 

terms += terms6 + terms7 + terms8 + terms9 + terms10 

terms += terms11 + terms12 + terms13 + terms14 + terms15

terms += terms16 + terms17 + terms18 + terms19
import spacy



# Perform NLP operations on GPU, if available.

spacy.prefer_gpu()



# Load Spacy english model

nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])

nlp.max_length = 5000000
# Create matcher and patterns



from spacy.matcher import PhraseMatcher



# Create a Matcher to case insensitive text matching

matcher = PhraseMatcher(nlp.vocab, attr='LEMMA')    



# Create patterns from terms

patterns = [nlp(d) for d in terms]

matcher.add(terms_group_id, None, *patterns)
# Defines the matcher



def cord_19_matcher(sample_pct):   

    # variables to test: test_limt is the total of docs to test; 

    # 0 = test off

    

    test_limit = 0

    counter = 0



    docs = df.sample(frac = sample_pct/100) if sample_pct < 100 else df

    tdocs = str(len(docs))



    print(f"{tdocs} documents to proccess...")

        

    # Maximun allowed length of string text document

    max_tlen = 100000



    # initialize array and total found variables

    findings_arr = []



    # loop all articles to match terms

    for idx, row in docs.iterrows():

        try:

            paper_id = row['paper_id']

            doc_text = row["doc_text"]            

            

            doc = nlp(doc_text)



            # get the matches

            matches = matcher(doc)



            # process all matches found in text

            if matches:

                for m in matches:

                    m_id, start, end = m[0],m[1],m[2]

                    term_group = nlp.vocab.strings[m_id]

                    term = doc[start:end].text



                    # put finding into json object

                    finding = {

                        "paper_id": paper_id,

                        "term_group": term_group,

                        "term": term

                    }



                    # append finding to findings array

                    findings_arr.append(finding)                



            counter += 1

            if counter % 100 == 0:

                print(f"{counter} documents proccessed")



            # breake loop if test control present

            if test_limit > 0:            

                if counter == test_limit:

                    print(test_limit, "sample count reached")

                    break



        except BaseException as e:

            print("Oops!  Error occurred in document loop.")

            print(str(e))

            print("Continuing...")

            continue

    

    return findings_arr
%%time



# Set sample parameter = % of papers to proccess

sample_pct = 100

#sample_pct = 1.2

#sample_pct = 10



findings_arr = cord_19_matcher(sample_pct)



tfound = len(findings_arr)

print(tfound, "matches found\n")
# Put findings array into a dataframe



findings = pd.DataFrame(findings_arr)



# exclude the following terms originally taken in account

#exc = ['term1','term2','term3']

#findings.where(~findings.term.isin(exc), inplace = True)
findings.info()
findings.head()
# Capitalize each term in findings

findings["term"] = findings["term"].str.capitalize()
findings['count'] = ''

cnt = findings.groupby('term').count()[['count']]

cnt_s = cnt.sort_values(by='count', ascending=False).copy()
# Show the bar chart



ax = cnt_s.plot(kind='barh', figsize=(12,50), 

                legend=False, color="coral", 

                fontsize=16)

ax.set_alpha(0.8)

ax.set_title("What do we know about diagnostics and surveillance?",

             fontsize=18)

ax.set_xlabel("Term Appearances", fontsize=16);

ax.set_ylabel("Terms", fontsize=14);

ax.set_xticks([0,200,400,600,800,1000,1200,1400,1600,

               1800,2000])



# Create a list to collect the plt.patches data

totals = []



# Fill totals list

for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



# Set bar labels using the list

for i in ax.patches:

    c = i.get_width()

    cnt = f'{c:,} '

    pct = str(round((c/total)*100, 2)) + '%'

    pct_f = "(" + pct + ")"

    ax.text(c+.3, i.get_y()+.4, cnt + pct_f, 

            fontsize=14, color='dimgrey')



# Invert graph 

ax.invert_yaxis()
from wordcloud import WordCloud, ImageColorGenerator

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np



# Fill the list of words to show

term_values = ""

for term in findings['term']:

    term_val = str(term).title()

    term_val = term_val.replace(' ','_')

    term_val = term_val.replace('-','_')

    term_values += term_val + ' '



# Generates the wordcloud object

wordcloud = WordCloud(background_color="white",

                      collocations=False).generate(term_values)



# Display the generated image

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.figure(figsize=((10,8)))

plt.show()
# Helper



def get_doc_text(paper_id):

    doc = df.loc[df["paper_id"]==paper_id].iloc[0]

    return doc["doc_text"]
findings_sta = findings.groupby(["term", "paper_id"]).size().reset_index(name="count")

findings_sta = findings_sta.sort_values(by=['term','count'], ascending=False)
answers = []



for term in terms:    

    term = term.capitalize()

    try:

        f = findings_sta[findings_sta["term"]==term]

        f = f.sort_values("count",ascending=False)

        for fc in f.iterrows():           

            paper_id = fc[1]["paper_id"]                        

            doc_text = get_doc_text(paper_id)

            

            answer = {

                "aspect": terms_group_id,

                "factor": term,

                "paper_id": paper_id,

                "doc_text": str(doc_text)

            }



            answers.append(answer)

            

            break

        

    except BaseException as e:

        print(str(e))

        continue



len(answers)
import ipywidgets as widgets

from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider
item_layout = Layout(

    display='flex',

    flex_flow='row',

    justify_content='space-between',

    width= '100%',

    height= '200px'

)
# Helpers



def get_text_area(text):

    ta = widgets.Textarea(

        value=str(text),

        placeholder='',

        description='',

        layout=item_layout,

        disabled=True

    )

    return ta



import json



def get_answer_text(factor):

    try:

        factor = factor.capitalize()

        ans = next(x for x in answers if x["factor"] == factor)

        ans = json.dumps(ans["doc_text"]).strip("'").strip('"')

        ans = ans.replace('\\n', '\n\n')

        return ans

    except BaseException:

        return ""

    

def get_question_answer(t_params):

    full_text = ''

    for t_param in t_params:

        try:

            doc_text = get_answer_text(t_param)

            if not doc_text in full_text:

                if len(full_text) > 0:

                    full_text += "\n\n"                

                full_text += doc_text

        except BaseException:

            continue

    

    return full_text
td1 = 'How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).'

text = get_question_answer(terms1)

ta1 = get_text_area(text)



td2 = 'Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.'

text = get_question_answer(terms2)

ta2 = get_text_area(text)



td3 = 'Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.'

text = get_question_answer(terms3)

ta3 = get_text_area(text)



td4 = 'National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).'

text = get_question_answer(terms4)

ta4 = get_text_area(text)



td5 = 'Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.'

text = get_question_answer(terms5)

ta5 = get_text_area(text)



td6 = 'Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).'

text = get_question_answer(terms6)

ta6 = get_text_area(text)



td7 = 'Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.'

text = get_question_answer(terms7)

ta7 = get_text_area(text)



td8 = 'Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.'

text = get_question_answer(terms8)

ta8 = get_text_area(text)



td9 = 'Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.'

text = get_question_answer(terms9)

ta9 = get_text_area(text)



td10 = 'Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.'

text = get_question_answer(terms10)

ta10 = get_text_area(text)



td11 = 'Policies and protocols for screening and testing.'

text = get_question_answer(terms11)

ta11 = get_text_area(text)



td12 = 'Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.'

text = get_question_answer(terms12)

ta12 = get_text_area(text)



td13 = 'Technology roadmap for diagnostics.'

text = get_question_answer(terms13)

ta13 = get_text_area(text)



td14 = 'Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.'

text = get_question_answer(terms14)

ta14 = get_text_area(text)



td15 = 'New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.'

text = get_question_answer(terms15)

ta15 = get_text_area(text)



td16 = 'Coupling genomics and diagnostic testing on a large scale.'

text = get_question_answer(terms16)

ta16 = get_text_area(text)



td17 = 'Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.'

text = get_question_answer(terms17)

ta17 = get_text_area(text)



td18 = 'Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.'

text = get_question_answer(terms18)

ta18 = get_text_area(text)



td19 = 'One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.'

text = get_question_answer(terms19)

ta19 = get_text_area(text)
ac1_tas = [ta1,ta2,ta3,ta4,ta5,ta6,ta7,ta8,ta9,ta10,

          ta11,ta12,ta13,ta14,ta15,ta16,ta17,ta18,ta19]

ac1 = widgets.Accordion(children=ac1_tas)

ac1.set_title(0, td1)

ac1.set_title(1, td2)

ac1.set_title(2, td3)

ac1.set_title(3, td4)

ac1.set_title(4, td5)

ac1.set_title(5, td6)

ac1.set_title(6, td7)

ac1.set_title(7, td8)

ac1.set_title(8, td9)

ac1.set_title(9, td10)

ac1.set_title(10, td11)

ac1.set_title(11, td12)

ac1.set_title(12, td13)

ac1.set_title(13, td14)

ac1.set_title(14, td15)

ac1.set_title(15, td16)

ac1.set_title(16, td17)

ac1.set_title(17, td18)

ac1.set_title(18, td19)
ac1