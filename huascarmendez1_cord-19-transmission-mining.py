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



terms_group_id = "transmission"



terms1 = [

    "Range of incubation periods",

    "Incubation periods",

    "Range of incubation",

    "Contagion time"

]



terms2 = [

    "Viral shedding",

    "Prevalence of asymptomatic transmission",

    "Prevalence of asymptomatic shedding"

]



terms3 = ["Seasonality of transmission"]



terms4 = [

    "Charge distribution",

    "Viral shedding",

    "Prevalence of asymptomatic shedding",

    "Adhesion to hydrophilic/phobic surfaces",

    "Adhesion to surfaces"

]



terms5 = [

    "Persistence and stability",

    "Persistence on substrates",

    "Persistence",

    "Stability",

    "Substrates",

    "Sources",

    "Nasal discharge",

    "Sputum",

    "Urine",

    "Fecal matter",

    "Blood"

]

terms6 = [

    "Environmental survival",

    "Persistence of virus on surfaces",

    "Persistence on surfaces"

]



terms7 = [

    "Natural history of the virus",

    "Natural history",

    "Viral shedding"

]



terms8 = [

    "Implementation of diagnostics",

    "Implementation of products",

    "Improve clinical processes",

    "Diagnostics",

    "Products",

    "Clinical processes"

]



terms9 = [

    "Disease models",

    "Animal models for infection",

    "Animal models",

    "Infection models",

    "Models for disease",

    "Models for transmission"

]



terms10 = [

    "Monitoring",

    "Tools to monitor phenotypic change",

    "Phenotypic change",

    "Tools to monitor potential adaptation",

    "Potential adaptation"

]



terms11 = [

    "Immunity",

    "Immune response and immunity",

    "Immune response"

]



terms12 = [

    "Effectiveness of movement control strategies",

    "Movement control",

    "Prevent secondary transmission",

    "Control strategies"

]



terms13 = [

    "Protective equipment",

    "Ppe",

    "Personal protective equipment",

    "Effectiveness of personal protective equipment",

    "Personal equipment"

]



terms14 = [

    "Role of the environment",

    "Role of the environment in transmission",

    "Environment in transmission",

]



terms = terms1 + terms2 + terms3 + terms4 + terms5 + terms6

terms += terms7 + terms8 + terms9 + terms10 + terms11 + terms12

terms += terms13 + terms14
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



ax = cnt_s.plot(kind='barh', figsize=(12,30), 

                legend=False, color="coral", 

                fontsize=16)

ax.set_alpha(0.8)

ax.set_title("What is Known About Transmission, Incubation, and Environmental Stability?"

             ,fontsize=18)

ax.set_xlabel("Term Appearances", fontsize=16);

ax.set_ylabel("Terms", fontsize=14);

ax.set_xticks([0,200,400,600,800,1000,1200,1400,

               1600,1800,2000,2200,2400])



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
findings_sta = findings.groupby(["term", "paper_id"]).size().reset_index(name="count")

findings_sta = findings_sta.sort_values(by=['term','count'], ascending=False)
# Helper



def get_doc_text(paper_id):

    doc = df.loc[df["paper_id"]==paper_id].iloc[0]

    return doc["doc_text"]
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
td1 = "Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery."

text = get_question_answer(terms1)

ta1 = get_text_area(text)



td2 = "Prevalence of asymptomatic shedding and transmission (e.g., particularly children)."

text = get_question_answer(terms2)

ta2 = get_text_area(text)



td3 = "Seasonality of transmission."

text = get_question_answer(terms3)

ta3 = get_text_area(text)



td4 = "Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding)."

text = get_question_answer(terms4)

ta4 = get_text_area(text)



td5 = "Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood)."

text = get_question_answer(terms5)

ta5 = get_text_area(text)



td6 = "Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic)."

text = get_question_answer(terms6)

ta6 = get_text_area(text)



td7 = "Natural history of the virus and shedding of it from an infected person"

text = get_question_answer(terms7)

ta7 = get_text_area(text)



td8 = "Implementation of diagnostics and products to improve clinical processes"

text = get_question_answer(terms8)

ta8 = get_text_area(text)



td9 = "Disease models, including animal models for infection, disease and transmission"

text = get_question_answer(terms9)

ta9 = get_text_area(text)



td10 = "Tools and studies to monitor phenotypic change and potential adaptation of the virus"

text = get_question_answer(terms10)

ta10 = get_text_area(text)



td11 = "Immune response and immunity"

text = get_question_answer(terms11)

ta11 = get_text_area(text)



td12 = "Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings"

text = get_question_answer(terms12)

ta12 = get_text_area(text)



td13 = "Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings"

text = get_question_answer(terms13)

ta13 = get_text_area(text)



td14 = "Role of the environment in transmission"

text = get_question_answer(terms14)

ta14 = get_text_area(text)
ac1_tas = [ta1,ta2,ta3,ta4,ta5,ta6,ta7,ta8,

           ta9,ta10,ta11,ta12,ta13,ta14]

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
ac1