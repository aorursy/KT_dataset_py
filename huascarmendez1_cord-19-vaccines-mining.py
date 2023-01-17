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



terms_group_id = "vaccines"



terms1 = [

    "Vaccines and therapeutics",

    "Efforts of vaccines",

    "Vaccines",

    "Efforts of therapeutics",

    "Therapeutics",

    "Therapeutics being developed",

    "Therapeutics being tried",

    "Effectiveness of drugs",

    "Drugs",

    "Drugs being developed",

    "Drugs being tried",

    "Clinical trials",

    "Bench trials",

    "Investigate less common viral inhibitors",

    "Naproxen",

    "Clarithromycin",

    "Minocyclinethat",

    "Exert effects on viral replication",

    "Exert effects"

]



terms2 = [

    "Potential complication",

    "Antibody-dependent enhancement",

    "Vaccine recipients",

    "Ade"

]



terms3 = [

    "Best animal models",

    "Predictive value",

    "Human vaccine"

]



terms4 = [

    "Capabilities to discover a therapeutic",

    "Not vaccine",

    "Clinical effectiveness studies",

    "Discover therapeutics",

    "include antiviral agents",

    "Antiviral agents"

]



terms5 = [

    "Alternative models",

    "Alternative",

    "Aid decision makers",

    "Prioritize and distribute scarce",

    "Prioritize scarce",

    "Distribute scarce",

    "Scarse",

    "Newly proven therapeutics",

    "Proven",

    "Production capacity",

    "Capacity",

    "Equitable distribution",

    "Timely distribution"

]



terms6 = [

    "Universal coronavirus vaccine",

    "Universal vaccine",

    "Coronavirus vaccine"

]



terms7 = [

    "Efforts to develop animal models",

    "Standardize challenge studies",

    "Challenge studies"

]



terms8 = [

    "Efforts to develop prophylaxis",

    "Develop prophylaxis",

    "Prophylaxis",

    "Healthcare workers"

]



terms9 = [

    "Approaches to evaluate risk",

    "Risk after vaccination",

    "After vaccine","After vaccination"

]



terms10 = [

    "Assays to evaluate vaccine",

    "Immune response",

    "Process development"

]



terms = terms1 + terms2 + terms3 + terms4 + terms5 

terms += terms6 + terms7 + terms8 + terms9 + terms10
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



ax = cnt_s.plot(kind='barh', figsize=(12,25), 

                legend=False, color="coral", 

                fontsize=16)

ax.set_alpha(0.8)

ax.set_title("What is Known About Vaccines and Therapeutics?",

             fontsize=18)

ax.set_xlabel("Term Appearances", fontsize=16);

ax.set_ylabel("Terms", fontsize=14);

ax.set_xticks([0,200,400,600,800,1000,1200,1400,1600,1800,2000])



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
t1d = "Effectiveness of drugs being developed and tried to treat COVID-19 patients. Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication."

text = get_question_answer(terms1)

ta1 = get_text_area(text)



t2d = "Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients."

text = get_question_answer(terms2)

ta2 = get_text_area(text)



t3d = "Exploration of use of best animal models and their predictive value for a human vaccine."

text = get_question_answer(terms3)

ta3 = get_text_area(text)



t4d = "Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents."

text = get_question_answer(terms4)

ta4 = get_text_area(text)



t5d = "Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need."

text = get_question_answer(terms5)

ta5 = get_text_area(text)

    

t6d = "Efforts targeted at a universal coronavirus vaccine."

text = get_question_answer(terms6)

ta6 = get_text_area(text)



t7d = "Efforts to develop animal models and standardize challenge studies."

text = get_question_answer(terms7)

ta7 = get_text_area(text)



t8d = "Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers."

text = get_question_answer(terms8)

ta8 = get_text_area(text)



t9d = "Approaches to evaluate risk for enhanced disease after vaccination."

text = get_question_answer(terms9)

ta9 = get_text_area(text)

        

t10d = "Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]."

text = get_question_answer(terms10)

ta10 = get_text_area(text)
ac1_tas = [ta1,ta2,ta3,ta4,ta5,ta6,ta7,ta8,ta9,ta10]

ac1 = widgets.Accordion(children=ac1_tas)

ac1.set_title(0, t1d)

ac1.set_title(1, t2d)

ac1.set_title(2, t3d)

ac1.set_title(3, t4d)

ac1.set_title(4, t5d)

ac1.set_title(5, t6d)

ac1.set_title(6, t7d)

ac1.set_title(7, t8d)

ac1.set_title(8, t9d)

ac1.set_title(9, t10d)
ac1