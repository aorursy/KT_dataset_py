# INPUT AREA, Provide -1 if you do not want to process data for a module 

# A module is one of Biorxiv, Comm, Non comm and Custom License. Please read Instruction of Usage above.



SAMPLE_SIZE_BIORXIV =  2000

SAMPLE_SIZE_COMM =     2500

SAMPLE_SIZE_NON_COMM =  2500

SAMPLE_SIZE_CUSTOM_LICENSE = 3000
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import json

from pprint import pprint

from copy import deepcopy

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

import spacy

from spacy.matcher import PhraseMatcher

from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')

import torch

import scipy.spatial

import time

import gc

import plotly.express as px

from IPython.core.display import display, HTML



RUN_MODE =   "SUBSET" #"ALL" 

RUN_SAMPLES = 100

BIORXIV = "biorxiv"

COMM = "comm"

NON_COMM = "non_comm"

CUSTOM_LICENSE = "custom_license"



FILE_BIORXIV_ARTICLES_INFO             = "biorxiv_article_info.txt"

FILE_COMM_ARTICLES_INFO                = "comm_article_info.txt"

FILE_NON_COMM_ARTICLES_INFO            = "non_comm_article_info.txt"

FILE_CUSTOM_LICENSE_ARTICLES_INFO      = "custom_license_article_info.txt"



lst_url_exclusions = ['//github.com', 'https://doi.org','https://doi.org/10','perpetuity.is', 'https://doi.org/10.1101/2020.03', 'https://doi.org/10.1101/2020.04']





## Functions to generate CSVs from Json files, four types of datasets are available here.

def save_article_info(obj, filename):

    with open(filename, 'a') as the_file:

        the_file.write("# PAPER_ID ----- : "  + obj.paper_id + "\n")

        the_file.write("# TITLE -----------: "  + obj.title + "\n")

        the_file.write("# RELEVANT SENTENCES ----------:")

        the_file.write("\n")

        for item in obj.lst_sentences:

            the_file.write("\n ==>")

            the_file.write("%s " % item)

            the_file.write("\n")

        

        if (len(obj.lst_rapid_assessment_sentences) > 0):

            the_file.write("# ASSESSMENT RELATED SENTENCES ----------:")

            the_file.write("\n")

            for item in obj.lst_rapid_assessment_sentences:

                the_file.write("\n ==>")

                the_file.write("%s " % item)

            the_file.write("\n")

        

        if (len(obj.lst_rapid_design_sentences) > 0):

            the_file.write("# DESIGN RELATED SENTENCES ----------:")

            the_file.write("\n")

            for item in obj.lst_rapid_design_sentences:

                the_file.write("\n ==>")

                the_file.write("%s " % item)

            the_file.write("\n")      

        

        if (len(obj.lst_design_experiments_sentences) > 0):

            the_file.write("# EXPERIMENT RELATED SENTENCES ----------:")

            the_file.write("\n")

            for item in obj.lst_design_experiments_sentences:

                the_file.write("\n ==>")

                the_file.write("%s " % item)

            the_file.write("\n")     

        

        the_file.write("# URL -------------:")

        for item in obj.lst_urls:

            the_file.write("\n ==>")

            the_file.write("%s " % item)

        if (len(obj.lst_urls)==0):

            the_file.write("No urls found.")

        the_file.write("\n")

        author_out = obj.authors

        if (obj.authors.strip() == ""):

            author_out = "NOT_FOUND"

        the_file.write("\n")

        the_file.write("# AUTHORS -----------: "  + obj.authors + "\n")

        the_file.write("# SCORE -----------: "  + str(obj.score) + "\n")

        the_file.write("# =========================================================: "  + "\n")







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

        if (section.strip() != ""):

            body += section.upper()

            body += " : "

        body += text

        body += "."

    

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



def load_files(dirname, SAMPLE_SIZE = 50):

    filenames = os.listdir(dirname)

    lst_orig_count = len(filenames)

    raw_files = []

    if (RUN_MODE == "SUBSET"):

        filenames = filenames[0: SAMPLE_SIZE]

    

    for filename in (filenames):

        filename = dirname + filename

        file = json.load(open(filename, 'rb'))

        raw_files.append(file)

    

    return (raw_files, lst_orig_count)



def generate_clean_df(all_files):

    cleaned_files = []

    

    for file in (all_files):

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

def find_phrases_in_title_npi(doc , span_start = 5 , span_end = 5):

    matcher = Matcher(nlp.vocab)

    pattern1= [{'LOWER': 'non'}, {'LOWER': 'pharmaceutical'}, {'LOWER': 'intervention'}]

    pattern2 = [{'LOWER': 'non'}, {'LOWER': 'pharmaceutical'}, {'LOWER': 'interventions'}]

    pattern3 = [{'LOWER': 'non'}, {'IS_PUNCT': True, 'OP' : '*'} , {'LOWER': 'pharmaceutical'}, {'IS_PUNCT': True, 'OP' : '*'}, {'LOWER': 'interventions'}]

    pattern4 = [{'LOWER': 'non'}, {'IS_PUNCT': True, 'OP' : '*'} , {'LOWER': 'pharmaceutical'}, {'IS_PUNCT': True, 'OP' : '*'}, {'LOWER': 'intervention'}]

    lst_spans = []

    #matcher.add('titlematcher', None, *phrase_patterns)

    matcher.add('titlematcher', None,  pattern1, pattern2, pattern3, pattern4)

    found_matches = matcher(doc)

    find_count = len(found_matches)

    for match_id, start, end in found_matches:

        string_id = nlp.vocab.strings[match_id]

        end = min(end + span_end, len(doc))

        start = max(start - span_start,0)

        span = doc[start:end]

        lst_spans.append(span.text)

    snippets = '| '.join([lst for lst in lst_spans])

    return find_count, snippets



def prepare_dataframe_for_nlp(df, nlp):

    df.fillna('', inplace=True)

    return(df)



def get_sents_from_snippets(lst_snippets, nlpdoc, paper_id):

    """

    Finding full sentences when snippets are passed to this function.

    """

    phrase_patterns = [nlp(text) for text in lst_snippets]

    matcher = PhraseMatcher(nlp.vocab)

    matcher.add('xyz', None, *phrase_patterns)

    sentences = nlpdoc

    res_sentences = []

    for sent in sentences.sents:

        found_matches = matcher(nlp(sent.text))

        find_count = len(found_matches)

        if len(found_matches) > 0:

            res_sentences.append(sent.text)

    res_sentences = list(set(res_sentences))

    return(res_sentences)



def limit_text_size(text):

#     if (len(text) > (10000)):

    text = text[0:40000]

    return(text)



def find_phrases_in_text(doc , phrase_list, span_start = 5 , span_end = 5):

    matcher = PhraseMatcher(nlp.vocab)

    #print(phrase_list)

    lst_spans = []

    phrase_patterns = [nlp(text) for text in phrase_list]

    matcher.add('covidmatcher', None, *phrase_patterns)

    found_matches = matcher(doc)

    find_count = len(found_matches)

            

    for match_id, start, end in found_matches:

        string_id = nlp.vocab.strings[match_id]

        end = min(end + span_end, len(doc) - 1)

        start = max(start - span_start,0)

        span = doc[start:end]

        lst_spans.append(span.text)

        #print("found a match.", span.text)

    snippets = '| '.join([lst for lst in lst_spans])

    ret_list = list(set(lst_spans))

    return(find_count, ret_list)



def generate_data(dir_path, SAMPLE_SIZE = 50):

    _files, count_files_orig = load_files(dir_path, SAMPLE_SIZE)

    df = generate_clean_df(_files)

    return(df, count_files_orig)



def add_lists(lst1, lst2, lst3, lst4):

    lst_final = list(lst1) + list(lst2) + list(lst3) + list(lst4)

    return(lst_final)



def do_scoring_npi(title_find_count

               , text_find_count_in

               , text_find_count_ph

               , text_find_count_non

               , text_find_count_npi):

    if ((text_find_count_in > 0) & (text_find_count_ph > 0) & (text_find_count_non > 0)):

        ret = 30 * title_find_count + 10 * text_find_count_npi + text_find_count_in + text_find_count_non + text_find_count_ph

    else:

        ret = 30 * title_find_count + 10 * text_find_count_npi 

    return(ret)



def process_url(url):

    ret = url

    #print(url in lst_url_exclusions)

    if url in lst_url_exclusions:

        ret = ''

    return(ret)



def is_main_url(d):

    if d.startswith('https://doi.org/'): # Could use /10.1101

        return (True)

    else:

        return (False)

def find_url_in_text(doc):

    main_url = "NOT FOUND"

    lst_urls = []

    matcher = Matcher(nlp.vocab)

    pattern = [{'LIKE_URL': True}]

    matcher.add('url', None,  pattern)

    found_matches = matcher(doc)

    #print(found_matches)

    for match_id, start, end in found_matches:

        url = doc[start:end]

        url = process_url(url.text)

        #print(url)

        if (url != ""):

            lst_urls.append(url)

        if is_main_url(url):

            main_url = url

    return(main_url , list(set(lst_urls)))



def get_summary_row_for_df(processed_articles, count_find, count_fund_infra, count_cost_benefits, module):

    dict_row ={"Module":module, "Processed": processed_articles, "Found": count_find

               , "Found Funding and Infrastructure": count_fund_infra

               , "Count Cost benefits": count_cost_benefits}

    return(dict_row)



def process_a_module(path, SAMPLE_SIZE = 50, MODULE = "provide-module"):

    

    df_data, count_orig_files = generate_data(path, SAMPLE_SIZE)

    df_master = df_data.copy()[["paper_id", "title"]]

    df_data = prepare_dataframe_for_nlp(df_data, nlp)

    df_data['small_text'] = list(map(limit_text_size, (df_data['text'])))

    

    df_data['nlp_title'] = list(map(nlp, (df_data['title'])))

    

    with nlp.disable_pipes("tagger", "parser", "ner"):

        df_data['nlp_snall_text'] = list(map(nlp, (df_data['small_text'])))

    

    

    df_master['title_find_count'], df_master['title_found_snippets'] = zip(*df_data['nlp_title'].apply(lambda title: find_phrases_in_title_npi((title))))

    

    

    phrase_list = [u"intervention"]

    df_master['text_find_count_in'], df_master['text_found_snippets_in'] = zip(*df_data['nlp_snall_text'].apply(lambda nlptext: find_phrases_in_text((nlptext), phrase_list)))





    phrase_list = [u"pharmaceutical"]

    df_master['text_find_count_ph'], df_master['text_found_snippets_ph'] = zip(*df_data['nlp_snall_text'].apply(lambda nlptext: find_phrases_in_text((nlptext), phrase_list)))





    phrase_list = [u"non"]

    df_master['text_find_count_non'], df_master['text_found_snippets_non'] = zip(*df_data['nlp_snall_text'].apply(lambda nlptext: find_phrases_in_text((nlptext), phrase_list)))





    phrase_list = [u"NPI"]

    df_master['text_find_count_npi'], df_master['text_found_snippets_npi'] = zip(*df_data['nlp_snall_text'].apply(lambda nlptext: find_phrases_in_text((nlptext), phrase_list)))

    

    df_master['lst_snippets']  = list(map(add_lists

                                              , (df_master['text_found_snippets_ph'])

                                              , (df_master['text_found_snippets_npi'])

                                              , (df_master['text_found_snippets_non'])

                                              , (df_master['text_found_snippets_in'])

                                             ) )



    df_master["score"] = list(map(do_scoring_npi

                                     , df_master['title_find_count']

                                     , df_master['text_find_count_in']

                                     , df_master['text_find_count_ph']

                                     , df_master['text_find_count_non']

                                     , df_master['text_find_count_npi']

                                     ))

    

    df_master    = df_master.sort_values('score', ascending = False)

    df_master['module'] = MODULE

    df_data['module'] = MODULE

    _ = gc.collect()

    return(df_master, df_data, count_orig_files)





def get_paper_info(paper_id, journal):

    if (journal == BIORXIV):

        df = df_biorxiv[df_biorxiv['paper_id'] == paper_id]

    if (journal == COMM):

        df = df_comm[df_comm['paper_id'] == paper_id]

    if (journal == NON_COMM):

        df = df_non_comm[df_non_comm['paper_id'] == paper_id]

    if (journal == CUSTOM_LICENSE):

        df = df_custom_license[df_custom_license['paper_id'] == paper_id]

    text = df.iloc[0]['text']#[0:5000]

    title = df.iloc[0]['title']

    authors = df.iloc[0]['authors']

    return(text, title, authors)



def print_list(lst, number_to_print = 5, shuffle = True):

    if len(lst) < number_to_print:

        number_to_print = len(lst)

    for i in range(-1*number_to_print, 0):

        print( lst[i])

        

def get_stats_from_articles(lst_articles):

    count_articles = 0

    count_cost_benefits = 0

    count_fund_infra = 0

    lst_cost_benefits = []

    lst_fund_infra = []

    for obj in lst_articles:

        count_articles = count_articles + 1

        if len(obj.lst_cost_benefits_sentences) > 0:

            count_cost_benefits = count_cost_benefits + 1

            lst_cost_benefits.append((obj.title, obj.lst_urls, obj.score))

        if len(obj.lst_funding_infra_sentences) > 0:

            count_fund_infra = count_fund_infra + 1

            lst_fund_infra.append((obj.title, obj.lst_urls, obj.score))

    return(count_articles,  count_cost_benefits, count_fund_infra, lst_cost_benefits)



def create_file(filename):

    with open(filename, 'w') as the_file:

        the_file.close()

        

def write_to_file(filename, Text):

    with open(filename, 'a') as the_file:

        the_file.write(Text)

        the_file.close()

        

def get_nlp_text_for_paper_id(paper_id, module):

    text, x, y = get_paper_info(paper_id, module)

    with nlp.disable_pipes("tagger", "parser", "ner"):

        return(nlp(text))

    

def get_td_string():

    tdstring = '<td style="text-align: left; vertical-align: middle; font-size:1.2em;">'

    return(tdstring)



def get_sentence_tr(sent):

    

    row = get_td_string() + f'{sent}</td></tr>'

    return(row)

    #return(  f'<tr>' + f'<td align = "left">{sent}</td>' + '<td>&nbsp;</td></tr>')

    

def display_article(serial , title, url , sentences, score , lst_other_keywords 

                    , lst_cost_benefits, lst_funding_infra_sentences

                    , lst_all_urls, authors, publish_date, npi_count, paper_id):

    if (publish_date == NOT_FOUND):

        publish_date = "N/A"

    if (url != "NOT FOUND"):

        link_text = f'<a href="{url}" target="_blank">{url}</a>'

    else:

        link_text = "N/A"

    text =  f'<h3>{serial}: {title}</h3><table border = "1">'

    tdstring = get_td_string() #'<td style="text-align: left; vertical-align: middle;">'

    text_info = f'&nbsp;&nbsp;&nbsp;&nbsp;<b>Score:</b> {score} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Date:</b> {publish_date} NPI Count:{npi_count}'

    text_1 = '<tr>' + tdstring  + '<b>URL:</b>' + link_text + f'{text_info}</td>' + '</tr>'

    text_paper = '<tr>' + tdstring  + '<b>Paper ID:</b>'+ f'{paper_id}</td>' + '</tr>'

    text_author = '<tr>' + tdstring + f'<b>Author(s): </b>{authors}</td></tr>'

    text = text + text_1 + text_paper + text_author

    #text += ''.join([f'<td><b>{col}</b></td>' for col in df.columns.values]) + '</tr>'

    i = 0

    if (len(sentences) > 0):

        #text +=  f'<tr><td align ="left"><b>Relevant Sentences</b></td></tr>'

        text +=  tdstring + '<b>Relevant Sentences</b></td></tr>'

        for sent in sentences:

            i = i + 1  

            text +=  get_sentence_tr(sent)

    

    if (len(lst_other_keywords) > 0):

        text +=  tdstring + '<b>Sentences containing keywords - "rapid", "design", "experiments", "assessment" (and/or)</b></td></tr>'

        for sent in lst_other_keywords:

            i = i + 1  

            text +=  get_sentence_tr(sent)

    

    if (len(lst_cost_benefits) > 0):

        text +=  tdstring + '<b>Sentences containing keywords - "cost", "benefits" (and/or)</b></td></tr>'

        for sent in lst_cost_benefits:

            i = i + 1  

            text +=  get_sentence_tr(sent)

    

    if (len(lst_funding_infra_sentences) > 0):

        text +=  tdstring + '<b>Sentences containing keywords - "funding", "infra","authorities" (and/or) </b></td></tr>'

        for sent in lst_funding_infra_sentences:

            i = i + 1  

            text +=  get_sentence_tr(sent)  

    if (len(lst_all_urls) > 0):

        text +=  tdstring + '<b>All urls which appear in the article</b></td></tr>'

        str_urls = '<br> '.join([u for u in lst_all_urls])

        text +=  get_sentence_tr(str_urls)  

    

    text += '</table>'

    display(HTML(text))

    

def get_df_from_article_list(lst_articles): 

    lst = []

    serial = 0

    for l in lst_articles:

        serial +=1

        str_rel = "Low"

        url = l.main_url

        if (l.main_url == "NOT FOUND"):

            url = "N/A"

        if (l.npi_count > 0):

            str_rel = "High"

        dict_row = {"Serial": serial, "Title":l.title, "URL": url, "Score": l.score , "Relevance": str_rel, "PaperID": l.paper_id}

        lst.append(dict_row)

    #print(len(lst_articles), len(lst))

    return(pd.DataFrame(lst))



def get_processing_flag(module):

    retval = False

    if (module == BIORXIV):

        if (SAMPLE_SIZE_BIORXIV != -1): retval = True

    if (module == COMM):

        if (SAMPLE_SIZE_COMM != -1): retval = True

    if (module == NON_COMM):

        if (SAMPLE_SIZE_NON_COMM != -1): retval = True      

    if (module == CUSTOM_LICENSE):

        if (SAMPLE_SIZE_CUSTOM_LICENSE != -1): retval = True

    return(retval)



def print_user_message(mess = "Pl provide" ):

    m = f'<font  size=4 , color=grey >{mess}</font>'

    display(HTML(m))

    

def display_dataframe(df, title = ""):

    #tdstring = f'<td style="text-align: left; vertical-align: middle; font-size:1.2em;">{v}</td>'

    if (title != ""):

        text = f'<h2>{title}</h2><table><tr>'

    else:

        text = '<table><tr>'

    text += ''.join([f'<td style="text-align: left; vertical-align: middle; font-size:1.2em;"><b>{col}</b></td>' for col in df.columns.values]) + '</tr>'

    for row in df.itertuples():

        #text +=  '<tr>' + ''.join([f'<td valign="top">{v}</td>' for v in row[1:]]) + '</tr>'

        text +=  '<tr>' + ''.join([ f'<td style="text-align: left; vertical-align: middle; font-size:1.1em;">{v}</td>' for v in row[1:]]) + '</tr>'

    text += '</table>'

    display(HTML(text))

    

def start_td():

    tdstring = '<td style="text-align: center; vertical-align: middle; font-size:1.2em;">'

    return(tdstring)

def end_td():

    tdstring = '</td>'

    return(tdstring)

def get_bolded(tstr):

    tdstring = '<b>'+ tstr + '</b>'

    return(tdstring)



def get_sentence_tr_vs(sent):

    row = get_td_string() + f'{sent}</td></tr>'

    return(row)

    #return(  f'<tr>' + f'<td align = "left">{sent}</td>' + '<td>&nbsp;</td></tr>')

    

def display_data_processing_info():

    text =  f'<h3>Table: Data Processing Information</h3><table border = "1">'

    

    td_header_1 = start_td() + get_bolded("Module") + end_td() 

    td_header_2 = start_td() + get_bolded("Total articles") + end_td() 

    td_header_3 = start_td() + get_bolded("Processed articles") + end_td() 

    td_header_4 = start_td() + get_bolded("Number of articles of interest") + end_td() 

    td_header_5 = start_td() + get_bolded("Excerpts of interest") + end_td() 

    

    text_header = "\n<tr>" + td_header_1 + td_header_2  + td_header_3 + td_header_4 + td_header_5+ "</tr>\n"

    #text_header = text_header +  "<tr>" + start_td() + get_bolded("total articles") + end_td() + "</tr>\n"

    

    text = text + text_header

    

    if get_processing_flag(BIORXIV):   

        td_data_1 = start_td() + "Biorxiv/Medrxiv" + end_td()

        td_data_2 = start_td() + str(count_biorxiv_orig) + end_td()

        td_data_3 = start_td() + str(df_biorxiv.shape[0]) + end_td()

        td_data_4 = start_td() + str(df_biorxiv_filter.shape[0]) + end_td()

        td_data_5 = start_td() + str(sum_bio_sents) + end_td()

        text_row  = "\n<tr>" + td_data_1 +  td_data_2 + td_data_3 + td_data_4 + td_data_5 + "</tr>\n"

        text = text +  text_row

        

    if get_processing_flag(NON_COMM):   

        td_data_1 = start_td() + "Non Comm" + end_td()

        td_data_2 = start_td() + str(count_non_comm_orig) + end_td()

        td_data_3 = start_td() + str(df_non_comm.shape[0]) + end_td()

        td_data_4 = start_td() + str(df_non_comm_filter.shape[0]) + end_td()

        td_data_5 = start_td() + str(sum_non_comm_sents) + end_td()

        text_row  = "\n<tr>" + td_data_1 +  td_data_2 + td_data_3 + td_data_4 + td_data_5 + "</tr>\n"

        text = text +  text_row

        

    if get_processing_flag(COMM):   

        td_data_1 = start_td() + "Comm" + end_td()

        td_data_2 = start_td() + str(count_comm_orig) + end_td()

        td_data_3 = start_td() + str(df_comm.shape[0]) + end_td()

        td_data_4 = start_td() + str(df_comm_filter.shape[0]) + end_td()

        td_data_5 = start_td() + str(sum_comm_sents) + end_td()

        text_row  = "\n<tr>" + td_data_1 +  td_data_2 + td_data_3 + td_data_4 + td_data_5 + "</tr>\n"

        text = text +  text_row    

        

    if get_processing_flag(CUSTOM_LICENSE):   

        td_data_1 = start_td() + "Custom License" + end_td()

        td_data_2 = start_td() + str(count_custom_license_orig) + end_td()

        td_data_3 = start_td() + str(df_custom_license.shape[0]) + end_td()

        td_data_4 = start_td() + str(df_custom_license_filter.shape[0]) + end_td()

        td_data_5 = start_td() + str(sum_cl_sents) + end_td()

        text_row  = "\n<tr>" + td_data_1 +  td_data_2 + td_data_3 + td_data_4 + td_data_5 + "</tr>\n"

        text = text +  text_row   

    text += '\n</table>'

    display(HTML(text))



NOT_FOUND = "<not found>"

def add_list(*lsts):

    retlist = []

    for l in lsts:

        retlist =retlist + l

    return(list(set(retlist)))



def get_date(paper_url):

    retval = NOT_FOUND

    if paper_url.startswith('https://doi.org/'): # Could use /10.1101

        retval = paper_url.replace('https://doi.org/10.1101/', '')[0:10]

    return(retval)



class article():

    def __init__(self, paper_id, score, journal, lst_snippets, npi_count):

        self.publish_date = ""

        self.npi_count = npi_count

        self.paper_id = paper_id

        self.main_url = ""

        self.score = score

        self.journal = journal

        self.lst_sentences = []

        self.lst_snippets = lst_snippets

        self.nlp_text = None

        self.text = None

        self.lst_urls = []

        self.title = None

        self.authors = None

        self.lst_funding_infra_snippets = []

        self.lst_funding_infra_sentences = []

        self.lst_cost_benefits_snippets =[]

        self.lst_cost_benefits_sentences = []

        self.lst_all_sentences = []

        self.lst_other_keywords_snippets  = []

        self.lst_other_keywords_sentences = []

        self.count_sentences = 0

        self.initialize()

        self.consolidate_all_sentences()

        

    def consolidate_all_sentences(self):

        self.lst_all_sentences =  add_list(self.lst_sentences

                                         , self.lst_funding_infra_sentences

                                         , self.lst_cost_benefits_sentences

                                         , self.lst_other_keywords_sentences)   

        self.count_sentences = len(self.lst_all_sentences)

    def save_biorxiv_all_info(self):

        write_to_file(FILE_BIORXIV_ARTICLES_INFO, "=====================  START ===========================\n")

        write_to_file(FILE_BIORXIV_ARTICLES_INFO, "TITLE:" + self.title + "\n")

        write_to_file(FILE_BIORXIV_ARTICLES_INFO, "SENTENCES:" + ' \n'.join(self.lst_sentences))

        write_to_file(FILE_BIORXIV_ARTICLES_INFO, "=====================  END ===========================\n")

    def find_url_in_text(self):

        self.main_url , self.lst_urls = find_url_in_text(self.nlp_text)

    def initialize(self):

        self.text, self.title, self.authors = get_paper_info(self.paper_id, self.journal)

        self.nlp_text = nlp(self.text)

        self.find_url_in_text()

        self.get_sents_from_snippets()

        self.get_cost_benefits_info()

        self.get_funding_infra_info()

        self.get_other_keywords_info()

        self.publish_date = get_date(self.main_url)

    def get_sents_from_snippets(self):

        self.lst_sentences = []

        self.lst_sentences = get_sents_from_snippets(self.lst_snippets, self.nlp_text, self.paper_id)

    def get_funding_infra_info(self):

        phrase_list = [ u"funding", u"fund"

                       , u"authorities"

                       , u"infrastructure"]

        count, snippets = find_phrases_in_text(self.nlp_text, phrase_list, span_start = 1, span_end = 1 )

        self.lst_funding_infra_snippets = snippets

        if (count > 0):

            self.lst_funding_infra_sentences = get_sents_from_snippets(self.lst_funding_infra_snippets, self.nlp_text, self.paper_id) 

    

    def get_other_keywords_info(self):

        phrase_list = [ u"experiment", u"rapid", u"assesment", u"design"]

        count, snippets = find_phrases_in_text(self.nlp_text, phrase_list, span_start = 1, span_end = 1 )

        self.lst_other_keywords_snippets = snippets

        if (count > 0):

            self.lst_other_keywords_sentences = get_sents_from_snippets(snippets, self.nlp_text, self.paper_id)

    

    def get_cost_benefits_info(self):

        phrase_list = [u"cost", u"benefit"]

        count, snippets = find_phrases_in_text(self.nlp_text, phrase_list, span_start = 1, span_end = 1 )

        self.lst_cost_benefits_snippets = snippets

        if (count > 0):

            self.lst_cost_benefits_sentences = get_sents_from_snippets(self.lst_cost_benefits_snippets, self.nlp_text, self.paper_id)  

    def info_cost_benefits(self):

        if ((len(self.lst_cost_benefits_sentences) > 0) & (len(self.lst_cost_benefits_sentences) < 10)):

            self.print_header()

            print("Cost Benefits Information:", self.lst_cost_benefits_sentences)

            print("Number of cost benefits sentences found:", len(self.lst_cost_benefits_sentences))

            self.print_footer()

    def print_header(self):

        strformat = "================== START ===========================\n TITLE: {} \n".format(self.title)

        print(strformat)

    def print_footer(self):

        strformat = "RELEVANT URLS:\n {} \n PAPER ID {}".format(self.lst_urls, self.paper_id)

        print(strformat)

        print("PaperID: ", self.paper_id , "  Score:" , self.score)

        print("======================= END  ==========================================\n")

    def print_1_basic_article_information(self):

        self.print_header()

        print(" -------------- PRINTING SOME EXTRACTED SENTENCES (MAX 5) Related to NPI -------------- ")

        if len(self.lst_sentences) > 5:

            print_list(self.lst_sentences[0:5])

            #print(self.lst_sentences[0:5])

        else:

            print_list(self.lst_sentences)

        self.print_footer()



def get_objectlist_from_df(df):

    lst_objs = list(map(article, 

                          (df['paper_id']) ,

                          df['score'], 

                          df['module'],

                          df['lst_snippets'],

                          df['text_find_count_npi']))

    #print("sorting")

    #lst_objs.sort(key=lambda x: x.score, reverse=True)

    return (lst_objs)
# Processing Happens here

if get_processing_flag(BIORXIV):

    path = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'

    start_time = time.time()

    df_biorxiv_master, df_biorxiv, count_biorxiv_orig = process_a_module(path, SAMPLE_SIZE = SAMPLE_SIZE_BIORXIV, MODULE = BIORXIV)

    df_biorxiv_filter = df_biorxiv_master[df_biorxiv_master['score'] > 0].reset_index()#.sort_values(['score'], ascending = False)

    lst_obj_biorxiv = get_objectlist_from_df(df_biorxiv_filter)

    lst_obj_biorxiv.sort(key=lambda x: x.score, reverse=True)

    sum_bio_sents = sum(c.count_sentences for c in lst_obj_biorxiv)

    #print(sum_bio_sents)

    

if get_processing_flag(COMM):    

    path = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'

    start_time = time.time()

    df_comm_master, df_comm, count_comm_orig = process_a_module(path, SAMPLE_SIZE_COMM, COMM)

    df_comm_filter    = df_comm_master[df_comm_master['score'] > 0].reset_index()

    lst_obj_comm = get_objectlist_from_df(df_comm_filter)



    lst_obj_comm.sort(key=lambda x: x.score, reverse=True)

    sum_comm_sents = sum(c.count_sentences for c in lst_obj_comm)

    #print(sum_bio_sents)

    

if get_processing_flag(NON_COMM):    

    path = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'

    start_time = time.time()

    df_non_comm_master, df_non_comm, count_non_comm_orig = process_a_module(path, SAMPLE_SIZE_NON_COMM, NON_COMM)

    df_non_comm_filter = df_non_comm_master[df_non_comm_master['score'] > 0].reset_index()

    lst_obj_non_comm = get_objectlist_from_df(df_non_comm_filter)

    lst_obj_non_comm.sort(key=lambda x: x.score, reverse=True)

    sum_non_comm_sents = sum(c.count_sentences for c in lst_obj_non_comm)

    #len(lst_obj_non_comm)

    #print(sum_non_comm_sents)

    

if get_processing_flag(CUSTOM_LICENSE):   

    path = "/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/"

    start_time = time.time()

    df_custom_license_master, df_custom_license, count_custom_license_orig = process_a_module(path, SAMPLE_SIZE_CUSTOM_LICENSE, CUSTOM_LICENSE)

    df_custom_license_filter  = df_custom_license_master[df_custom_license_master['score'] > 0].reset_index()

    lst_obj_custom_license = get_objectlist_from_df(df_custom_license_filter)

    lst_obj_custom_license.sort(key=lambda x: x.score, reverse=True)

    sum_cl_sents = sum(c.count_sentences for c in lst_obj_custom_license)

    #print(sum_cl_sents)



display_data_processing_info()
#df_data = pd.DataFrame()

lst = []

lst_hit_ratio = []

if get_processing_flag(BIORXIV):

    

    dict_hit_ratio = { "module": "biorxiv", "ratio": df_biorxiv_filter.shape[0]/df_biorxiv.shape[0], "type" : "article_hit_ratio" }    

    lst_hit_ratio.append(dict_hit_ratio)

    dict_hit_ratio = { "module": "biorxiv", "ratio": sum_bio_sents/df_biorxiv.shape[0], "type" : "snippets_hit_ratio" }

    lst_hit_ratio.append(dict_hit_ratio)

    

    

    dict_row = {"count": count_biorxiv_orig , "module" : "biorxiv", "type" :"total"}

    lst.append(dict_row)

    dict_row = {"count": df_biorxiv.shape[0] , "module" : "biorxiv", "type" :"processed"}

    lst.append(dict_row)

    dict_row = {"count": df_biorxiv_filter.shape[0] , "module" : "biorxiv", "type" :"found"}

    lst.append(dict_row)

    dict_row = {"count": sum_bio_sents , "module" : "biorxiv", "type" :"excerpts"}

    lst.append(dict_row)



if get_processing_flag(NON_COMM):

    

   

    dict_hit_ratio = { "module": "non_comm", "ratio": df_non_comm_filter.shape[0]/df_non_comm.shape[0], "type" : "article_hit_ratio" }    

    lst_hit_ratio.append(dict_hit_ratio)

    dict_hit_ratio = { "module": "non_comm", "ratio": sum_non_comm_sents/df_non_comm.shape[0], "type" : "snippets_hit_ratio" }

    lst_hit_ratio.append(dict_hit_ratio)

    

    

    dict_row = {"count": count_non_comm_orig , "module" : "non_comm", "type" :"total"}

    lst.append(dict_row)

    dict_row = {"count": df_non_comm.shape[0] , "module" : "non_comm", "type" :"processed"}

    lst.append(dict_row)

    dict_row = {"count": df_non_comm_filter.shape[0] , "module" : "non_comm", "type" :"found"}

    lst.append(dict_row)

    dict_row = {"count": sum_non_comm_sents , "module" : "non_comm", "type" :"excerpts"}

    lst.append(dict_row)



if get_processing_flag(COMM):

    

    dict_hit_ratio = { "module": "comm", "ratio": df_comm_filter.shape[0]/df_comm.shape[0], "type" : "article_hit_ratio" }    

    lst_hit_ratio.append(dict_hit_ratio)

    dict_hit_ratio = { "module": "comm", "ratio": sum_comm_sents/df_comm.shape[0], "type" : "snippets_hit_ratio" }

    lst_hit_ratio.append(dict_hit_ratio)

    

    dict_row = {"count": count_comm_orig , "module" : "comm", "type" :"total"}

    lst.append(dict_row)

    dict_row = {"count": df_comm.shape[0] , "module" : "comm", "type" :"processed"}

    lst.append(dict_row)

    dict_row = {"count": df_comm_filter.shape[0] , "module" : "comm", "type" :"found"}

    lst.append(dict_row)

    dict_row = {"count": sum_comm_sents , "module" : "comm", "type" :"excerpts"}

    lst.append(dict_row)



if get_processing_flag(CUSTOM_LICENSE):

    



    dict_hit_ratio = { "module": "custom_license", "ratio": df_custom_license_filter.shape[0]/df_custom_license.shape[0], "type" : "article_hit_ratio" }    

    lst_hit_ratio.append(dict_hit_ratio)

    dict_hit_ratio = { "module": "custom_license", "ratio": sum_cl_sents/df_custom_license.shape[0], "type" : "snippets_hit_ratio" }

    lst_hit_ratio.append(dict_hit_ratio)

    

    

    dict_row = {"count": count_custom_license_orig , "module" : "custom_license", "type" :"total"}

    lst.append(dict_row)

    dict_row = {"count": df_custom_license.shape[0] , "module" : "custom_license", "type" :"processed"}

    lst.append(dict_row)

    dict_row = {"count": df_custom_license_filter.shape[0] , "module" : "custom_license", "type" :"found"}

    lst.append(dict_row)

    dict_row = {"count": sum_cl_sents , "module" : "custom_license", "type" :"excerpts"}

    lst.append(dict_row)



df_data = pd.DataFrame(lst)

df_hit_ratio = pd.DataFrame(lst_hit_ratio)

#df = df_data

fig = px.bar(df_hit_ratio, x="module", y="ratio", color='type', barmode='group',  title = "Hit Ratio of various modules", template = "plotly_dark")

fig.show()
fig = px.bar(df_data, x="type", y="count", color='module', barmode='group', log_y= True, title = "Various stats from articles after initial screening",  template = "plotly_dark")

fig.show()
if get_processing_flag(BIORXIV):

    serial = 0

    for l in lst_obj_biorxiv:

        if (l.npi_count > 0):

            serial = serial + 1

            display_article(serial, l.title, l.main_url, l.lst_sentences, l.score

                        , l.lst_other_keywords_sentences

                        , l.lst_cost_benefits_sentences

                        , l.lst_funding_infra_sentences

                        , l.lst_urls

                        , l.authors

                        , l.publish_date

                        , l.npi_count

                        , l.paper_id)

else:

    print_user_message("This module has not been processed. Please set Sample size other than -1 in the Input area for this module to be processed.")
if get_processing_flag(NON_COMM):

    serial = 0

    for l in lst_obj_non_comm:

        if (l.npi_count > 0):

            serial = serial + 1

            display_article(serial, l.title, l.main_url, l.lst_sentences, l.score

                        , l.lst_other_keywords_sentences

                        , l.lst_cost_benefits_sentences

                        , l.lst_funding_infra_sentences

                        , l.lst_urls

                        , l.authors

                        , l.publish_date

                        , l.npi_count

                        , l.paper_id)

else:

    print_user_message("This module has not been processed. Please set Sample size other than -1 in the Input area for this module to be processed.")
if get_processing_flag(COMM):

    serial = 0

    for l in lst_obj_comm:

        if (l.npi_count > 0):

            serial = serial + 1

            display_article(serial, l.title, l.main_url, l.lst_sentences, l.score

                        , l.lst_other_keywords_sentences

                        , l.lst_cost_benefits_sentences

                        , l.lst_funding_infra_sentences

                        , l.lst_urls

                        , l.authors

                        , l.publish_date

                        , l.npi_count

                        , l.paper_id)

else:

    print_user_message("This module has not been processed. Please set Sample size other than -1 in the Input area for this module to be processed.")
if get_processing_flag(CUSTOM_LICENSE):

    serial = 0

    for l in lst_obj_custom_license:

        if (l.npi_count > 0):

            serial = serial + 1

            display_article(serial, l.title, l.main_url, l.lst_sentences, l.score

                        , l.lst_other_keywords_sentences

                        , l.lst_cost_benefits_sentences

                        , l.lst_funding_infra_sentences

                        , l.lst_urls

                        , l.authors

                        , l.publish_date

                        , l.npi_count

                        , l.paper_id)

else:

    print_user_message("This module has not been processed. Please set Sample size other than -1 in the Input area for this module to be processed.")
if get_processing_flag(BIORXIV):

    display_dataframe(get_df_from_article_list(lst_obj_biorxiv))

else:

    print_user_message("This module has not been processed. Please set Sample size other than -1 in the Input area for this module to be processed.")
if get_processing_flag(NON_COMM):

    display_dataframe(get_df_from_article_list(lst_obj_non_comm))

else:

    print_user_message("This module has not been processed. Please set Sample size other than -1 in the Input area for this module to be processed.")
if get_processing_flag(COMM):

    display_dataframe(get_df_from_article_list(lst_obj_comm))

else:

    print_user_message("This module has not been processed. Please set Sample size other than -1 in the Input area for this module to be processed.")
if get_processing_flag(CUSTOM_LICENSE):

    display_dataframe(get_df_from_article_list(lst_obj_custom_license))

else:

    print_user_message("This module has not been processed. Please set Sample size other than -1 in the Input area for this module to be processed.")