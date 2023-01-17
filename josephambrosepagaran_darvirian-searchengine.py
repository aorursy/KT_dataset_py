from IPython.display import Image

Image("/kaggle/input/nlpimage/nlpsearch_engine.jpg")
# ============================================================================

# PART I: LOAD THE DATA

# ============================================================================

## Read worddic from all CORD-19 papers



import pickle



## Load pickle file worddic

pickle_in = open('/kaggle/input/darvirian/worddic_all_200407.pkl', 'rb')

worddic = pickle.load(pickle_in)



## Load pickle file worddic

pickle_in = open('/kaggle/input/darvirian/df.pkl', 'rb')

df = pickle.load(pickle_in)



## Load pickle file sentences

pickle_in = open('/kaggle/input/darvirian/sentences_200407.pkl', 'rb')

sentences = pickle.load(pickle_in)
# ============================================================================

# PART II: The Search Engine

# ============================================================================

import re

from collections import Counter



def search(searchsentence):

    # split sentence into individual words

    searchsentence = searchsentence.lower()

    # split sentence in words and keep characters as in worddic

    words = searchsentence.split(' ')

    words = [re.sub(r'[^a-zA-Z.]', '', str(w)) for w in words]



    # temp dictionaries

    enddic = {}

    idfdic = {}

    closedic = {}



    # remove words if not in worddic 

    # (keep only the words that are in the dictionary)

    words = [word for word in words if word in worddic.keys()]

    numwords = len(words)



    # metrics fullcount_order and fullidf_order: 

    # sum of number of occurences of all words in each doc (fullcount_order) 

    # and sum of TF-IDF score (fullidf_order)

    for word in words:

        # print(word)

        for indpos in worddic[word]:

            # print(indpos)

            index = indpos[0]

            amount = len(indpos[1])

            idfscore = indpos[2]

            # check if the index is already in the dictionary: 

            # add values to the keys

            if index in enddic.keys():

                enddic[index] += amount

                idfdic[index] += idfscore

            # if not, just make a two new keys and store the values

            else:

                enddic[index] = amount

                idfdic[index] = idfscore

    fullcount_order = sorted(enddic.items(), 

                             key=lambda x: x[1], 

                             reverse=True

                            )

    fullidf_order = sorted(idfdic.items(), 

                           key=lambda x: x[1], 

                           reverse=True

                          )



    # metric combocount_order: 

    # percentage of search words (as in dict) that appear in each doc

    # (and is it a reason to give these docs more relevance)

    alloptions = {k: worddic.get(k) for k in words}

    comboindex = [item[0] for worddex in alloptions.values() for item in worddex]

    combocount = Counter(comboindex) # count the time of each index

    for key in combocount:

        combocount[key] = combocount[key] / numwords

    combocount_order = sorted(combocount.items(), 

                              key=lambda x: x[1], 

                              reverse=True

                             )



    # metric closedic: if words appear in same order as in search

    if len(words) > 1:

        x = [index[0] for record in [worddic[z] for z in words] for index in record]

        y = sorted(list(set([i for i in x if x.count(i) > 1])))



        # dictionary of documents 

        # and all positions (for docs with more than one search word in it)

        closedic = {}

        for wordbig in [worddic[x] for x in words]:

            for record in wordbig:

                if record[0] in y:

                    index = record[0]

                    positions = record[1]

                    try:

                        closedic[index].append(positions)

                    except:

                        closedic[index] = []

                        closedic[index].append(positions)

        # Index add to comprehension:

        # closedic2 = [record[1] for wordbig in [worddic[x] 

        # for x in words] for record in wordbig if record[0] in y]



        # metric: fdic number of times 

        # search words appear in a doc in descending order

        # TODO check

        x = 0

        fdic = {}

        for index in y: # list with docs with more than one search word

            csum = []            

            for seqlist in closedic[index]:

                while x > 0:

                    secondlist = seqlist # second word positions

                    x = 0

                    # first and second word next to each other (in same order)

                    sol = [1 for i in firstlist if i + 1 in secondlist]

                    csum.append(sol)

                    fsum = [item for sublist in csum for item in sublist] 

                    fsum = sum(fsum) 

                    fdic[index] = fsum

                    fdic_order = sorted(fdic.items(), 

                                        key=lambda x: x[1], reverse=True)

                while x == 0:

                    firstlist = seqlist # first word positions 

                    x = x + 1 

    else:

        fdic_order = 0



    # TODO another metric for if they are not next to each other but still close



    return(searchsentence, 

           words, 

           fullcount_order, 

           combocount_order, 

           fullidf_order, 

           fdic_order

          )
# =============================================================================

# PART III: Rank and return (rule based)

# =============================================================================

import pandas as pd

import time 



# Find sentence of search word(s)

def search_sentence(doc_number, search_term):

    sentence_index = []

    search_list = search_term.split()

    for sentence in sentences[doc_number]:

        for search_word in search_list:

            if search_word.lower() in sentence.lower():

                # df.Sentences[doc_number].index(sentence)

                sentence_index.append(sentence) 

    return sentence_index



def rank(term):



    start_time = time.time()

    # get results from search

    results = search(term)

    disp_search_words = results[1]

    # get metrics

    # number of search words found in dictionary

    num_search_words = len(results[1]) 

    # number of search words (as in dict) in each doc (in descending order)

    num_score = results[2] 

    # percentage of search words (as in dict) in each doc (in descending order)

    per_score = results[3] 

    # sum of tfidf of search words in each doc (in ascending order)

    tfscore = results[4] 

    order_score = results[5] # fidc order



    # list of documents in order of relevance

    final_candidates = []



    # no search term(s) not found

    if num_search_words == 0:

        print('Search term(s) not found')



    # single term searched (as in dict): return the following 5 scores

    if num_search_words == 1:

        # document numbers

        num_score_list = [l[0] for l in num_score] 

        # take max 3 documents from num_score

        num_score_list = num_score_list[:3] 

        # add the best percentage score

        num_score_list.append(per_score[0][0]) 

        # add the best tf score

        num_score_list.append(tfscore[0][0]) 

        # remove duplicate document numbers

        final_candidates = list(set(num_score_list)) 





    # more than one search word (and found in dictionary)

    if num_search_words > 1:



        # rule1: doc with high fidc order_score (>1) 

        # and 100% percentage search words (as in dict) on no. 1 position

        first_candidates = []



        # first candidate(s) comes from fidc order_score (with value > 1)

        for candidates in order_score:

            if candidates[1] >= 1:

                first_candidates.append(candidates[0])



        second_candidates = []



        for match_candidates in per_score:

            # if all words are in a document: add to second_candidates

            # TODO check why per_score sometimes > 1 (change to >=1 ?)

            if match_candidates[1] == 1:

                second_candidates.append(match_candidates[0])

        # first final candidates have the highest score of 

        # search words next to each other 

        # and all search words (as in dict) in document  

        for match_candidates in first_candidates:

            if match_candidates in second_candidates:

                final_candidates.append(match_candidates)



        # rule2: add max 4 other words 

        # with order_score greater than 1 (if not yet in final_candiates)

        t3_order = first_candidates[0:3]

        for each in t3_order:

            if each not in final_candidates:

                final_candidates.insert(len(final_candidates), 

                                        each

                                       )



        # rule3: add 2 top td-idf results to final_candidates

        final_candidates.insert(len(final_candidates), 

                                tfscore[0][0]

                               )

        final_candidates.insert(len(final_candidates), 

                                tfscore[1][0]

                               )



        # rule4: next add four other high percentage score 

        # (if not yet in final_candiates)

        # the first 4 high percentages scores 

        # (if equal to 100% of search words in doc)

        t3_per = second_candidates[0:3] 

        for each in t3_per:

            if each not in final_candidates:

                final_candidates.insert(len(final_candidates), each)



        # rule5: next add any other no. 1 result in num_score, 

        # per_score, tfscore and order_score (if not yet in final_candidates)

        othertops = [num_score[0][0], 

                     per_score[0][0], 

                     tfscore[0][0], 

                     order_score[0][0]

                    ]

        for top in othertops:

            if top not in final_candidates:

                final_candidates.insert(len(final_candidates), top)



    # top results: sentences with search words, 

    # paper ID (and documet number), authors and abstract

    df_results = pd.DataFrame(columns=['Title', 

                                       'Paper_id', 

                                       'Document_no', 

                                       'Authors', 

                                       'Abstract', 

                                       'Sentences'

                                      ]

                             )

    for index, results in enumerate(final_candidates):

        # if index < 5:

        df_results.loc[index+1, 'Title'] = df.title[results]

        df_results.loc[index+1, 'Paper_id'] = df.paper_id[results]

        df_results.loc[index+1, 'Document_no'] = results

        df_results.loc[index+1, 'Authors'] = df.authors[results]

        df_results.loc[index+1, 'Abstract'] = df.abstract[results]

        search_results = search_sentence(results, term)

        # remove duplicate sentences

        unique_search_results = list(dict.fromkeys(search_results))

        df_results.loc[index+1, 'Sentences'] = unique_search_results

    

    end_time = time.time()

    # print final candidates

    print('\nFound search words:', disp_search_words)

    print('Number of documents found:', len(final_candidates))

    print('Processing Time: ', round(end_time - start_time, 2), ' s')

    # print('Ranked papers (document numbers):', final_candidates)        

        

    return final_candidates, df_results





# truncate words/phases up to length sz

# for hoover tool to be a bit tidy 

# when showing the sentence count, paper ID, paper title, and abstract

def truncate_str(input_str,sz):

    if len(input_str) < sz:

      return input_str

    else :

      return input_str[:sz] + ' ...'



def truncate_palatte(desc_num_sentence, 

                     desc_paper_id, 

                     desc_paper_title, 

                     desc_paper_abstract,

                     desc_paper_sentences

                    ):

    sz = len(desc_num_sentence)

    if sz < 256:

        source = ColumnDataSource(data=dict(paper_id=desc_paper_id, 

                                            title=desc_paper_title, 

                                            abstract=desc_paper_abstract, 

                                            sentences=desc_paper_sentences, 

                                            paper=[x for x in range(0,sz)], 

                                            count=desc_num_sentence, 

                                            color=viridis(sz)

                                            )  

                                 )   

    else :

        sz = 256 - 1

        source = ColumnDataSource(data=dict(paper_id=desc_paper_id[:sz], 

                                            title=desc_paper_title[:sz], 

                                            abstract=desc_paper_abstract[:sz], 

                                            sentences=desc_paper_sentences[:sz],

                                            paper=[x for x in range(0,sz)], 

                                            count=desc_num_sentence[:sz], 

                                            color=viridis(sz)

                                            )  

                                 )

 

    return source

    

import numpy as np

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource, HoverTool

from bokeh.palettes import viridis

from bokeh.io import show, output_notebook



def bokeh_interactive_plot(rank_result):



    #collect lists for hoover tool visualisation 

    paper_id = list(rank_result['Paper_id'])

    paper_title = list(rank_result['Title'])

    paper_abstract = list(rank_result['Abstract'])

    paper_sentences = list(rank_result['Sentences'])



    # sort according to decreasing sentence counts

    num_sentence_per_paper = [len(rank_result['Sentences'][x]) 

                              for x in range(1,len(rank_result)+1)

                             ]

    descending_sort_idx = np.array(num_sentence_per_paper).argsort()[::-1]



    sort_idx = descending_sort_idx

    desc_num_sentence = [num_sentence_per_paper[sort_idx[x]] 

                         for x in range(len(sort_idx))

                        ]

    desc_paper_id = [paper_id[sort_idx[x]] for x in range(len(sort_idx))]

    # displaying  paper_title up to 200 characters

    desc_paper_title = [truncate_str(str(paper_title[sort_idx[x]]),200) 

                        for x in range(len(sort_idx))

                       ]

    # displaying paper_abstract up to 400 characters

    desc_paper_abstract = [truncate_str(str(paper_abstract[sort_idx[x]]),400) 

                           for x in range(len(sort_idx))

                          ]

    # displaying paper_sentences up to 600 characters

    desc_paper_sentences = [truncate_str(str(paper_sentences[sort_idx[x]]),600) 

                           for x in range(len(sort_idx))

                          ]



    # Bokeh's mapping of column names and data lists

    source = truncate_palatte(desc_num_sentence, 

                              desc_paper_id, 

                              desc_paper_title, 

                              desc_paper_abstract,

                              desc_paper_sentences

                             )

    

    sz = len(desc_num_sentence)

    x_val = [x for x in range(0,sz)]

    y_val = desc_num_sentence



    p = figure(x_range=(-int(max(x_val)/6),max(x_val)+int(max(x_val)/2)), 

               y_range=(-int(max(y_val)/2),max(y_val)+int(max(y_val)/5)), 

               plot_height=750, 

               plot_width=950, 

               title="Sentence Counts per paper",

               toolbar_location=None, 

               tools=""

              )



    # Render and show the vbar plot

    p.vbar(x='paper', top='count', width=0.9, color='color', source=source)

    # Hover tool referring to our own data field using @

    p.add_tools(HoverTool(tooltips=[("count", "@count"),

                                    ("ID", "@paper_id"),

                                    ("title", "@title"),

                                    ("abstract", "@abstract"),

                                    ("sentences", "@sentences")

                                   ]

                         )

               )



    # Set to output the plot in the notebook

    output_notebook()

    # Show the plot

    return show(p, notebook_handle=True)
papers, rank_result = rank('Real-time tracking whole genomes mechanism coordinating rapid dissemination information online GISAID NCBI GenBank MN908947 MN996532 AY278741 KY417146 MK211376 development diagnostics therapeutics Chloroquine track variations virus time')

bokeh_interactive_plot(rank_result)
papers, rank_result = rank('Wuhan seafood market geographic temporal sample sets distribution genomic differences phylogenetic strain circulation genetic lineages multi-lateral agreements Nagoya Protocol')

bokeh_interactive_plot(rank_result)
papers, rank_result = rank('Evidence animal livestock market infected zoonotic transmission field surveillance genetic sequencing ACE2 receptor binding reservoir epidemic  pathogenesis')

bokeh_interactive_plot(rank_result)
papers, rank_result = rank('Evidence farmers infected role origin zoonotic transmission pathogenesis human-to-human transmission')

bokeh_interactive_plot(rank_result)
papers, rank_result = rank('Surveillance wildlife livestock farms SARS-CoV-2 other coronaviruses SARS-CoV MERS-CoV SARS-CoV-2 HKU1 NL63 OC43 229E Southeast Asia Indonesia Malaysia Singapore Philippines East Timor Brunei Cambodia Laos Myanmar Thailand Vietnam')

bokeh_interactive_plot(rank_result)
papers, rank_result = rank('Experimental infections host range pathogen ACE2 receptor')

bokeh_interactive_plot(rank_result)
papers, rank_result = rank('Animal host evidence RaTG13 Rhinolophus affinis bats ferrets cats pangolins')

bokeh_interactive_plot(rank_result)
papers, rank_result = rank('Socioeconomic behavioral risk factors healthcare services business closure unemployment')

bokeh_interactive_plot(rank_result)
papers, rank_result = rank('Sustainable risk reduction strategies community-based measures regular handwashing quarantine')

bokeh_interactive_plot(rank_result)