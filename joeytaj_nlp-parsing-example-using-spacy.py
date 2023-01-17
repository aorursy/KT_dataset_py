import spacy

from spacy.en import English

from spacy.symbols import *

nlp = English()



import pandas as pd

import numpy as np

from fuzzywuzzy import fuzz

from tqdm import tqdm, tqdm_notebook



import matplotlib.pyplot as plt

import seaborn as sns



root_path = '../input/'
def get_distinct_questions(train, test):

    df1 = train[['question1']].copy()

    df2 = train[['question2']].copy()

    df1_test = test[['question1']].copy()

    df2_test = test[['question2']].copy()



    df2.rename(columns = {'question2':'question1'},inplace=True)

    df2_test.rename(columns = {'question2':'question1'},inplace=True)



    questions = df1.append(df2)

    questions = questions.append(df1_test)

    questions = questions.append(df2_test)

    

    questions.drop_duplicates(subset = ['question1'],inplace=True)



    questions.reset_index(inplace=True,drop=True)

    del df1,df1_test, df2, df2_test

    return questions



def parse_question(doc):

    #doc = nlp(question)

    

    pobj = []

    dobj = []

    num = []

    qword = []

    noun = []

    verb = []

    adj = []

    ents = []

    ent_types = []

    

    

    ent = []

    ent_type = ""

    sent_count = 0

    for s in doc.sents:

        sent_count+=1

        for word in s:

            #print("(" + word.ent_type_  + "," + str(word.pos)  + "," + word.pos_  + "," + str(word.pos)  + "," + word.tag_  \

            #      + "," + str(word.tag)  + "," + word.dep_    + "," + str(word.dep)  + "," + word.lemma_ + ") ")

            #ENTITIES

            if word.ent_type == 0:

                if len(ent) > 0:

                    ents.append('_'.join(ent))                    

                    ent_types.append(ent_type)

                    ent = []

                    ent_type = ""

            elif word.ent_type > 0 and word.ent_iob == 3:

                if len(ent) > 0:

                    ents.append('_'.join(ent))                    

                    ent_types.append(ent_type)

                    ent = []

                ent.append(word.lemma_)

                ent_type = word.ent_type_ 

            elif word.ent_type > 0 and word.ent_iob == 1:                

                ent.append(word.lemma_)

                

            #QUESTIONS

            if word.tag_.find('W') == 0:

                qword.append(word.lemma_)

            #NOUNS

            elif word.pos in [90,94]:

                noun.append(word.lemma_)

                

                #pobj

                if word.dep == 435:

                    pobj.append(word.lemma_)

                #dobj

                elif word.dep == 412:

                    dobj.append(word.lemma_)



            #NUMBER

            elif word.pos in [91]:

                num.append(word.lemma_)

            #ADJ

            elif word.pos in [82]:

                adj.append(word.lemma_)

            #VERB

            elif word.pos in [98]:

                verb.append(word.lemma_)     

            

    if len(ent) > 0:

        ents.append('_'.join(ent))                    

        ent_types.append(ent_type)

        ent = []   

    #print(sent_count, pobj, dobj, num, qword, noun, verb, adj, ents, ent_types)

    return sent_count, pobj, dobj, num, qword, noun, verb, adj, ents, ent_types





def match_count(list1, list2):

    return len(set(list1).intersection(set(list2)))



def diff_count(list1, list2):

    return len([obj for obj in list1 if obj not in list2] + [obj for obj in list2 if obj not in list1])



def get_nlp_features(nlp_parts1, nlp_parts2):



    sent_count1, pobj1, dobj1, num1, qword1, noun1, verb1, adj1, ents1, ent_types1 = nlp_parts1

    sent_count2, pobj2, dobj2, num2, qword2, noun2, verb2, adj2, ents2, ent_types2 = nlp_parts2

    

    ret = []

    

    f = diff_count

    ret1 = [abs(sent_count1 - sent_count2), f(pobj1,pobj2), f(dobj1,dobj2),\

            f(num1,num2), f(qword1,qword2), f(noun1,noun2),\

            f(verb1,verb2), f(adj1,adj2), f(ents1,ents2), f(ent_types1,ent_types1)]

    

    f = match_count

    ret2 = [f(pobj1,pobj2), f(dobj1,dobj2),\

            f(num1,num2), f(qword1,qword2), f(noun1,noun2),\

            f(verb1,verb2), f(adj1,adj2), f(ents1,ents2), f(ent_types1,ent_types1)]

    ret3 = [ret2[0] * ret2[0],ret2[1] * ret2[1],ret2[2] * ret2[2],ret2[3] * ret2[3],ret2[4] * ret2[4],ret2[5] * ret2[5]\

            ,ret2[6] * ret2[6],ret2[7] * ret2[7],ret2[8] * ret2[8]]

    f = fuzz.QRatio

    ret4 = [f(pobj1,pobj2), f(dobj1,dobj2), f(num1,num2), f(qword1,qword2), f(noun1,noun2), f(verb1,verb2), f(adj1,adj2), f(ents1,ents2)]

    

    f = fuzz.WRatio

    ret5 = [f(pobj1,pobj2), f(dobj1,dobj2), f(num1,num2), f(qword1,qword2), f(noun1,noun2), f(verb1,verb2), f(adj1,adj2), f(ents1,ents2)]

          

    f = fuzz.token_set_ratio

    ret6 = [f(pobj1,pobj2), f(dobj1,dobj2), f(num1,num2), f(qword1,qword2), f(noun1,noun2), f(verb1,verb2), f(adj1,adj2), f(ents1,ents2)]

    

    f = fuzz.token_sort_ratio

    ret7 = [f(pobj1,pobj2), f(dobj1,dobj2), f(num1,num2), f(qword1,qword2), f(noun1,noun2), f(verb1,verb2), f(adj1,adj2), f(ents1,ents2)]

    

    f = fuzz.partial_ratio

    ret8 = [f(pobj1,pobj2), f(dobj1,dobj2), f(num1,num2), f(qword1,qword2), f(noun1,noun2), f(verb1,verb2), f(adj1,adj2), f(ents1,ents2)]

    

    f = fuzz.partial_token_sort_ratio

    ret9 = [f(pobj1,pobj2), f(dobj1,dobj2), f(num1,num2), f(qword1,qword2), f(noun1,noun2), f(verb1,verb2), f(adj1,adj2), f(ents1,ents2)]

    

    f = fuzz.partial_token_set_ratio

    ret10 = [f(pobj1,pobj2), f(dobj1,dobj2), f(num1,num2), f(qword1,qword2), f(noun1,noun2), f(verb1,verb2), f(adj1,adj2), f(ents1,ents2)]

    

    ret11 = [sent_count1, sent_count2, len(pobj1), len(pobj2), len(dobj1), len(dobj2), len(num1),len(num2),\

            len(qword1),len(qword2), len(noun1),len(noun2), len(verb1),len(verb2), len(adj1),len(adj2), len(ents1),len(ents2)]

    

    

    ret.extend(ret1)

    ret.extend(ret2)

    ret.extend(ret3)

    ret.extend(ret4)

    ret.extend(ret5)

    ret.extend(ret6)

    ret.extend(ret7)

    ret.extend(ret8)

    ret.extend(ret9)

    ret.extend(ret10)

    ret.extend(ret11)

    return tuple(ret)
%%time

train_data =  pd.read_csv(root_path + 'train.csv', header=0)

test_data =  pd.read_csv(root_path + 'test.csv', header=0)
train_questions = get_distinct_questions(train_data, test_data)
nlp_parse_lookup = {}

index = 0

for doc in tqdm_notebook(nlp.pipe([str(q) for q in train_questions['question1']], n_threads=16, batch_size=10000), total = len(train_questions)):

    nlp_parse_lookup[str(train_questions.iloc[index]['question1'])] = parse_question(doc)

    index += 1

    
type_list =['pobj', 'dobj', 'num', 'qword', 'noun', 'verb', 'adj', 'ents']



columns=[]

columns.append('sent_diff_count') 

columns.extend([t + "_diff_count" for t in type_list])  

columns.append('ent_types_diff_count') 



columns.extend([t + "_match_count" for t in type_list])  

columns.append('ent_types_match_count') 



columns.extend([t + "_match_square" for t in type_list])  

columns.append('ent_types_match_square')                

                                                                                                                

columns.extend([t + "_QRatio" for t in type_list])                                                             

columns.extend([t + "_WRatio" for t in type_list])                                                             

columns.extend([t + "_token_set_ratio" for t in type_list])                                                             

columns.extend([t + "_token_sort_ratio" for t in type_list])                                                             

columns.extend([t + "_partial_ratio" for t in type_list])                                                             

columns.extend([t + "_partial_token_sort_ratio" for t in type_list])                                                             

columns.extend([t + "_partial_token_set_ratio" for t in type_list])

                                                                                                              

columns.extend(['sent_count1', 'sent_count2','len_pobj1', 'len_pobj2', 'len_dobj1', 'len_dobj2',\

                'len_num1', 'len_num2', 'len_qword1', 'len_qword2', 'len_noun1', 'len_noun2', \

                'len_verb1', 'len_verb2', 'len_adj1', 'len_adj2', 'len_ents1', 'len_ents2'])
feature_list = [get_nlp_features(nlp_parse_lookup[str(q[0])], nlp_parse_lookup[str(q[1])]) \

                for q in tqdm_notebook(train_data[['question1','question2']].values, total = len(train_data))]

nlp_feat = pd.DataFrame(feature_list, columns=columns)
feature_list = [get_nlp_features(nlp_parse_lookup[str(q[0])], nlp_parse_lookup[str(q[1])]) \

                for q in tqdm_notebook(test_data[['question1','question2']].values, total = len(test_data))]

nlp_test_feat = pd.DataFrame(feature_list, columns=columns)
nlp_feat.to_csv(root_path + 'quora_train_features_nlp.tsv', index=False, sep='\t')

nlp_test_feat.to_csv(root_path + 'quora_test_features_nlp.tsv', index=False, sep='\t')
nlp_feat['is_duplicate'] = train_data['is_duplicate']
mcorr = nlp_feat.corr()
#Check

mcorr.sort_values(['is_duplicate'])['is_duplicate']
for column_name in mcorr.columns:

    index = 0

    matches = mcorr.query('abs(' + str(column_name) + ') >= 0.995').sort_values(column_name)[column_name]    

    if len(matches) > 1:

        print()

        print(column_name  + "\n----------------")

        for match in matches:            

            if matches.index[index] != column_name:

                print(matches.index[index] + '\t' + str(match))

       

            index += 1
def plot_real_feature(fname, train_feat):

    fig = plt.figure()

    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)

    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)

    ax3 = plt.subplot2grid((3, 2), (2, 0))

    ax4 = plt.subplot2grid((3, 2), (2, 1))

    ax1.set_title('Distribution of %s' % fname, fontsize=20)

    sns.distplot(train_feat[fname], 

                 bins=50, 

                 ax=ax1)    

    sns.distplot(train_feat[train_feat.is_duplicate == 1][fname], 

                 bins=50, 

                 ax=ax2,

                 label='is dup')    

    sns.distplot(train_feat[train_feat.is_duplicate == 0][fname], 

                 bins=50, 

                 ax=ax2,

                 label='not dup')

    ax2.legend(loc='upper right', prop={'size': 18})

    sns.boxplot(y=fname, 

                x='is_duplicate', 

                data=train_feat, 

                ax=ax3)

    sns.violinplot(y=fname, 

                   x='is_duplicate', 

                   data=train_feat, 

                   ax=ax4)

    plt.show()



def plot_corr(mcorr):    

    

    mask = np.zeros_like(mcorr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')

    g.set_xticklabels(mcorr.columns, rotation=90)

    g.set_yticklabels(reversed(mcorr.columns))

    plt.show()
plot_real_feature('pobj_match_count', nlp_feat)