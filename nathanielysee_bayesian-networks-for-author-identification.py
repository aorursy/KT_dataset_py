'''

import time

import os

import pickle

import pandas as pd

from nltk.internals import find_jars_within_path



pathstring = "C:/Users/Nathaniel/Desktop/stanford-parser-full-2017-06-09"

os.environ["STANFORD_MODELS"] = pathstring + "/stanford-parser-3.8.0-models.jar"

os.environ["STANFORD_PARSER"] = pathstring + "/stanford-parser.jar"

os.environ["JAVAHOME"] = "C:/Program Files/Java/jdk-9.0.1"

os.environ["CLASSPATH"] = pathstring



input_file = 'test.csv'

output_file = "test_trees.p"



from nltk.parse import stanford

import nltk as nl

parser = nl.parse.stanford.StanfordParser(model_path=(pathstring + "/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"),java_options='-mx16000m')

parser._classpath = tuple(find_jars_within_path(pathstring))

data = pd.read_csv(input_file)

data['tokens'] = data['text'].map(nl.word_tokenize)

t = time.clock()

treelist = dict()

nones = 0

for i, sent in enumerate(data['text']):

    l = len(data['tokens'][i])

    if l < 200:

        treelist[i] = next(next(parser.raw_parse_sents([sent])))    

    else:

        treelist[i] = None

        nones = nones + 1

        print(l, end=' ')

    print(i, ": ", time.clock() - t)

print(nones)



with open(output_file,"wb") as filehandle:

    pickle.dump(treelist,filehandle)

    

'''
import pickle

int_outputs = "../input/intermediate-outputs"



with open(int_outputs+"/test_matrices_all.p", "rb") as filehandle:

    test_matrices = pickle.load(filehandle)
test_matrices[0]
def get_rels_ch3(tree, stemmer=None):

    '''takes a single tree and an optional stemmer, returns a matrix as a pandas DataFrame'''

    tree.chomsky_normal_form()

    tree = nl.ParentedTree.convert(tree)#ParentedTree allows us to access a node's parents

    start_rows = pd.MultiIndex.from_tuples([("LEAF","_END")])

    start_cols = pd.MultiIndex.from_tuples([("ROOT", 'R')])

    child_par = pd.DataFrame([0],index=start_rows, columns=start_cols)#Dataframe to return

    

    for sub in tree.subtrees():

        par = sub.parent()

        if par:

            p_ind = sub.parent_index()

            par_label = par.label()

            child_label = sub.label()

            

            par_label = re.sub(r"(\|[^-]*)|(-[^-]*)","+",par_label)#simplify composite labels

            child_label = re.sub(r"(\|[^-]*)|(-[^-]*)","+",child_label)#like VP<NN-PP> to VP++

            

            new_cols = pd.MultiIndex.from_tuples([(par_label, 'L'), (par_label, 'R')])

            new_row = pd.MultiIndex.from_tuples([("NODE",child_label)])

            new_entry = pd.DataFrame([[1-p_ind, p_ind]],index=new_row, columns=new_cols)

            

            child_par = child_par.add(new_entry, fill_value=0)

            

            if not sub.left_sibling() and not sub.right_sibling():

                child_par.loc[("LEAF","_END"),(par_label,'R')] +=1

        

            if sub.height() == 2:#nltk does not consider leaves to be subtrees

                for leaf in sub.leaves():#so we must handle them separately

                    if stemmer:

                        leaf = stemmer.stem(leaf)

                    leaf = pd.MultiIndex.from_tuples([("LEAF",leaf.lower())])

                    leaf_par = pd.MultiIndex.from_tuples([(child_label, "C")])

                    

    

                    new_leaf = pd.DataFrame([1],index=leaf, columns=leaf_par)

                    child_par = child_par.add(new_leaf, fill_value=0)

            return child_par.to_sparse()
def get_counts(column, trees,stem = True):

    '''takes a pandas series, like the author column of the training data, and a list or dictionary of trees.

    the column is used for identifying who each sentence belongs to, and must line up with the list/dict indices

    returns the sum of all matrices generated from the list of trees

    '''

    t1 = time.clock()

    

    stemmer = None

    if stem: stemmer = nl.stem.snowball.SnowballStemmer("english")

        

    start_rows = pd.MultiIndex.from_tuples([("LEAF","_END"), ("LEAF","_OTHER"),("NODE","_OTHER")])

    start_cols = pd.MultiIndex.from_tuples([("_OTHER", 'L'), ("_OTHER", 'R'),("_OTHER", 'C')])

        

    categories = column.unique()

    counts = dict()

    

    for cat in categories: #build an empty matric for each category (author)

        counts[cat] = pd.SparseDataFrame(index=start_rows, columns=start_cols)

    counts['TOTAL'] = pd.SparseDataFrame(index=start_rows, columns=start_cols)

    for i in column.index:

        tree = trees[i]

        if tree:        

            counts[column[i]] = counts[column[i]].add(get_rels_ch3(tree,stemmer), fill_value=0)

        t2 = time.clock()

        print(i,":",t2 - t1)#for gauging the speed of this function

    for i in categories:

        counts['TOTAL'] = counts['TOTAL'].add(counts[i], fill_value=0)

    return counts
def get_counts2(column, trees,stem = True): #for getting matrices separately

    t1 = time.clock()

    stemmer = None

    if stem: stemmer = nl.stem.snowball.SnowballStemmer("english")  

    counts = []    

    for i in column.index:

        tree = trees[i]

        if tree:

            counts_i = get_rels_ch3(tree,stemmer)

            

            counts.append(counts_i)

        else:

            counts.append(None)

        t2 = time.clock()

        print(i,":",t2 - t1)

    return counts
def get_probs(train_dict,test_list,ids,weights,smoothing=0.5):

    t0=time.clock()

    train_cats = {i:v.to_dense().fillna(0) for (i, v) in train_dict.items() if i != "TOTAL"}#training 'master' matrix associated with each category

    train_total = train_dict["TOTAL"].to_dense().fillna(0)

    

    all_rows = train_total.index

    all_columns = train_total.columns

    

    results = pd.DataFrame(index=ids.index, columns=['id']+[i for i in train_cats],dtype=float)

    

    for (n,test_mat) in enumerate(test_list):#for each matrix to be tested in test list

        cat_probs = pd.Series({i:1.0 for i in train_cats})

        if test_mat is not None:

            test_matrix = test_mat.to_dense().fillna(0)

            nonzeros = test_matrix[test_matrix>0].stack().stack()#convert to a single multi-indexed column listing all nonzero entries

            for (j, count) in nonzeros.iteritems():#for each nonzero value in matrix



                row=(j[0],j[1] if (j[0],j[1]) in all_rows else "_OTHER")



                column = (j[3] if (j[3],j[2]) in all_columns else "_OTHER", j[2])

                column_total_nonzeros = train_total[column][train_total[column] >0].size

                if column_total_nonzeros == 0: #if column is empty for all authors, shift to 'other' column

                    column = ("_OTHER", j[2])

                    column_total_nonzeros = train_total[column][train_total[column] >0].size

                

                for (cat,train_matrix) in train_cats.items():

                    numerator = smoothing



                    denominator = (column_total_nonzeros+1)*smoothing



                    if (column in train_matrix.columns and column_total_nonzeros >=2):

                        denominator = denominator + train_matrix.loc[:,column].sum()

                        if (row in train_matrix.index):

                            numerator = numerator +train_matrix.loc[row,column]



                    cat_probs[cat] = cat_probs[cat]*pow(numerator / denominator,count)

                    

                cat_probs = cat_probs/cat_probs.sum()



        cat_probs=cat_probs * weights       

        results.loc[ids.index[n]] = cat_probs/cat_probs.sum()

        print(n,":",time.clock() - t0)

    

    results['id'] = ids

    

    return results
def probs_regress(train_dict,test_list,ids,weights,smoothing=0.5, regressors={}):

    probs = get_probs(train_dict,test_list,ids,weights,smoothing)

    num_columns = probs.columns.drop('id')

    for i, v in regressors.items():

        probs[i] = regressors[i].predict(probs[i])

    results_sum = probs[num_columns].sum(axis=1)

    probs[num_columns] = probs[num_columns].divide(results_sum, axis=0)

    return probs
def calibrate_reg(train_dict, calib_set,id_calib, weights, smoothing=0.5):

    

    raw_results = probs_regress(train_dict,calib_set,id_calib,weights,smoothing)

    raw_results_n = pd.DataFrame(index = raw_results.index, columns = ['EAP','HPL','MWS'])



    for i,v in raw_results['id'].iteritems():#generates a dataframe of 'correct' results

        conversion = {'EAP':[1,0,0], 'HPL':[0,1,0], 'MWS':[0,0,1]}

        raw_results_n.loc[i,:] =conversion[v]

    

    regressors = {}

    for i,v in raw_results_n.iteritems():

        ir = sk.isotonic.IsotonicRegression(out_of_bounds = 'clip')

        regressors[i] = ir.fit(raw_results[i], raw_results_n[i])

         

    return regressors
import pandas as pd

import nltk as nl

import sklearn as sk

from sklearn import isotonic

import pickle

import time

import re

train_data = pd.read_csv('../input/spooky-author-identification/train.csv')

test_data = pd.read_csv('../input/spooky-author-identification/test.csv')

int_outputs = "../input/intermediate-outputs"



#with open(int_outputs+"/train_trees.p", "rb") as filehandle:

#    train_trees = pickle.load(filehandle)

#with open(int_outputs+"/test_trees.p", "rb") as filehandle:

#    test_trees = pickle.load(filehandle)



#train_matrices_all = get_counts(test_data['id'], test_trees)    

with open(int_outputs+"/test_matrices_all.p", "rb") as filehandle:

    test_matrices_all = pickle.load(filehandle)



#train_matrices_master = get_counts(train_data['author'][:16000], train_trees)    

with open(int_outputs+"/train_matrices_master.p", "rb") as filehandle:

    train_matrices_master = pickle.load(filehandle)



#train_matrices_calib = get_counts2(train_data['author'][16000:18000], train_trees)    

with open(int_outputs+"/train_matrices_calib.p", "rb") as filehandle:

    train_matrices_calib  = pickle.load(filehandle)



#train_matrices_holdout = get_counts2(train_data['author'][18000:], train_trees)

with open(int_outputs+"/train_matrices_holdout.p", "rb") as filehandle:

    train_matrices_holdout = pickle.load(filehandle)



#isotonic_regs = calibrate_reg(train_matrices_master, train_matrices_calib + train_matrices_holdout,train_data['author'][16000:], train_data['author'][16000:].value_counts(), smoothing=0.5)

with open(int_outputs+"/isotonic_regs.p", "rb") as filehandle:

    isotonic_regs = pickle.load(filehandle)

    



    

probs = probs_regress(train_dict=train_matrices_master,

                      test_list = test_matrices_all,

                      ids = test_data['id'],

                      weights = train_data['author'][:16000].value_counts(),

                      smoothing=0.5,

                      regressors=isotonic_regs

                     )    

probs.to_csv('submission.csv',index=False)
probs
def prep2(train_dict, threshold):

    train_cats = {i:v.to_dense().fillna(0) for (i, v) in train_dict.items() if i != "TOTAL"}

    train_total = train_dict["TOTAL"].to_dense().fillna(0)

    

    row_sums=train_total.sum(axis=1)#sum of rows



    all_oth_rows = train_total.loc[row_sums < threshold,:].drop([("LEAF", "_OTHER"),("NODE", "_OTHER")],axis=0)

    

    all_oth_leaf = all_oth_rows.loc['LEAF']

    all_oth_node = all_oth_rows.loc['NODE']

    

    all_oth_leaf_ind = pd.MultiIndex.from_product([["LEAF"],all_oth_leaf.index])

    all_oth_node_ind = pd.MultiIndex.from_product([["NODE"],all_oth_node.index])

      

    train_total.loc[('LEAF', '_OTHER'),:] = all_oth_leaf.sum(axis=0)

    train_total.loc[('NODE', '_OTHER'),:] = all_oth_node.sum(axis=0)

    

    train_total = train_total.drop(all_oth_leaf_ind, axis=0)

    train_total = train_total.drop(all_oth_node_ind, axis=0)

    

    

    all_oth_l_ind = train_total.columns.intersection(pd.MultiIndex.from_product([all_oth_node.index,["L"]]))

    all_oth_r_ind = train_total.columns.intersection(pd.MultiIndex.from_product([all_oth_node.index,["R"]]))

    all_oth_c_ind = train_total.columns.intersection(pd.MultiIndex.from_product([all_oth_node.index,["C"]]))

    

    all_oth_l = train_total.loc[:,all_oth_l_ind]

    all_oth_r = train_total.loc[:,all_oth_r_ind]

    all_oth_c = train_total.loc[:,all_oth_c_ind]

    

    

    train_total.loc[:,('_OTHER', 'L')] = all_oth_l.sum(axis=1)

    train_total.loc[:,('_OTHER', 'R')] = all_oth_r.sum(axis=1)

    train_total.loc[:,('_OTHER', 'C')] = all_oth_c.sum(axis=1)

    

    train_total = train_total.drop(all_oth_l_ind, axis=1)

    train_total = train_total.drop(all_oth_r_ind, axis=1)

    train_total = train_total.drop(all_oth_c_ind, axis=1)

    

    for (i, matrix) in train_cats.items():

        oth_leaf_ind = matrix.index.intersection(all_oth_leaf_ind)

        oth_node_ind = matrix.index.intersection(all_oth_node_ind)

        

        oth_leaf = matrix.loc[oth_leaf_ind,:]

        oth_node = matrix.loc[oth_node_ind,:]

        

        

        train_cats[i].loc[('LEAF', '_OTHER'),:] = oth_leaf.sum(axis=0)

        train_cats[i].loc[('NODE', '_OTHER'),:] = oth_node.sum(axis=0)

        

        train_cats[i] = train_cats[i].drop(oth_leaf_ind, axis=0)

        train_cats[i] = train_cats[i].drop(oth_node_ind, axis=0)

        

        oth_l_ind = matrix.columns.intersection(all_oth_l_ind)

        oth_r_ind = matrix.columns.intersection(all_oth_r_ind)

        oth_c_ind = matrix.columns.intersection(all_oth_c_ind)



        oth_l = matrix.loc[:, oth_l_ind]

        oth_r = matrix.loc[:, oth_r_ind]

        oth_c = matrix.loc[:, oth_c_ind]



        train_cats[i].loc[:,('_OTHER', 'L')] = oth_l.sum(axis=1)

        train_cats[i].loc[:,('_OTHER', 'R')] = oth_r.sum(axis=1)

        train_cats[i].loc[:,('_OTHER', 'C')] = oth_c.sum(axis=1)

        

        

        train_cats[i] = train_cats[i].drop(columns=oth_l_ind)

        train_cats[i] = train_cats[i].drop(columns=oth_r_ind)

        train_cats[i] = train_cats[i].drop(columns=oth_c_ind)

        

        

    return {'TOTAL':train_total, **train_cats}