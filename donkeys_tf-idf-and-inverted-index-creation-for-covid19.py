import os

import numpy as np

import pandas as pd 

from memory_profiler import profile

from typing import List



import kaggle_uploader
!pip install memory_utils
import memory_utils



memory_utils.print_memory()
from tqdm.auto import tqdm

tqdm.pandas()
!ls -l /kaggle/input/covid-nlp-preprocess
!ls -l /kaggle/
!du .
class COVDoc:

    def __init__(self):

        self.doc_id = None

        self.filepath_proc = None

        self.filepath_orig = None

        self.text_proc = None

        self.text_orig = None

        self.tokenized_proc = None

        self.doc_type = None

    

    #this function allows me to lazy-load the original text to save memory

    def load_orig(self):

            with open(self.filepath_orig) as f:

                d = json.load(f)

                body = ""

                for idx, paragraph in enumerate(d["body_text"]):

                    body += f" {paragraph}"

                self.text_orig = body
class DocList:

    doc_list: List[COVDoc] = None

    doc_mode = True

    full_text_mode = False

    #this index will break if multiple iterations at the same time. then need separate object

    iter_idx = 0

    

    def __init__(self):

        self.doc_list = []

        

    def append(self, item:COVDoc):

        self.doc_list.append(item)

    

    def __iter__(self):

        iter_idx = 0

        def doc_iterator(docs):

            for doc in docs:

                #doc_mode = mode where the whole document object is returned

                if self.doc_mode:

                    yield doc

                    continue

                #full_text_mode = mode where the full text is returned in one long string

                if self.full_text_mode:

                    tmp = " ".join(doc.text_proc)

                    yield tmp

                    del tmp

                    continue

                #default mode if not doc_mode or full_text_mode, return whole document as list of words

                yield doc.text_proc

            return 

        return doc_iterator(self.doc_list)

    

    def __len__(self):

        return len(self.doc_list)

    



    
import glob, os, json



paragraphs = []



def load_docs(base_path, base_path_orig, doc_type):

    loaded_docs = DocList()

    file_paths_proc = glob.glob(base_path)

    file_names_proc = [os.path.basename(path) for path in file_paths_proc]

    file_names_orig = [os.path.splitext(filename)[0]+".json" for filename in file_names_proc]

    file_paths_orig = []

    for filename in file_names_orig:

        if filename.startswith("PMC"):

            file_paths_orig.append(os.path.join(base_path_orig, "pmc_json", filename))

        else:

            file_paths_orig.append(os.path.join(base_path_orig, "pdf_json", filename))

#        file_paths_orig = [os.path.join(base_path_orig, filename) for filename in file_names_orig]

    for idx, filepath_proc in enumerate(tqdm(file_paths_proc)):

        doc = COVDoc()

        doc.doc_type = doc_type

        loaded_docs.append(doc)

        doc.filepath_proc = filepath_proc

        doc.filepath_orig = file_paths_orig[idx]

        with open(filepath_proc) as f:

            d = f.read()

            #print(d)

            tokenized = d.strip().split(" ")

            tokenized[0] = tokenized[0].strip()

            doc.doc_id = tokenized[0]

            del tokenized[0]

            doc.text_proc = tokenized

    return loaded_docs
memory_utils.print_memory()
!ls -l /kaggle/input/covid-nlp-preprocess/output/whole
df_metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

df_metadata.head()
med_docs = load_docs("/kaggle/input/covid-nlp-preprocess/output/whole/biorxiv_medrxiv/*.txt", "/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv", "medx")

len(med_docs)
memory_utils.print_memory()
comuse_docs = load_docs("/kaggle/input/covid-nlp-preprocess/output/whole/comm_use_subset/*.txt", "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset", "comm_user")

len(comuse_docs)
memory_utils.print_memory()
noncom_docs = load_docs("/kaggle/input/covid-nlp-preprocess/output/whole/noncomm_use_subset/*.txt", "/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset", "noncomm")

len(noncom_docs)
memory_utils.print_memory()
custom_docs = load_docs("/kaggle/input/covid-nlp-preprocess/output/whole/custom_license/*.txt", "/kaggle/input/CORD-19-research-challenge/custom_license/custom_license", "custom")

len(custom_docs)
memory_utils.print_memory()
all_doc_lists = [med_docs, comuse_docs, noncom_docs, custom_docs]
memory_utils.print_memory()
total_docs = 0

for doc_list in all_doc_lists:

    total_docs += len(doc_list)

total_docs
import collections



def count_words():

    word_count = collections.Counter()

    with tqdm(total=total_docs) as pbar:

        for doc_list in all_doc_lists:

            doc_list.doc_mode = False

            for doc_text in doc_list:

                word_count.update(doc_text)

                pbar.update()

    return word_count



word_count = count_words()

len(word_count)
memory_utils.print_memory()
def redo_docs():

    #all_texts = []

    with tqdm(total=total_docs) as pbar:

        for doc_list in all_doc_lists:

            doc_list.doc_mode = True

            for doc in doc_list:

                text = []

                for token in doc.text_proc:

                    if word_count[token] < 20:

                        splits = token.split("_")

                        if len(splits) > 1:

                            text.extend(splits)

                    else:

                        text.append(token)

                #text = [token for token in doc.text_proc if word_count[token] > 20]

                doc.text_proc = text

                pbar.update()

            #all_texts.extend(" ".join(doc.text_proc) for doc in doc_list)

    #return all_texts



#redo_docs()
memory_utils.print_memory()
#redo_docs()

word_count = count_words()

len(word_count)
memory_utils.print_memory()
all_docs = DocList()

all_docs.doc_mode = False

all_docs.full_text_mode = True

for doc_list in all_doc_lists:

    doc_list.doc_mode = True

    all_docs.doc_list.extend(doc_list)

all_docs.doc_list[0]
from sklearn.feature_extraction.text import TfidfVectorizer



#https://stackoverflow.com/questions/34449127/sklearn-tfidf-transformer-how-to-get-tf-idf-values-of-given-words-in-documen?noredirect=1&lq=1

vect = TfidfVectorizer()

tfidf_matrix = vect.fit_transform(all_docs)

feature_names = vect.get_feature_names()

#TODO: tqdm in custom iterator
memory_utils.print_memory()
def weights_for_doc(doc_idx):

    feature_index = tfidf_matrix[doc_idx , :].nonzero()[1]

    tfidf_scores = zip(feature_index, [tfidf_matrix[doc_idx, x] for x in feature_index])

    return tfidf_scores

word_count["bob"]
threshold_sizes = []

for x in tqdm(range(200)):

    n_of_words = 0

    for word, count in word_count.items():

        if count > x:

            n_of_words += 1

    threshold_sizes.append(n_of_words)

    
threshold_df = pd.DataFrame()

threshold_df["words"] = threshold_sizes

threshold_df.plot()
pd.set_option('display.max_rows', 500)

threshold_df.head(10)

index_threshold = 100
import collections



word_count = collections.Counter()

all_docs.full_text_mode = False

all_docs.doc_mode = False

for doc_text in tqdm(all_docs):

    word_count.update(doc_text)

len(word_count)

all_docs.full_text_mode = False

all_docs.doc_mode = True

all_doc_ids = [doc.doc_id for doc in all_docs]

all_doc_ids[:10]
memory_utils.print_memory()
from collections import defaultdict



i_index = defaultdict(list)



skipped = 0

not_skipped = 0

for idx in tqdm(range(len(all_docs))):

    tfidf_scores = weights_for_doc(idx)

    #weighted_features = []

    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:

        wc = word_count[w]

        if wc < index_threshold:

            #reduce size of index or it will not fit in memory

            skipped += 1

            continue

        not_skipped += 1

        n_s = np.float32(s)

        n_idx = np.uint16(idx)

        i_index[w].append((n_idx, n_s))

#        weighted_features.append((w,s))

    del tfidf_scores

#print(tfidf_scores)

print(f"skipped {skipped} features, used {not_skipped} features")

#TODO: numpy arrays
memory_utils.print_memory()
from operator import itemgetter



count = 0

for word_list in tqdm(i_index.values()):

    word_list.sort(key=itemgetter(1), reverse=True)

    count += 1

memory_utils.print_memory()
for word in tqdm(i_index):

    np_scores = np.array(i_index[word])

    i_index[word] = np_scores
memory_utils.print_memory()
memory_utils.print_memory()
i_index["patient"][:20][0][0]
!mkdir upload_dir
import pickle



with open("upload_dir/tfidf_matrix.pickle", "wb") as f:

    pickle.dump(tfidf_matrix, f)

    

memory_utils.print_memory()



!ls -l upload_dir
with open("upload_dir/feature_names.pickle", "wb") as f:

    pickle.dump(feature_names, f)#



memory_utils.print_memory()



!ls -l upload_dir
with open("upload_dir/i_index.pickle", "wb") as f:

    pickle.dump(i_index, f)



memory_utils.print_memory()



!ls -l upload_dir
with open("upload_dir/doc_ids.pickle", "wb") as f:

    pickle.dump(all_doc_ids, f)



memory_utils.print_memory()



!ls -l upload_dir
import kaggle_uploader



from kaggle_secrets import UserSecretsClient



user_secrets = UserSecretsClient()

api_secret = user_secrets.get_secret("kaggle api key")



kaggle_uploader.resources = []

kaggle_uploader.init_on_kaggle("donkeys", api_secret)

kaggle_uploader.base_path = "./upload_dir"

kaggle_uploader.title = "COVID TF-IDF"

kaggle_uploader.dataset_id = "covid-tfidf"

kaggle_uploader.user_id = "donkeys"

kaggle_uploader.add_resource("doc_ids.pickle", "pickled doc ids for TF-IDF outputs")

kaggle_uploader.add_resource("tfidf_matrix.pickle", "pickled TF-IDF matrix for covid19 dataset")

kaggle_uploader.add_resource("feature_names.pickle", "pickled TF-IDF features names")

kaggle_uploader.add_resource("i_index.pickle", "pickled inverted TF-IDF index for covid19 dataset")

kaggle_uploader.update("new version from kernel")

#kaggle_uploader.update("new version")
