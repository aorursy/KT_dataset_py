import numpy as np
import pandas as pd
import glob
import os
import json
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords as nltkstopwords
from nltk.tokenize import word_tokenize
import re
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric, remove_stopwords, strip_multiple_whitespaces
from wordcloud import WordCloud
import matplotlib.pyplot as plt
base_path = "../input/CORD-19-research-challenge"

biorxiv_medrxiv = "biorxiv_medrxiv/biorxiv_medrxiv/pdf_json"
comm_use_subset = "comm_use_subset/comm_use_subset/pdf_json"
noncomm_use_subset = "noncomm_use_subset/noncomm_use_subset/pdf_json"
custom_license = "custom_license/custom_license/pdf_json"
files = []

def read_files(directory):
    files_in_dir = [f for f in glob.glob(os.path.join(base_path, directory) + "/*.json", recursive=True)]
    files.extend(files_in_dir)
    raw_body_texts_of_file = []
    for f in files_in_dir:
        data = json.load(open(f))
        #for key in data.keys():
        #    print("{}: {}\n".format(key, data[key]))
        
        body_text = ""
        for i in range(len(data["body_text"])):
            body_text += " " + data["body_text"][i]["text"]
        
        body_text = re.sub(' +', ' ', body_text)
        raw_body_texts_of_file.append(body_text)
    return raw_body_texts_of_file

raw_body_texts = []
raw_body_texts.extend(read_files(biorxiv_medrxiv))
raw_body_texts.extend(read_files(comm_use_subset))
#raw_body_texts.extend(read_files(noncomm_use_subset))
#raw_body_texts.extend(read_files(custom_license))
def print_title(idx):
    print(json.load(open(files[idx]))["metadata"]["title"])

def print_text(filenumber, start_idx, end_idx):
    text = raw_body_texts[filenumber]
    start_idx = max(0, start_idx)
    end_idx = min(len(text), end_idx)
    print(text[start_idx:end_idx])
    
def print_body_text(filenumber):
    data = json.load(open(files[filenumber]))
    body_text = ""
    for i in range(len(data["body_text"])):
        body_text += " " + data["body_text"][i]["text"]
    body_text = re.sub(' +', ' ', body_text)
    if len(body_text) > 1000:
        print(body_text[:1000])
    else:
        print(body_text)
print("Found {} raw body texts".format(len(raw_body_texts)))
NUMBER_OF_FILES = len(raw_body_texts)
WINDOW_SIZE = 200
factor___risk_pattern = r"(factor(.){0,9}risk)" # for example for "factors of risk"
risk_factor_pattern = r"(risk(.){0,4}factor)" # for example risk factors
risk_pattern = r"(risk)"
high_risk_pattern = r"(high(.){0,6}risk)"
comorbdit_pattern = r"(comorbdit)"
co_infects_pattern = r"(co(.){0,4}infect)"
neonat_pattern = r"(neonat)"
pregnant_pattern = r"(pregnant)"
smoking_pattern = r"(smoking)"
cancer_pattern = r"(cancer)"
averse_outcomes_pattern = r"(advers(.){0,4}outcome)"

PATTERNS = [
    factor___risk_pattern,
    risk_factor_pattern,
    #risk_pattern,
    high_risk_pattern,
    #comorbdit_pattern,
    #co_infects_pattern,
    #neonat_pattern,
    #pregnant_pattern,
    #smoking_pattern,
    #cancer_pattern,
    averse_outcomes_pattern
]

CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation]
CUSTOM_FILTERS_EXCLUDE_NUMERIC = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_numeric]
%%time
def extract_windows_containing(pattern, print_out=True):
    indices = []
    preprocessed_texts = []
    for idx in range(NUMBER_OF_FILES):
        filtered_sentence = raw_body_texts[idx]
        preprocessed_texts.append(filtered_sentence)
        
        indices_of_file = [(m.start(0), m.end(0)) for m in re.finditer(pattern, filtered_sentence)]
        indices.append(indices_of_file)
    
    return indices, preprocessed_texts


indices = [[] for _ in range(NUMBER_OF_FILES)]
for pattern in PATTERNS:
    indices_, preprocessed_texts = extract_windows_containing(pattern)
    for i in range(len(indices_)):
        indices[i].extend(indices_[i])

print("Found {} candidates".format(len([1 for a in indices if len(a)!=0])))
%%time
def process_file(file_number, indices_of_file, filters):
    tokenized_matches = []
    for match in indices_of_file:
        start = match[0]-WINDOW_SIZE
        end = match[1]+WINDOW_SIZE
        text = preprocessed_texts[file_number][start:end]
        tokenized = preprocess_string(text, filters)
        tokenized_matches.append(tokenized)

    return tokenized_matches

tokenized_data = []
for file_number, indices_of_file in enumerate(indices):
    if len(indices_of_file) != 0:
        for data in process_file(file_number, indices_of_file, CUSTOM_FILTERS_EXCLUDE_NUMERIC):
            for word in data:
                if len(word) > 2:
                    tokenized_data.append(word)

nltk_stop_words = set(nltkstopwords.words('english'))
for word in ["high", "risk", "factor", "patients", "factors", "disease"]:
    nltk_stop_words.add(word)
without_stopwords = [word for word in tokenized_data if word not in nltk_stop_words]
print("Number of words without stopwords: {}\n with stopwords {}".format(len(without_stopwords), len(tokenized_data)), end="\n\n")
counts = Counter(without_stopwords)
print(counts.most_common(500))
text = " ".join(without_stopwords)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
POTENTIAL_RISKS = [
    "smoking",
    "pulmonary diseas", 
    "elder",
    "diabetes",
    "old",
    "age",
    "cancer", 
    "cardiac",
    "cardio"]
def doit(only_directly_related_files=False):
    print_counter = 0
    overall_processed_indices_count = 0
    for file_number, indices_of_file in enumerate(indices):
        if only_directly_related_files and file_number in not_directly_related_files:
            continue
        if len(indices_of_file) != 0:
            matches = process_file(file_number, indices_of_file, CUSTOM_FILTERS)
            processed_indices_of_file = []
            indices_of_file = sorted(indices_of_file)
            # adjust indices
            for i in range(len(indices_of_file)):
                if i != 0 and len(processed_indices_of_file) != 0:
                    if abs(processed_indices_of_file[-1][0] - indices_of_file[i][0]) > 100 and abs(processed_indices_of_file[-1][1] - indices_of_file[i][1]) > 100:
                        processed_indices_of_file.append(indices_of_file[i])
                    else:
                        min_ = min(indices_of_file[i][0], processed_indices_of_file[-1][0])
                        max_ = max(indices_of_file[i][1], processed_indices_of_file[-1][1])
                        del processed_indices_of_file[-1]
                        processed_indices_of_file.append((min_, max_))
                else:
                    processed_indices_of_file.append(indices_of_file[i])
            overall_processed_indices_count += len(processed_indices_of_file)

            for index, match in zip(processed_indices_of_file, matches):
                for pattern in POTENTIAL_RISKS:
                    if pattern in match:
                        windows_size = max(int(len(" ".join(match))/2), 300)
                        if print_counter < 10:
                            print("File number: {} index pair: {}".format(file_number, index))
                            print_text(file_number, index[0]-windows_size, index[1]+windows_size)
                            print("\n\n")
                            print_counter += 1
                        break
    return overall_processed_indices_count

overall_processed_indices_count = doit()
print("Total number of possible text passages about risk factors: {}".format(overall_processed_indices_count))
file_number = 239
start_idx = 0
end_idx = 2000

print_text(file_number, start_idx, end_idx)
def is_file_directly_related(filenumber):
    data = json.load(open(files[filenumber]))
    synonyms_for_covid_19 = [
        r"(covid)",
        r"(sars)",
        r"(cov(.){0,4}2)",
        r"(novo)",
        r"(corona)",
    ]
    body_text = ""
    for i in range(len(data["body_text"])):
        body_text += " " + data["body_text"][i]["text"].lower()
    body_text = re.sub(' +', ' ', body_text)
    
    at_least_one_match = False
    for synonym_pattern in synonyms_for_covid_19:
        matches = [(m.start(0), m.end(0)) for m in re.finditer(synonym_pattern, body_text)]
        if len(matches) != 0:
            return True
    return False

not_directly_related_files = []
for i in range(len(files)):
    if not is_file_directly_related(i):
        not_directly_related_files.append(i)
print("Found {} files which might not directly relate to Covid-19".format(len(not_directly_related_files)))
print_title(not_directly_related_files[1])
print_body_text(not_directly_related_files[1])
overall_processed_indices_count = doit(only_directly_related_files=True)
print("Total number of possible text passages about risk factors: {}".format(overall_processed_indices_count))
