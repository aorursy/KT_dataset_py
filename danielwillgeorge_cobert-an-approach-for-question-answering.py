!pip install transformers

from datetime import datetime
import string

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('all')
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os
__print__ = print
def print(string):
    os.system(f'echo \"{string}\"')
    __print__(string)
def preprocess():
    """ This method reads article metadata to a Pandas DataFrame.
    
    """
    root_path = '/kaggle/input/CORD-19-research-challenge'
    metadata_path = f'{root_path}/metadata.csv'
    
    meta_df = pd.read_csv(
        metadata_path, 
        dtype={'pubmed_id': str, 'Microsoft Academic Paper ID': str, 'doi': str}, 
        low_memory=False)
    
    df = meta_df[['sha', 'title', 'abstract', 'license', 'publish_time', 'journal', 'url']]
    df.dropna(subset=['sha', 'abstract'], inplace=True)
    
    df.set_index('sha', inplace=True)
    df.index.name = 'paper_id'
    
    return df
def identify_relevant_abstracts(key_word, df):
    """ This method uses GloVe 100d word embeddings to identify the relevant abstracts, which 
        we will explore in more detail to answer questions.
    
    """

    # This container will hold a list of `paper_ids`.
    relevant_abstracts = []

    # retrieve GloVe embeddings    
    glove_file = datapath('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt')
    word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
    glove2word2vec(glove_file, word2vec_glove_file)

    model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
            
    for i, row in df.iterrows():
        abstract = df.loc[i]["abstract"]
        current_index = i

        most_similar_words = model.most_similar(key_word)
        for similar_word in most_similar_words:
            if (similar_word[0] in abstract) and (current_index not in relevant_abstracts):
                relevant_abstracts.append(current_index)

    print("Identified relevant abstracts: %s" % len(relevant_abstracts))
    print(relevant_abstracts)
    return relevant_abstracts
study_types = [
    "Regression",
    "Simulation",
    "Meta-Regression",
    "Systemic",
    "Time-series",
    "Retrospective",
    "Eco-epidemiological",
    "Ecological",
    "Modelling"
]
factors_questions = [
    "Factors of COVID-19?",
    "What causes COVID-19 to decline?",
    "What causes COVID-19 to fall?",
    "What caused COVID-19 to decrease?",
    "What causes COVID-19 spread?",
    "What associations with COVID-19?",
    "What associations COVID-19?",
    "What interventions COVID-19?",
    "What impacts COVID-19?",
    "What correlates with COVID-19?"
]
evidence_questions = [
    "Number of cases?",
    "Number of locations?",
    "Number of countries?",
    "Which countries?"
]
def answer_question_with_model(tokenizer, model, question, text):
    """ This method uses a QA model to answer a question.  Reference:
        https://huggingface.co/transformers/model_doc/albert.html
        
    """
    
    input_dict = tokenizer.encode_plus(question, text, return_tensors="pt")
    input_ids = input_dict["input_ids"].tolist()

    # ALBERT model can support 512 or fewer tokens.
    if len(input_ids[0]) > 512:
        return False

    start_scores, end_scores = model(**input_dict)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = ''.join(all_tokens[torch.argmax(start_scores):torch.argmax(end_scores)+1]).replace('▁', ' ').strip()

    return answer
def answer_parts_of_speech(answer):
    """ This method breaks an answer into parts of speech, to allow us to 
        identify factors, which most often show up as nouns.

    """

    answer_tokens = nltk.word_tokenize(answer)
    answer_parts_of_speech = nltk.pos_tag(answer_tokens)
            
    factors = []
    factor_to_join = []
    
    for word, part_of_speech in answer_parts_of_speech:
        if part_of_speech in ["NN", "NNS"]:
            if len(factor_to_join) == 0:
                factors.append(word)
            else:
                factor_to_join.append(word)
                factors.append(' '.join(factor_to_join))
                factor_to_join = []
        elif part_of_speech in ["JJ"]:
            factor_to_join.append(word)

    return factors
def identify_study_type(title, abstract):
    """ This method identifies the study type of an abstract by pattern matching.
    
    """
    
    study_type = ""
    
    for selection in study_types:
        if selection.lower() in title.lower():
            if selection == "Regression":
                study_type = "Ecological Regression"
                break
            elif selection == "Systemic":
                study_type = "Systemic review"
                break
            elif selection == "Time-series":
                study_type = "Time-series analysis"
                break
            elif selection == "Retrospective":
                study_type = "Retrospective Study"
                break
            elif selection == "Eco-epidemiological":
                study_type = "Eco-epidemiological Study"
                break
            elif selection == "Ecological":
                study_type = "Ecological Study"
                break
            elif selection == "Modelling":
                study_type = "Modelling Study"
                break

    if study_type == "":            
        for selection in study_types:
            if selection.lower() in abstract.lower():
                if selection == "Regression":
                    study_type = "Ecological Regression"
                    break
                elif selection == "Systemic":
                    study_type = "Systemic review"
                    break
                elif selection == "Time-series":
                    study_type = "Time-series analysis"
                    break
                elif selection == "Retrospective":
                    study_type = "Retrospective Study"
                    break
                elif selection == "Eco-epidemiological":
                    study_type = "Eco-epidemiological Study"
                    break
                elif selection == "Ecological":
                    study_type = "Ecological Study"
                    break
                elif selection == "Modelling":
                    study_type = "Modelling Study"
                    break

    if study_type == "":
        study_type = "Retrospective Study"  
        
    return study_type
def generate_summary_table_csv(relevant_abstracts, question, df):
    """ This method generates summary tables corresponding with a question.
    
    """

    # we instantiate the pd.DataFrame which will hold our summary table
    columns = ["Date", "Study", "Study Link", "Journal", "Study Type", "Factors",
                "Influential", "Excerpt", "Measure of Evidence", "Added on"]
    rows = []
    summary_table_df = pd.DataFrame(columns=columns)

    # we instantiate the tokenizer and model for our albert-xlarge-v2
    tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
    model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")

    for paper_id in relevant_abstracts:
        print("Iterating `paper_id` %s" % paper_id)
        abstract = df.loc[paper_id]["abstract"]
        
        # prepare abstract for the model
        abstract = abstract.translate(str.maketrans('', '', string.punctuation))
        abstract = abstract.replace("SARS-CoV-2", "COVID-19")
        abstract = abstract.replace("Covid-19", "COVID-19")

        # identify factors
        factors_retrieved = False
        for fq in factors_questions:
            if factors_retrieved == True:
                continue

            factors_answer = answer_question_with_model(tokenizer, model, fq, abstract)
            
            if (factors_answer == False) or (factors_answer == "[CLS]"):
                continue
            
            factors = answer_parts_of_speech(factors_answer)

            if factors == []:
                continue
                
            factors_retrieved = True

            for factor in factors:
                # first, fill in metadata columns
                publish_time = datetime.strptime(df.loc[paper_id]["publish_time"], '%Y-%m-%d')
                date = publish_time.strftime('%-m/%d/%y')
                title = df.loc[paper_id]["title"]
                study_link = df.loc[paper_id]["url"]
                journal = df.loc[paper_id]["license"]
                added_on = datetime.today().strftime('%-m/%d/%y')                

                # now, we ask whether the factor is influential.
                excerpt_question = """How does %s influence COVID-19?""" % factor
                excerpt = answer_question_with_model(tokenizer, model, excerpt_question, abstract)

                if (excerpt == False) or (excerpt == "[CLS]"):
                    excerpt = "-"

                # Then, if an excerpt can be found, conclude that the factor was influential.
                if (excerpt is not None) and (excerpt != "-"):
                    influential = "Y"
                else:
                    influential = "N"

                # Then, we identify the study type of the abstract.
                study_type = identify_study_type(title, abstract)
                
                # Finally, we identify the evidence if any exists.
                evidence = ""
                for eq in evidence_questions:
                    evidence_answer = answer_question_with_model(tokenizer, model, eq, abstract)                        

                    if (evidence_answer != False) and (evidence_answer != "[CLS]"):
                        if eq == "Number of cases?":
                            evidence = "cases: " + evidence_answer
                        elif eq == "Number of locations?":
                            evidence = "locations: " + evidence_answer
                        elif eq == "Number of countries?":
                            evidence = "countries: " + evidence_answer
                        elif eq == "Which countries?":
                            evidence = "countries: " + evidence_answer
                            
                if evidence == "":
                    evidence = "-"

                df_row = {"Date": date,
                          "Study": title,
                          "Study Link": study_link,
                          "Journal": journal,
                          "Study Type": study_type,
                          "Factors": factor,
                          "Influential": influential,
                          "Excerpt": excerpt,
                          "Measure of Evidence": evidence,
                          "Added on": added_on
                         }
    
                rows.append(df_row)

    summary_table_df = pd.DataFrame(rows, columns=columns)
    print(summary_table_df)
    print("Finalizing summary table csv %s" % question)
    summary_table_df.to_csv("%s-v1.csv" % question, index=False)
def main():
    df = preprocess()

    #relevant_abstracts = identify_relevant_abstracts(key_word, df)

    question_dict = {
        "Seasonality of transmission": ["a789d41d9bafdf73dab3e1a6c90f46c1ce963ff9", "e116dfb0acbbf969bf78e312780ae45e65ac638e","0d11705a07ab7028753be9f85fc714007e2ee841","78b825a616f8756c05ba9af7f8c87572c58ee731","31405dd697c54599864408c6cae1725043d5acd8","9082bff2bab68c199d1ce43d6cfdfc4abe8179fb","1979adc54a27e3dee0ffbf2b08b583bfb9900bb4","84af9c7197860f0aeef586622f26f2fd13d5fbfd","6c0620455fe27bceb7d411f31f7fa05be84bf50c","888c4a8022d2ce985b917103d649420f72bdb349"],
        "How does temperature and humidity affect the transmission of 2019-nCoV": ["e116dfb0acbbf969bf78e312780ae45e65ac638e","0d11705a07ab7028753be9f85fc714007e2ee841","78b825a616f8756c05ba9af7f8c87572c58ee731","9082bff2bab68c199d1ce43d6cfdfc4abe8179fb","31405dd697c54599864408c6cae1725043d5acd8","84af9c7197860f0aeef586622f26f2fd13d5fbfd","6c0620455fe27bceb7d411f31f7fa05be84bf50c","c3c0a8ba2dc4e9f7ca6e4152f3266a1616e1a63f; f8d6e0978748ee23eeaa1eb9c50dc22bed31ea7b","cc7a5fcd4ce8ced4b5005d4ea8d09da2fcdf9f0a; 72565d63479f6c7a483ebfa2ac7b7ef10b021628"],
        "Effectiveness of workplace distancing to prevent secondary transmission": ["b72e843b66eeb54b85568d509994443b5dac047e","11ca9a2c809a5ff5401bbd5e16a2742b5d4d9bd8","9c33486a49de4aea64ce61c0a2c21a88c316b6a8","2913d91f13fd59c698f68ba63008d8e0550c0607"],
        "Effectiveness of school distancing": ["76a1a3f4055df0fd3d7041316d7d8ba48ac98b12","7e65f55efd6ab86bfcbdaf22146c652e47e6f235","33807c0c3367aebc5ed29500a4a9cfba882cce16","11ca9a2c809a5ff5401bbd5e16a2742b5d4d9bd8","a3ec2c34f77f54f03fdc1e60db040ede8a93a03a","78c92c6c7176ea5ca38ddc44462279df3325c4d6"],
        "Effectiveness of inter_inner travel restriction": ["f7ed51444c210f58c010f7d6a8e8ff454520a796","7197c20da00b41eef947e8d0d821a41ad1638f7c","a6bfd3583719947b0790e282d33772593e202011","9f1421f795084d05cda18dcd08dc9bec99fac178","d644cced28a5b2246b394cb5204087c857196e01","0cbe23280cccea688ea36bc5314f3af18148d4ae","abd6288b4399dd34f431fef5ad539a99ddb7ffeb","6e218868d9a3bf4057ccf0be71cd2ac6828a9c76","3d847478e2ff0104ed05c49db2c2e37f75ceece6; 60672300dca1b56b2be5cb96875ef2994dcf4965","7662e461bdac4972293ba461b73f4b7be24cb387"],
        "Effectiveness of community contact reduction": ["a60e5f418229143cbfd6bf2b3f0c53a2ec9d09ae","b301c06e1c036a4b8f2803b8ada254ba227912e6","5262a0c9e3150f9b3e3d33a45a91b9e9cca7da86","a6bfd3583719947b0790e282d33772593e202011","9f1421f795084d05cda18dcd08dc9bec99fac178","0cbe23280cccea688ea36bc5314f3af18148d4ae","3b597f1ae76cfae9a60ca5a13a6353511063956f","abd6288b4399dd34f431fef5ad539a99ddb7ffeb","9f3c081d9cef02ea81a9666a2077639725b65ac8","ab782d7af76c72ab3f5559ebbce93766d799bedf"],
        "Effectiveness of case isolation_isolation of exposed individuals to prevent secondary transmission": ["a60e5f418229143cbfd6bf2b3f0c53a2ec9d09ae","92ea1980ea8bd1105a53e6b8cc132d2448199864","27c7020d5ee9f6d7b2cf92a1990cd56072cc1bc0","6e80241c8b6547c944ca073b224e2bf05064f75d","0cbe23280cccea688ea36bc5314f3af18148d4ae","76a1a3f4055df0fd3d7041316d7d8ba48ac98b12","3b597f1ae76cfae9a60ca5a13a6353511063956f","8f8d59261474f6961ad9b59f0bef8e67b6fc6734","abd6288b4399dd34f431fef5ad539a99ddb7ffeb"],
        "Effectiveness of a multifactorial strategy to prevent secondary transmission": ["b301c06e1c036a4b8f2803b8ada254ba227912e6","f7ed51444c210f58c010f7d6a8e8ff454520a796","a6bfd3583719947b0790e282d33772593e202011","9f1421f795084d05cda18dcd08dc9bec99fac178","6e80241c8b6547c944ca073b224e2bf05064f75d","76a1a3f4055df0fd3d7041316d7d8ba48ac98b12","5a51bc6bfc087af2ae924c899952fc7474a0c4ce","abd6288b4399dd34f431fef5ad539a99ddb7ffeb","7cdd84cbbaa193437e5665afb32b365d75a6077f","6e218868d9a3bf4057ccf0be71cd2ac6828a9c76"]
    }

    for question in question_dict:
        generate_summary_table_csv(question_dict[question], question, df)
main()