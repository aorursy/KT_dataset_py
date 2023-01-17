!ls /kaggle/input/covid-nlp-preprocess/output/
!ls /kaggle/input/covid-nlp-preprocess/output/paragraphs
!ls /kaggle/input/covid-nlp-preprocess/output/whole
!ls /kaggle/input/covid-nlp-preprocess/output/whole/biorxiv_medrxiv | head -n 10
with open("/kaggle/input/covid-nlp-preprocess/output/whole/biorxiv_medrxiv/006df1a5284369a9e2ff2dc7ab267a9f70294d8d.txt", "r") as f:

    text = f.read()

    print(text[:400])
!ls /kaggle/input/CORD-19-research-challenge
import pandas as pd



df_metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

df_metadata.head()
df_metadata[df_metadata["cord_uid"] == "zvrfqkol"]
!ls /kaggle/input/covid-nlp-preprocess/output/paragraphs
!ls /kaggle/input/covid-nlp-preprocess/output/paragraphs/biorxiv_medrxiv | head -n 10
import json



with open("/kaggle/input/covid-nlp-preprocess/output/paragraphs/biorxiv_medrxiv/006df1a5284369a9e2ff2dc7ab267a9f70294d8d.json") as f:

    d = json.load(f)

    print("doc_id: "+d["doc_id"])

    texts = ""

    for paragraph in d["body_text"]:

        paragraph_text = " ".join(paragraph["text"])

        texts += paragraph_text + "\n\n"

    print(texts)
