import pandas as pd
import sqlite3

con = sqlite3.connect("../input/database.sqlite")

datasets = ["MNIST", "CIFAR", "SVHN", "PASCAL", "KITTI", "TFD", "SensIT", "Connect4", "Protein", "STL10",
            "adult", "credit", "kr-vs-kp", "promoters", "votes", "UCI", "UCI data", "digg", "HepTH",
            "citeseer", "MovieLens", "RocketFuel", "tweet", "twitter", "bAbI", "TreeBank", "Text8",
            "faces", "SARCOS", "NORB", "TIMIT", "ImageNet", "street", "Street View", "VGG", "Caltech-101",
            "Pascal VOC", "FM-IQA", "AP News", "newsgroups", "diabetes", "HES", "prostate", "MS COCO",
            "Toronto Face", "glaucoma", "Alzheimer’s", "news20", "CURVES", "scleroderma", "dots",
            "puzzle", "MADELON", "ENRON", "WIPO", "reuters"]

total_papers = pd.read_sql_query("SELECT COUNT(Id) TotalPapers FROM Papers", con)["TotalPapers"][0]

def dataset_papers(dataset_name):
    con = sqlite3.connect("../input/database.sqlite")
    sample = pd.read_sql_query("""
    SELECT *
    FROM Papers
    WHERE PaperText LIKE '%% %s %%'""" % dataset_name, con)
    return(sample)

dataset_counts = []

for dataset_name in datasets:
    papers = dataset_papers(dataset_name)
    dataset_counts.append([dataset_name, len(papers), round(100.0*len(papers)/total_papers, 1)])

print(pd.DataFrame(dataset_counts, columns=["Dataset", "NumPapers", "PercentOfPapers"]).sort_values("NumPapers", ascending=False))

import re

term = "Pascal"

papers = dataset_papers(term)

print("Number of papers with the term: %d" % len(papers))

papers["Context"] = ""
for i in range(len(papers)):
    m = re.search(term, papers["PaperText"][i], re.IGNORECASE)
    if m:
        p = m.start()
        papers.loc[i, "Context"] = papers["PaperText"][i][p-50:p+50]

for context in papers["Context"]:
    print(context)
    print("\n\n=======================================================\n\n")

