kaggle_tasks = {"task1": "For COVID-19, What is known about transmission, incubation, and environmental stability?",
                "task2": "For COVID-19, What do we know about COVID-19 risk factors?",
                "task3": "For COVID-19, What do we know about virus genetics, origin, and evolution?",
                "task4": "For COVID-19, What do we know about vaccines and therapeutics?",
                "task5": "For COVID-19, What do we know about non-pharmaceutical interventions?",
                "task6": "For COVID-19, What has been published about medical care?",
                "task7": "For COVID-19, Are there geographic variations in the rate of COVID-19 spread? Are there geographic variations in the mortality rate of COVID-19? there any evidence to suggest geographic based virus mutations?",
                "task8": "For COVID-19, What do we know about diagnostics and surveillance?",
                "task9": "For COVID-19, What has been published about information sharing and inter-sectoral collaboration?",
                "task10": "For COVID-19, What has been published about ethical and social science considerations?"}
import os
from tqdm import tqdm
import pandas as pd
import pickle
def load_metadata_papers(dataset_folder, metadata_file):
    """
    load metadata from metadata.csv
    :param dataset_folder: directory where the metadata is in
    :param metadata_file: raw metadata file name
    :return: metadata dictionary of pairs(key:sha,value: other-info)
    """
    metadata = {}
    metadata_path = dataset_folder + metadata_file

    df = pd.read_csv(metadata_path)
    for index, row in tqdm(df.iterrows(), desc="loading metadata"):
        if row["abstract"] != "Unkown" and not pd.isna(row["abstract"]) and row[
            "abstract"].strip() != "" and not pd.isna(row["sha"]) and not pd.isna(row["title"]):
            metadata[row["sha"]] = {"abstract": row["abstract"], "title": row["title"], "authors": row["authors"],
                                    "journal": row["journal"],
                                    "publish_time": row["publish_time"], "doi": row["doi"]}

    print("loaded:", len(metadata), "instances")
    return metadata
dataset_folder = "../input/CORD-19-research-challenge/"
# load metadata of papers
metadata = load_metadata_papers(dataset_folder, "metadata.csv")
def load_from_save(save_path):
    with open(save_path, 'rb') as pickled:
        return_results = pickle.load(pickled)
    return return_results

insights = load_from_save("../input/pretrained-insights/-results_save.pkl")
import pprint
def get_insights_results(task_name, top_k=20, second_filter=["covid-19"]):
    top_similar_sentences = insights[task_name]
    top_similar_filtered = []
    for each in top_similar_sentences:
        if each[0] in metadata:
            title_cased = metadata[each[0]]["title"].lower()
            abstract_cased = metadata[each[0]]["abstract"].lower()
            instance = {}
            if any(e in title_cased for e in second_filter) or any(e in abstract_cased for e in second_filter):
                instance["title"] = metadata[each[0]]["title"]
                instance["sentence"] = each[1]
                authors = metadata[each[0]]["authors"]
                instance["authors"] = "None" if pd.isna(authors) else authors
                publish_time = metadata[each[0]]["publish_time"]
                instance["publish_time"] = "None" if pd.isna(publish_time) else publish_time
                doi = metadata[each[0]]["doi"] if isinstance(metadata[each[0]]["doi"], str) else str(
                    metadata[each[0]]["doi"])
                instance["doi"] = "https://doi.org/" + doi
                top_similar_filtered.append(instance)
        selected_num = top_k if len(top_similar_filtered) else len(top_similar_filtered)
    return top_similar_filtered[:selected_num]

task_name="task1"
print(f"==================Top 20 insights for {task_name}: {kaggle_tasks[task_name]}==================")
pprint.pprint(get_insights_results(task_name))
task_name="task2"
tok_k=30
print(f"=============Top {tok_k} insights for {task_name}: {kaggle_tasks[task_name]}==================")
pprint.pprint(get_insights_results(task_name,top_k=tok_k))