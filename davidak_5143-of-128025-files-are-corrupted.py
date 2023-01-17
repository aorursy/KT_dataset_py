import os
import json

data_path = "../input/CORD-19-research-challenge/document_parses/"
num_files = 0
num_corrupted = 0

for dir in [ "pdf_json/", "pmc_json/" ]:
    files = os.listdir(data_path + dir)
    num_files += len(files)
    for file in files:
        file_path = data_path + dir + file
        try:
            j = json.load(open(file_path, "rb"))
        except:
            num_corrupted += 1
            #print(f"Erron: Could not open {file_path}")

print(f"{num_corrupted} of {num_files} files are corrupted.")