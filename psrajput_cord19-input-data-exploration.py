import numpy as np, pandas as pd, os, json, glob

from tqdm.notebook import tqdm
# Printing all files using nested for loop (for understanding input structure)



# for dirname, _, filenames in os.walk(('/kaggle/input'):

#     for filename in filenames: 

#         print(os.path.join(dirname, filename))
directory = "/kaggle/input/CORD-19-research-challenge/"
# Folders or Files in Input Data

filesOrFolders = os.listdir(directory)

print (filesOrFolders, "\n\nNumber of Files/Folders in Input Data:" ,len(filesOrFolders))
# Lets read Metadata

with open(directory + "metadata.readme", 'r') as metadata:

    print(metadata.read())
# Listing the number of JSON files in directories & Total

folders = {'nus_pdf': '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/', 

           'nus_pmc': '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pmc_json/',

           'bm_pdf': '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/',

           'cus_pdf': '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/',

           'cus_pmc': '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pmc_json/',

           'cl_pdf': '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/',

           'cl_pmc': '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pmc_json/'

          }



TotalFiles = 0



for folder, location in folders.items():

    print("Number of JSON files in", folder, ":", len(os.listdir(location)))

    TotalFiles += len(os.listdir(location))

    

print("Total JSON files:", TotalFiles)
#Loading all JSONs from each Folder



def loadingJSON(loaded_json, abc):

    items = os.listdir(abc)

    for files in tqdm(items):

        files = abc + files

        file = json.load(open(files))

        loaded_json.append(file)



nus_pdf, nus_pmc, bm_pdf, cus_pdf, cus_pmc, cl_pdf, cl_pmc = [], [], [], [], [], [], [], 



loadingJSON(nus_pdf, folders["nus_pdf"])

loadingJSON(nus_pmc, folders["nus_pmc"])

loadingJSON(bm_pdf, folders["bm_pdf"])

loadingJSON(cus_pdf, folders["cus_pdf"])

loadingJSON(cus_pmc, folders["cus_pmc"])

loadingJSON(cl_pdf, folders["cl_pdf"])

loadingJSON(cl_pmc, folders["cl_pmc"])
# Sorted Keys of the JSONs



folder_list = [nus_pdf[0], nus_pmc[0], bm_pdf[0], cus_pdf[0], cus_pmc[0], cl_pdf[0], cl_pmc[0]]

folder_name = ["NUS PDF", "NUS PMC", "BM PDF", "CUS PDF", "CUS PMC", "CL PDF", "CL PMC"]

    

for index, folder in enumerate(folder_list):

    keys = ', '.join(sorted(list(folder.keys())))

    print(f"{folder_name[index]}:", keys)
# Paper IDs



for i, f in enumerate(folder_list):

    print(f"{folder_name[i]}:", f["paper_id"])
# Back Matter



for i, f in enumerate(folder_list):

    print(f"{folder_name[i]}:", f["back_matter"], "\n" )
# Metadata



for i, f in enumerate(folder_list):

    print(f"{folder_name[i]}:", f["metadata"], "\n" )
# Ref Entries



for i, f in enumerate(folder_list):

    print(f"{folder_name[i]}:", f["ref_entries"], "\n" )
# abstract



print(f"{folder_name[0]}:", folder_list[0]["abstract"], "\n" )

print(f"{folder_name[2]}:", folder_list[2]["abstract"], "\n" )

print(f"{folder_name[3]}:", folder_list[3]["abstract"], "\n" )

print(f"{folder_name[5]}:", folder_list[5]["abstract"], "\n" )
# Body Text



# print(f"{folder_name[0]}:", folder_list[0]["body_text"], "\n" )

# print(f"{folder_name[1]}:", folder_list[1]["body_text"], "\n" )

# print(f"{folder_name[2]}:", folder_list[2]["body_text"], "\n" )

# print(f"{folder_name[3]}:", folder_list[3]["body_text"], "\n" )

# print(f"{folder_name[4]}:", folder_list[4]["body_text"], "\n" )

print(f"{folder_name[5]}:", folder_list[5]["body_text"], "\n" )

# print(f"{folder_name[6]}:", folder_list[6]["body_text"], "\n" )